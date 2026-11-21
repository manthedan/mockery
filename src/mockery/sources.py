from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .models import CaptureMode, SpeakerRole
from .vad import ChannelVADSegmenter, SileroVADRuntime, VADEvent

# Binary WebSocket frame channel tags (1st byte of each binary message)
CHANNEL_INTERVIEWER = 0x01  # Local Microphone (Remote Google Meet mode)
CHANNEL_CANDIDATE = 0x02  # Isolated Chrome Tab Audio (Remote Google Meet mode)
CHANNEL_ROOM_MIC = 0x03  # Single Room Microphone (In-Person / Solo Practice mode)

WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class SegmentedSpeech:
    utterance_id: str
    rev: int
    kind: Literal["partial", "final"]
    speaker: SpeakerRole
    audio: np.ndarray
    start_ms: int
    end_ms: int


class SpeakerDiarizer(Protocol):
    """
    Phase 2 Seam: Single-microphone speaker diarization adapter
    (e.g., NVIDIA Nemotron 3 Diarization, Streaming Sortformer, or Pyannote + interviewer voiceprint).

    Note on online vs. offline accuracy: Streaming diarizers occasionally mislabel rapid speaker
    transitions and refine them 1-3s later. Because downstream `InterviewSession.upsert_utterance`
    keys by `(id, rev)`, a Phase 2 diarizer can emit a higher-`rev` update for an earlier `utterance_id`
    to flip `speaker` cleanly in place without duplicating turns or breaking card boundaries.
    """

    def identify_speaker(self, audio: np.ndarray, start_ms: int, end_ms: int) -> SpeakerRole: ...


class ToggleSpeakerDiarizer:
    """Default hotkey-toggleable speaker tagger for Solo Practice / Phase 1 single-mic testing."""

    def __init__(self, initial_speaker: SpeakerRole = "interviewer") -> None:
        self.active_speaker: SpeakerRole = initial_speaker

    def set_speaker(self, speaker: SpeakerRole) -> None:
        self.active_speaker = speaker

    def identify_speaker(self, audio: np.ndarray, start_ms: int, end_ms: int) -> SpeakerRole:
        return self.active_speaker


class CrossChannelBleedFilter:
    """
    Guards against acoustic echo bleed when an interviewer runs without headphones:
    Chrome's `getUserMedia` echo canceller only cancels WebRTC peer audio within the same tab,
    not always audio played by a separate Google Meet tab.
    If an `interviewer` mic utterance overlaps temporally with `candidate` tab speech and >= 70%
    of its words match what the Meet tab just played, we drop the mic segment as speaker bleed.
    """

    def __init__(self, window_ms: int = 12000) -> None:
        self.window_ms = window_ms
        self._recent_candidate: deque[tuple[int, int, set[str]]] = deque()

    def record_candidate_text(self, start_ms: int, end_ms: int, text: str) -> None:
        tokens = set(WORD_RE.findall(text.lower()))
        if not tokens:
            return
        self._recent_candidate.append((start_ms, end_ms, tokens))
        cutoff = end_ms - self.window_ms
        while self._recent_candidate and self._recent_candidate[0][1] < cutoff:
            self._recent_candidate.popleft()

    def is_echo_bleed(self, start_ms: int, end_ms: int, mic_text: str) -> bool:
        mic_tokens = WORD_RE.findall(mic_text.lower())
        if len(mic_tokens) < 3:
            return False
        mic_set = set(mic_tokens)

        overlapping_cand_tokens: set[str] = set()
        for c_start, c_end, c_tokens in self._recent_candidate:
            # Allow +-2500ms acoustic/VAD jitter window
            if start_ms <= (c_end + 2500) and end_ms >= (c_start - 2500):
                overlapping_cand_tokens.update(c_tokens)

        if not overlapping_cand_tokens:
            return False

        overlap_ratio = len(mic_set & overlapping_cand_tokens) / len(mic_set)
        return overlap_ratio >= 0.70


class SessionAudioRouter:
    """
    Normalizes both Remote Dual-Stream (Google Meet) and Single-Mic (In-Person) audio
    into a unified stream of `SegmentedSpeech` events with stable `utterance_id` + `rev`.
    """

    def __init__(
        self,
        vad_runtime: SileroVADRuntime,
        mode: CaptureMode = "remote_meet",
        diarizer: SpeakerDiarizer | None = None,
        initial_offset_ms: int = 0,
    ) -> None:
        self.mode = mode
        self._interviewer_vad = ChannelVADSegmenter(vad_runtime, initial_offset_ms=initial_offset_ms)
        self._candidate_vad = ChannelVADSegmenter(vad_runtime, initial_offset_ms=initial_offset_ms)
        self._room_vad = ChannelVADSegmenter(vad_runtime, initial_offset_ms=initial_offset_ms)
        self._toggle_diarizer = ToggleSpeakerDiarizer()
        self._diarizer: SpeakerDiarizer = diarizer or self._toggle_diarizer
        self.bleed_filter = CrossChannelBleedFilter()

    def set_manual_speaker(self, speaker: SpeakerRole) -> list[SegmentedSpeech]:
        """
        Switches the single-mic manual speaker label, flushing any active room-mic segment
        under the PREVIOUS speaker first so rapid Tab toggles never misattribute an in-progress turn.
        """
        flushed: list[SegmentedSpeech] = []
        prev_speaker = self._toggle_diarizer.active_speaker
        if speaker != prev_speaker:
            room_ev = self._room_vad.flush()
            if room_ev is not None:
                flushed.append(
                    SegmentedSpeech(
                        utterance_id=room_ev.utterance_id,
                        rev=room_ev.rev,
                        kind="final",
                        speaker=prev_speaker,
                        audio=room_ev.audio,
                        start_ms=room_ev.start_ms,
                        end_ms=room_ev.end_ms,
                    )
                )
        self._toggle_diarizer.set_speaker(speaker)
        return flushed

    @property
    def manual_speaker(self) -> SpeakerRole:
        return self._toggle_diarizer.active_speaker

    def ingest_binary_frame(self, payload: bytes) -> list[SegmentedSpeech]:
        if len(payload) < 3:
            return []
        channel_id = payload[0]
        pcm_bytes = payload[1:]
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]

        if channel_id == CHANNEL_INTERVIEWER:
            return self._wrap_events(self._interviewer_vad.push_pcm_int16(pcm_bytes), "interviewer")
        if channel_id == CHANNEL_CANDIDATE:
            return self._wrap_events(self._candidate_vad.push_pcm_int16(pcm_bytes), "candidate")
        if channel_id == CHANNEL_ROOM_MIC:
            events = self._room_vad.push_pcm_int16(pcm_bytes)
            return [
                SegmentedSpeech(
                    utterance_id=ev.utterance_id,
                    rev=ev.rev,
                    kind=ev.kind,
                    speaker=self._diarizer.identify_speaker(ev.audio, ev.start_ms, ev.end_ms),
                    audio=ev.audio,
                    start_ms=ev.start_ms,
                    end_ms=ev.end_ms,
                )
                for ev in events
            ]
        return []

    def flush_all(self) -> list[SegmentedSpeech]:
        out: list[SegmentedSpeech] = []
        for role, seg in (("interviewer", self._interviewer_vad), ("candidate", self._candidate_vad)):
            ev = seg.flush()
            if ev is not None:
                out.append(
                    SegmentedSpeech(
                        utterance_id=ev.utterance_id,
                        rev=ev.rev,
                        kind="final",
                        speaker=role,
                        audio=ev.audio,
                        start_ms=ev.start_ms,
                        end_ms=ev.end_ms,
                    )
                )
        room_ev = self._room_vad.flush()
        if room_ev is not None:
            role = self._diarizer.identify_speaker(room_ev.audio, room_ev.start_ms, room_ev.end_ms)
            out.append(
                SegmentedSpeech(
                    utterance_id=room_ev.utterance_id,
                    rev=room_ev.rev,
                    kind="final",
                    speaker=role,
                    audio=room_ev.audio,
                    start_ms=room_ev.start_ms,
                    end_ms=room_ev.end_ms,
                )
            )
        out.sort(key=lambda s: s.start_ms)
        return out

    @staticmethod
    def _wrap_events(events: list[VADEvent], speaker: SpeakerRole) -> list[SegmentedSpeech]:
        return [
            SegmentedSpeech(
                utterance_id=ev.utterance_id,
                rev=ev.rev,
                kind=ev.kind,
                speaker=speaker,
                audio=ev.audio,
                start_ms=ev.start_ms,
                end_ms=ev.end_ms,
            )
            for ev in events
        ]
