from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from .models import short_id
from .state_machines import VAD_TURN_TRANSITIONS, VADTurnState

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32ms @ 16kHz
CONTEXT_SAMPLES = 64  # Silero VAD v5 context window @ 16kHz
FRAME_MS = 32


@dataclass
class VADEvent:
    utterance_id: str
    rev: int
    kind: Literal["partial", "final"]
    audio: np.ndarray  # float32 1D PCM @ 16kHz
    start_ms: int
    end_ms: int


class SileroVADRuntime:
    """Shared lightweight ONNX runtime for Silero VAD v5 (~2.2MB, CPU-instant)."""

    def __init__(self, repo_id: str = "onnx-community/silero-vad", filename: str = "onnx/model.onnx") -> None:
        try:
            model_path = hf_hub_download(
                repo_id=repo_id, filename=filename, local_files_only=True
            )
        except Exception:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def step(self, frame_with_context: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        out, next_state = self._session.run(
            None,
            {
                "input": frame_with_context,
                "state": state,
                "sr": self._sr,
            },
        )
        return float(out[0, 0]), next_state


class ChannelVADSegmenter:
    """
    Explicit 4-state (`SILENCE` -> `ONSET` -> `SPEAKING` <-> `TRAILING_PAUSE` -> `SILENCE`)
    per-channel streaming VAD segmenter designed for 45+ minute sessions.
    """

    def __init__(
        self,
        runtime: SileroVADRuntime,
        *,
        speech_threshold: float = 0.50,
        silence_threshold: float = 0.35,
        pre_roll_frames: int = 10,
        min_speech_ms: int = 280,
        min_silence_ms: int = 700,
        partial_interval_ms: int = 1250,
        soft_segment_ms: int = 14000,
        soft_silence_ms: int = 280,
        max_segment_ms: int = 24000,
        initial_offset_ms: int = 0,
    ) -> None:
        self._rt = runtime
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.min_silence_frames = max(1, min_silence_ms // FRAME_MS)
        self.partial_interval_frames = max(8, partial_interval_ms // FRAME_MS)
        self.soft_segment_frames = max(50, soft_segment_ms // FRAME_MS)
        self.soft_silence_frames = max(1, soft_silence_ms // FRAME_MS)
        self.max_segment_frames = max(100, max_segment_ms // FRAME_MS)

        self._onnx_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._frame_inp = np.zeros((1, CONTEXT_SAMPLES + FRAME_SAMPLES), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)

        self._turn_state: VADTurnState = VADTurnState.SILENCE
        self._total_frames_processed = max(0, initial_offset_ms // FRAME_MS)
        self._active_utt_id = ""
        self._active_rev = 0
        self._segment_frames: list[np.ndarray] = []
        self._segment_start_frame = 0
        self._speech_frame_count = 0
        self._trailing_silence_frames = 0
        self._frames_since_last_partial = 0

    @property
    def state(self) -> VADTurnState:
        return self._turn_state

    def _transition(self, target: VADTurnState) -> None:
        if target == self._turn_state:
            return
        allowed = VAD_TURN_TRANSITIONS[self._turn_state]
        if target not in allowed:
            raise RuntimeError(f"Illegal VADTurnState transition: {self._turn_state} -> {target}")
        self._turn_state = target

    def push_pcm_int16(self, pcm_bytes: bytes) -> list[VADEvent]:
        if not pcm_bytes:
            return []
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        np.multiply(samples, np.float32(1.0 / 32768.0), out=samples)
        return self.push_float32(samples)

    def push_float32(self, samples: np.ndarray) -> list[VADEvent]:
        if samples.size == 0:
            return []
        if self._pending.size > 0:
            buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        else:
            buf = samples.astype(np.float32, copy=False)

        events: list[VADEvent] = []
        n_frames = buf.size // FRAME_SAMPLES
        offset = 0
        for _ in range(n_frames):
            frame = buf[offset : offset + FRAME_SAMPLES]
            offset += FRAME_SAMPLES
            ev = self._process_frame(frame)
            if ev is not None:
                events.append(ev)

        self._pending = buf[offset:].copy()
        return events

    def _process_frame(self, frame: np.ndarray) -> VADEvent | None:
        inp = self._frame_inp
        inp[0, CONTEXT_SAMPLES:] = frame
        prob, self._onnx_state = self._rt.step(inp, self._onnx_state)
        # Shift tail into context window for next 32ms frame (zero allocations)
        inp[0, :CONTEXT_SAMPLES] = frame[-CONTEXT_SAMPLES:]
        frame_idx = self._total_frames_processed
        self._total_frames_processed += 1

        if self._turn_state == VADTurnState.SILENCE:
            self._pre_roll.append(frame.copy())
            if prob >= self.speech_threshold:
                self._transition(VADTurnState.ONSET)
                self._active_utt_id = short_id("utt")
                self._active_rev = 0
                pre_len = len(self._pre_roll)
                self._segment_start_frame = max(0, frame_idx - pre_len + 1)
                self._segment_frames = list(self._pre_roll)
                self._pre_roll.clear()
                self._speech_frame_count = 1
                self._trailing_silence_frames = 0
                self._frames_since_last_partial = 0
                if self._speech_frame_count >= self.min_speech_frames:
                    self._transition(VADTurnState.SPEAKING)
            return None

        # Active speech turn (`ONSET`, `SPEAKING`, or `TRAILING_PAUSE`)
        self._segment_frames.append(frame.copy())
        self._frames_since_last_partial += 1

        if prob >= self.silence_threshold:
            self._speech_frame_count += 1
            self._trailing_silence_frames = 0
            if self._speech_frame_count >= self.min_speech_frames:
                self._transition(VADTurnState.SPEAKING)
        else:
            self._trailing_silence_frames += 1
            self._transition(VADTurnState.TRAILING_PAUSE)

        seg_len = len(self._segment_frames)
        if (
            self._trailing_silence_frames >= self.min_silence_frames
            or (seg_len >= self.soft_segment_frames and self._trailing_silence_frames >= self.soft_silence_frames)
            or seg_len >= self.max_segment_frames
        ):
            return self._finalize_current_segment(current_frame_idx=frame_idx + 1)

        if (
            self._turn_state == VADTurnState.SPEAKING
            and self._frames_since_last_partial >= self.partial_interval_frames
        ):
            self._frames_since_last_partial = 0
            self._active_rev += 1
            audio = np.concatenate(self._segment_frames)
            return VADEvent(
                utterance_id=self._active_utt_id,
                rev=self._active_rev,
                kind="partial",
                audio=audio,
                start_ms=self._segment_start_frame * FRAME_MS,
                end_ms=(frame_idx + 1) * FRAME_MS,
            )

        return None

    def flush(self) -> VADEvent | None:
        if self._turn_state == VADTurnState.SILENCE:
            return None
        return self._finalize_current_segment(current_frame_idx=self._total_frames_processed)

    def _finalize_current_segment(self, current_frame_idx: int) -> VADEvent | None:
        frames = self._segment_frames
        speech_frames = self._speech_frame_count
        utt_id = self._active_utt_id
        next_rev = self._active_rev + 1
        start_ms = self._segment_start_frame * FRAME_MS
        end_ms = current_frame_idx * FRAME_MS

        self._transition(VADTurnState.SILENCE)
        self._active_utt_id = ""
        self._active_rev = 0
        self._segment_frames = []
        self._speech_frame_count = 0
        self._trailing_silence_frames = 0
        self._frames_since_last_partial = 0

        if speech_frames < self.min_speech_frames or not frames:
            return None

        audio = np.concatenate(frames)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < 1e-4:
            return None

        return VADEvent(
            utterance_id=utt_id,
            rev=next_rev,
            kind="final",
            audio=audio,
            start_ms=start_ms,
            end_ms=end_ms,
        )
