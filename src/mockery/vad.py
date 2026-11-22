from __future__ import annotations

from dataclasses import dataclass
import threading
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
    """Shared lightweight ONNX runtime for Silero VAD v5 (~2.2MB, CPU-instant, lazy-loaded)."""

    def __init__(
        self,
        repo_id: str = "onnx-community/silero-vad",
        filename: str = "onnx/model.onnx",
        *,
        eager: bool = False,
    ) -> None:
        self._repo_id = repo_id
        self._filename = filename
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._error: str | None = None
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        if eager:
            self.ensure_loaded()

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def error_message(self) -> str | None:
        return self._error

    def status_dict(self) -> dict[str, object]:
        return {
            "ready": self._session is not None,
            "error": self._error,
        }

    def ensure_loaded(self) -> ort.InferenceSession:
        sess = self._session
        if sess is not None:
            return sess
        with self._lock:
            if self._session is not None:
                return self._session
            try:
                try:
                    model_path = hf_hub_download(
                        repo_id=self._repo_id,
                        filename=self._filename,
                        local_files_only=True,
                    )
                except Exception:
                    model_path = hf_hub_download(
                        repo_id=self._repo_id,
                        filename=self._filename,
                    )
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._error = None
                return self._session
            except Exception as exc:
                self._error = str(exc)
                raise

    def step(self, frame_with_context: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        session = self.ensure_loaded()
        out, next_state = session.run(
            None,
            {
                "input": frame_with_context,
                "state": state,
                "sr": self._sr,
            },
        )
        return float(out[0, 0]), next_state


_EMPTY_F32 = np.empty(0, dtype=np.float32)


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
        self._pending = _EMPTY_F32

        # Preallocated ring buffer for silence pre-roll (zero allocations per 32ms frame)
        self._pre_roll_cap = max(0, pre_roll_frames)
        self._pre_roll_buf = np.empty((self._pre_roll_cap, FRAME_SAMPLES), dtype=np.float32)
        self._pre_roll_write = 0
        self._pre_roll_count = 0

        # Preallocated contiguous segment buffer (eliminates per-frame copies + O(N^2) np.concatenate)
        self._seg_cap_frames = self.max_segment_frames + self._pre_roll_cap + 2
        self._seg_buf = np.empty(self._seg_cap_frames * FRAME_SAMPLES, dtype=np.float32)
        self._seg_frame_count = 0

        self._turn_state: VADTurnState = VADTurnState.SILENCE
        self._total_frames_processed = max(0, initial_offset_ms // FRAME_MS)
        self._active_utt_id = ""
        self._active_rev = 0
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

    def _append_seg_frame(self, frame: np.ndarray) -> None:
        idx = self._seg_frame_count
        if idx >= self._seg_cap_frames:
            new_cap = max(self._seg_cap_frames * 2, idx + 16)
            grown = np.empty(new_cap * FRAME_SAMPLES, dtype=np.float32)
            used = idx * FRAME_SAMPLES
            grown[:used] = self._seg_buf[:used]
            self._seg_buf = grown
            self._seg_cap_frames = new_cap
        start = idx * FRAME_SAMPLES
        self._seg_buf[start : start + FRAME_SAMPLES] = frame
        self._seg_frame_count = idx + 1

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

        self._pending = buf[offset:].copy() if offset < buf.size else _EMPTY_F32
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
            if self._pre_roll_cap > 0:
                self._pre_roll_buf[self._pre_roll_write] = frame
                self._pre_roll_write = (self._pre_roll_write + 1) % self._pre_roll_cap
                if self._pre_roll_count < self._pre_roll_cap:
                    self._pre_roll_count += 1
            if prob >= self.speech_threshold:
                self._transition(VADTurnState.ONSET)
                self._active_utt_id = short_id("utt")
                self._active_rev = 0
                pre_len = self._pre_roll_count
                self._segment_start_frame = max(0, frame_idx - pre_len + 1)
                if pre_len > 0:
                    start_slot = (self._pre_roll_write - pre_len) % self._pre_roll_cap
                    if start_slot + pre_len <= self._pre_roll_cap:
                        n_samp = pre_len * FRAME_SAMPLES
                        self._seg_buf[:n_samp] = self._pre_roll_buf[
                            start_slot : start_slot + pre_len
                        ].reshape(-1)
                    else:
                        first_n = self._pre_roll_cap - start_slot
                        first_samp = first_n * FRAME_SAMPLES
                        rem_n = pre_len - first_n
                        rem_samp = rem_n * FRAME_SAMPLES
                        self._seg_buf[:first_samp] = self._pre_roll_buf[start_slot:].reshape(-1)
                        self._seg_buf[first_samp : first_samp + rem_samp] = self._pre_roll_buf[
                            :rem_n
                        ].reshape(-1)
                self._seg_frame_count = pre_len
                self._pre_roll_write = 0
                self._pre_roll_count = 0
                self._speech_frame_count = 1
                self._trailing_silence_frames = 0
                self._frames_since_last_partial = 0
                if self._speech_frame_count >= self.min_speech_frames:
                    self._transition(VADTurnState.SPEAKING)
            return None

        # Active speech turn (`ONSET`, `SPEAKING`, or `TRAILING_PAUSE`)
        self._append_seg_frame(frame)
        self._frames_since_last_partial += 1

        if prob >= self.silence_threshold:
            self._speech_frame_count += 1
            self._trailing_silence_frames = 0
            if self._speech_frame_count >= self.min_speech_frames:
                self._transition(VADTurnState.SPEAKING)
        else:
            self._trailing_silence_frames += 1
            self._transition(VADTurnState.TRAILING_PAUSE)

        seg_len = self._seg_frame_count
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
            audio = self._seg_buf[: seg_len * FRAME_SAMPLES].copy()
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
        seg_frames = self._seg_frame_count
        speech_frames = self._speech_frame_count
        utt_id = self._active_utt_id
        next_rev = self._active_rev + 1
        start_ms = self._segment_start_frame * FRAME_MS
        end_ms = current_frame_idx * FRAME_MS

        self._transition(VADTurnState.SILENCE)
        self._active_utt_id = ""
        self._active_rev = 0
        self._seg_frame_count = 0
        self._speech_frame_count = 0
        self._trailing_silence_frames = 0
        self._frames_since_last_partial = 0

        if speech_frames < self.min_speech_frames or seg_frames == 0:
            return None

        view = self._seg_buf[: seg_frames * FRAME_SAMPLES]
        rms = float(np.sqrt(float(np.dot(view, view)) / view.size))
        if rms < 1e-4:
            return None

        return VADEvent(
            utterance_id=utt_id,
            rev=next_rev,
            kind="final",
            audio=view.copy(),
            start_ms=start_ms,
            end_ms=end_ms,
        )
