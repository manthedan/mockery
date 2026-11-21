from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from parakeet_mlx import BaseParakeet, from_pretrained
from parakeet_mlx.audio import get_logmel

from .models import PARAKEET_CHECKPOINTS
from .state_machines import ASR_MODEL_TRANSITIONS, ASRModelState

logger = logging.getLogger(__name__)

CONTROL_TOKEN_RE = re.compile(r"<unk>|<pad>|<\|[^|>]*\|>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ASRTranscription:
    text: str
    inference_ms: float
    audio_duration_ms: float


class ParakeetEngine:
    """
    Thread-safe local Apple Silicon STT engine powered by NVIDIA Parakeet TDT on MLX.
    Explicitly transitions across `ASRModelState` (`idle` -> `loading` -> `ready` <-> `switching` | `error`)
    with a monotonic load generation counter so superseded background loads are discarded.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
        self._model: BaseParakeet | None = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._load_gen: int = 0
        self.state: ASRModelState = ASRModelState.IDLE
        self.error_message: str | None = None
        self.last_inference_ms: float = 0.0

    def _set_state_locked(self, next_state: ASRModelState) -> None:
        if next_state == self.state:
            return
        if next_state not in ASR_MODEL_TRANSITIONS[self.state]:
            raise RuntimeError(f"Illegal ASRModelState transition: {self.state} -> {next_state}")
        self.state = next_state

    def switch_model(self, new_model_id: str) -> None:
        clean_id = new_model_id.strip()
        if not clean_id:
            return
        with self._state_lock:
            if clean_id == self.model_id and self.state in (
                ASRModelState.LOADING,
                ASRModelState.SWITCHING,
                ASRModelState.READY,
            ):
                return
            self._load_gen += 1
            gen = self._load_gen
            self.model_id = clean_id
            next_st = ASRModelState.LOADING if self.state == ASRModelState.IDLE else ASRModelState.SWITCHING
            self._set_state_locked(next_st)
            self.error_message = None
        with self._lock:
            self._model = None
            mx.clear_cache()
        thread = threading.Thread(
            target=self._load_sync,
            args=(gen, clean_id),
            name=f"parakeet-loader-{gen}",
            daemon=True,
        )
        thread.start()

    def start_background_load(self) -> None:
        with self._state_lock:
            if self.state in (ASRModelState.LOADING, ASRModelState.SWITCHING, ASRModelState.READY):
                return
            self._load_gen += 1
            gen = self._load_gen
            target_id = self.model_id
            self._set_state_locked(ASRModelState.LOADING)
            self.error_message = None
        thread = threading.Thread(
            target=self._load_sync,
            args=(gen, target_id),
            name=f"parakeet-loader-{gen}",
            daemon=True,
        )
        thread.start()

    def _load_sync(self, expected_gen: int, target_id: str) -> None:
        try:
            logger.info("Loading Parakeet MLX model: %s (gen=%d) ...", target_id, expected_gen)
            t0 = time.perf_counter()
            model = from_pretrained(target_id, dtype=mx.bfloat16)

            with self._state_lock:
                if self._load_gen != expected_gen:
                    logger.info("Discarding superseded Parakeet load for %s (gen=%d)", target_id, expected_gen)
                    return

            with self._lock:
                dummy = mx.zeros((8000,), dtype=mx.float32)
                mel = get_logmel(dummy, model.preprocessor_config)
                warm = model.generate(mel)
                mx.eval(warm)
                with self._state_lock:
                    if self._load_gen != expected_gen:
                        logger.info("Discarding superseded Parakeet load after warmup (gen=%d)", expected_gen)
                        return
                    self._model = model
                    self._set_state_locked(ASRModelState.READY)
                    self.error_message = None

            elapsed = time.perf_counter() - t0
            logger.info("Parakeet MLX ready in %.2fs (%s)", elapsed, target_id)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load Parakeet MLX model")
            with self._state_lock:
                if self._load_gen == expected_gen:
                    self.error_message = str(exc)
                    self._set_state_locked(ASRModelState.ERROR)

    def ensure_loaded(self) -> BaseParakeet:
        if self._model is not None:
            return self._model
        if self.state in (ASRModelState.IDLE, ASRModelState.ERROR):
            self.start_background_load()
        while self.state in (ASRModelState.LOADING, ASRModelState.SWITCHING):
            time.sleep(0.05)
        if self._model is None:
            raise RuntimeError(f"Parakeet model unavailable: {self.error_message or 'unknown error'}")
        return self._model

    def transcribe_pcm(self, pcm_float32: np.ndarray) -> ASRTranscription:
        model = self.ensure_loaded()
        min_samples = model.preprocessor_config.hop_length * 2
        if pcm_float32.size < min_samples:
            return ASRTranscription(text="", inference_ms=0.0, audio_duration_ms=0.0)

        audio_duration_ms = (pcm_float32.size / model.preprocessor_config.sample_rate) * 1000.0
        pad_samples = int(model.preprocessor_config.sample_rate * 0.2)
        t0 = time.perf_counter()
        with self._lock:
            padded = np.pad(
                np.ascontiguousarray(pcm_float32, dtype=np.float32),
                (pad_samples, pad_samples),
                mode="constant",
            )
            audio_mx = mx.array(padded)
            mel = get_logmel(audio_mx, model.preprocessor_config)
            aligned = model.generate(mel)[0]

        raw_text = CONTROL_TOKEN_RE.sub("", aligned.text)
        clean_text = WHITESPACE_RE.sub(" ", raw_text).strip()
        inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_inference_ms = round(inference_ms, 1)

        return ASRTranscription(
            text=clean_text,
            inference_ms=inference_ms,
            audio_duration_ms=audio_duration_ms,
        )

    def status_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "generation": self._load_gen,
            "model_id": self.model_id,
            "checkpoints": PARAKEET_CHECKPOINTS,
            "last_inference_ms": self.last_inference_ms,
            "error": self.error_message,
        }
