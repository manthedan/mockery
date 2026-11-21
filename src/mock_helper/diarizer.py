from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import SpeakerRole

logger = logging.getLogger("mock_helper.diarizer")

SAMPLE_RATE = 16000
DEFAULT_WESPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
DEFAULT_WESPEAKER_FILE = "voxceleb_resnet34_LM.onnx"


@dataclass
class DiarizedSubSegment:
    speaker: SpeakerRole
    audio: np.ndarray
    start_ms: int
    end_ms: int
    confidence: float
    embedding: np.ndarray | None = None


def _build_kaldi_mel_filterbank(
    *,
    sr: int = SAMPLE_RATE,
    n_fft: int = 512,
    n_mels: int = 80,
    low_freq: float = 20.0,
    high_freq: float = 8000.0,
) -> np.ndarray:
    def hz2mel(f: np.ndarray | float) -> np.ndarray | float:
        return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    mels = np.linspace(float(hz2mel(low_freq)), float(hz2mel(high_freq)), n_mels + 2)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fft_mels = np.asarray(hz2mel(fft_freqs), dtype=np.float64)
    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = mels[m], mels[m + 1], mels[m + 2]
        up = (fft_mels - left) / max(1e-6, center - left)
        down = (right - fft_mels) / max(1e-6, right - center)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb


_MEL_FB_80 = _build_kaldi_mel_filterbank()
_HAMMING_WIN_400 = np.hamming(400).astype(np.float32)[None, :]
_FBANK_EPS = np.float32(np.finfo(np.float32).eps)
_FBANK_FRAME_LEN = 400
_FBANK_FRAME_SHIFT = 160


def _extract_kaldi_fbank80_raw(audio_f32: np.ndarray) -> np.ndarray:
    """
    Computes un-normalized (pre-CMN) 80-bin Kaldi log-mel filterbank frames matching the
    WeSpeaker ONNX reference (Hamming window, mel-space triangular weights, float32 eps floor)
    using zero-copy `sliding_window_view` framing.
    """
    sig = np.asarray(audio_f32, dtype=np.float32)
    if sig.ndim > 1:
        sig = sig.mean(axis=-1)
    if len(sig) < _FBANK_FRAME_LEN:
        sig = np.pad(sig, (0, _FBANK_FRAME_LEN - len(sig)))
    # Zero-copy strided frame view (avoids allocating a giant int64 index table)
    strided = np.lib.stride_tricks.sliding_window_view(sig, _FBANK_FRAME_LEN)[::_FBANK_FRAME_SHIFT]
    # Scale to Kaldi PCM16 amplitude in one contiguous float32 copy
    frames = strided * np.float32(32768.0)
    frames -= frames.mean(axis=1, keepdims=True)
    # In-place pre-emphasis (0.97)
    pre = np.empty_like(frames)
    pre[:, 0] = frames[:, 0] * np.float32(0.03)
    np.subtract(frames[:, 1:], frames[:, :-1] * np.float32(0.97), out=pre[:, 1:])
    pre *= _HAMMING_WIN_400
    spec = np.fft.rfft(pre, n=512, axis=1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    mel_energy = np.maximum(power @ _MEL_FB_80, _FBANK_EPS)
    return np.log(mel_energy).astype(np.float32)


def extract_kaldi_fbank80(audio_f32: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy Kaldi-compatible 80-bin log-mel filterbank with per-utterance CMN,
    matching WeSpeaker VoxCeleb ResNet34-LM training features (16kHz, 25ms/10ms).
    """
    feats = _extract_kaldi_fbank80_raw(audio_f32)
    feats -= feats.mean(axis=0, keepdims=True)
    return feats


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _select_wespeaker_providers(ort_module: Any) -> list[str]:
    import os

    override = os.getenv("WESPEAKER_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    available = set(ort_module.get_available_providers())
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class WeSpeakerEmbedder:
    """
    Local ONNX WeSpeaker ResNet34-LM voice embedding extractor (256-d L2-normalized vectors).
    Thread-safe, lazily downloaded once from Hugging Face (~26.5 MB, Apache 2.0) and cached.
    Uses Apple Silicon CoreML acceleration when available (~13x faster batched windows) with
    automatic CPU fallback.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_WESPEAKER_REPO,
        filename: str = DEFAULT_WESPEAKER_FILE,
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self._session: Any | None = None
        self._active_provider: str = "unloaded"
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._loading = False
        self._error: str | None = None
        self.last_inference_ms: float = 0.0

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def error(self) -> str | None:
        return self._error

    def start_background_load(self) -> None:
        if self._session is not None or self._loading:
            return
        self._loading = True

        def _worker() -> None:
            try:
                self.ensure_loaded()
            except Exception as exc:
                self._error = str(exc)
                logger.warning("WeSpeaker background load failed: %s", exc)
            finally:
                self._loading = False

        threading.Thread(target=_worker, daemon=True, name="wespeaker-loader").start()

    def ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            t0 = time.perf_counter()
            try:
                model_path = hf_hub_download(
                    self.repo_id, self.filename, local_files_only=True
                )
            except Exception:
                model_path = hf_hub_download(self.repo_id, self.filename)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3

            providers = _select_wespeaker_providers(ort)
            try:
                sess = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=providers,
                )
            except Exception as exc:
                if providers != ["CPUExecutionProvider"]:
                    logger.warning(
                        "WeSpeaker CoreML init failed (%s); falling back to CPUExecutionProvider",
                        exc,
                    )
                    sess = ort.InferenceSession(
                        model_path,
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    raise

            self._session = sess
            active_provs = sess.get_providers()
            self._active_provider = active_provs[0] if active_provs else "CPUExecutionProvider"
            self._error = None
            logger.info(
                "Loaded WeSpeaker ONNX (%s, provider=%s) in %.0fms",
                self.filename,
                self._active_provider,
                (time.perf_counter() - t0) * 1000.0,
            )

    def embed_one(self, audio_f32: np.ndarray) -> np.ndarray:
        return self.embed_batch([audio_f32])[0]

    def _run_batch_feats(self, batch: np.ndarray) -> np.ndarray:
        assert self._session is not None
        t0 = time.perf_counter()
        with self._infer_lock:
            embs = self._session.run(["embs"], {"feats": batch})[0]
        self.last_inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return (embs / norms).astype(np.float32)

    def embed_sliding_windows(
        self,
        audio_f32: np.ndarray,
        starts: list[int],
        win_n: int,
    ) -> np.ndarray:
        """
        Extracts L2-normalized [len(starts), 256] embeddings for overlapping equal-length
        windows of `audio_f32`. Computes the Kaldi log-mel filterbank ONCE across the full
        segment and slices frame-aligned windows with exact per-window CMN.
        """
        if not starts:
            return np.zeros((0, 256), dtype=np.float32)
        self.ensure_loaded()
        win_frames = 1 + (max(win_n, _FBANK_FRAME_LEN) - _FBANK_FRAME_LEN) // _FBANK_FRAME_SHIFT
        raw_full = _extract_kaldi_fbank80_raw(audio_f32)
        batch = np.empty((len(starts), win_frames, 80), dtype=np.float32)
        for i, st in enumerate(starts):
            if st % _FBANK_FRAME_SHIFT == 0:
                f0 = st // _FBANK_FRAME_SHIFT
                if f0 + win_frames <= raw_full.shape[0]:
                    w = raw_full[f0 : f0 + win_frames]
                    batch[i] = w - w.mean(axis=0, keepdims=True)
                    continue
            batch[i] = extract_kaldi_fbank80(audio_f32[st : st + win_n])
        return self._run_batch_feats(batch)

    def embed_batch(self, chunks: list[np.ndarray]) -> np.ndarray:
        """
        Extracts L2-normalized [N, 256] voice embeddings. All chunks in a single batch call
        must have identical length (or will be truncated/padded to the longest chunk in the batch).
        """
        if not chunks:
            return np.zeros((0, 256), dtype=np.float32)
        self.ensure_loaded()

        feats_list = [extract_kaldi_fbank80(c) for c in chunks]
        max_t = max(f.shape[0] for f in feats_list)
        if all(f.shape[0] == max_t for f in feats_list):
            batch = np.stack(feats_list, axis=0)
        else:
            batch = np.zeros((len(feats_list), max_t, 80), dtype=np.float32)
            for i, f in enumerate(feats_list):
                batch[i, : f.shape[0], :] = f

        return self._run_batch_feats(batch)

    def status_dict(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "loading": self._loading,
            "model": self.repo_id,
            "provider": self._active_provider,
            "last_inference_ms": self.last_inference_ms,
            "error": self._error,
        }


def _active_speech_sec(audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Gain-invariant estimate of active voiced speech duration (in seconds) within a PCM clip.
    Uses 20ms frame RMS relative to the clip's own 90th-percentile energy (with a floor for
    pure room hiss) so both near and far room speakers pass while sparse 1-word + silence
    windows (e.g. 'Let's' + 1.2s pause) are accurately measured as short.
    """
    arr = np.asarray(audio_f32, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    rms_all = float(np.sqrt(np.mean(arr * arr) + 1e-12))
    if rms_all < 0.003:
        return 0.0
    frame_n = int(0.02 * sr)  # 20ms
    n_frames = len(arr) // frame_n
    if n_frames <= 0:
        return len(arr) / float(sr) if rms_all >= 0.005 else 0.0
    frames = arr[: n_frames * frame_n].reshape(n_frames, frame_n)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    p90 = float(np.percentile(frame_rms, 90))
    if p90 < 0.004:
        return 0.0
    thresh = max(0.003, 0.20 * p90)
    return float(np.sum(frame_rms >= thresh)) * 0.02


def _farthest_pair_kmeans2(
    embs: np.ndarray,
    *,
    iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Deterministic 2-means clustering on L2-normalized embeddings initialized from the
    minimum-cosine-similarity (farthest) pair.
    Returns (labels_0_or_1, c0, c1, centroid_cosine_sim).
    """
    sim_mat = embs @ embs.T
    min_flat = int(np.argmin(sim_mat))
    i0, i1 = divmod(min_flat, len(embs))
    c0 = embs[i0].copy()
    c1 = embs[i1].copy()
    labels = np.zeros(len(embs), dtype=np.int32)
    for _ in range(iters):
        s0 = embs @ c0
        s1 = embs @ c1
        new_labels = (s1 > s0).astype(np.int32)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        if np.any(labels == 0):
            c0 = _l2_normalize(embs[labels == 0].mean(axis=0))
        if np.any(labels == 1):
            c1 = _l2_normalize(embs[labels == 1].mean(axis=0))
    return labels, c0, c1, float(c0 @ c1)


class OnlineTwoSpeakerDiarizer:
    """
    Zero-enrollment online 2-speaker acoustic voice classifier + mid-segment handoff splitter
    for single-mic (`in_person`) capture.

    - Only cleanly voiced speech (measured via gain-invariant 20ms active-speech duration)
      can seed or update speaker centroids, preventing sparse 1-word/silence lead-ins from
      poisoning speaker 1.
    - First voiced utterance seeds the primary speaker (`default_speaker`, usually `interviewer`).
    - First voiced speech whose cosine similarity drops below `new_speaker_sim_threshold` (0.34)
      automatically seeds the second speaker (`candidate`).
    - Long segments (>= 2.8s) run a fast batched sliding-window voice check (1.5s window, 0.6s hop)
      to detect rapid <700ms speaker handoffs and split PCM cleanly at the lowest-RMS energy trough.
    - Clicking a speaker badge in the UI calls `record_manual_correction(utt_id, speaker)` so
      centroids immediately adapt to user feedback.
    """

    def __init__(
        self,
        embedder: WeSpeakerEmbedder,
        *,
        new_speaker_sim_threshold: float = 0.34,
        switch_margin: float = 0.14,
        window_sec: float = 1.5,
        hop_sec: float = 0.6,
        min_split_segment_sec: float = 2.8,
        min_update_dur_sec: float = 1.0,
        min_window_voiced_sec: float = 0.55,
        min_seed_voiced_sec: float = 0.65,
    ) -> None:
        self.embedder = embedder
        self.new_speaker_sim_threshold = new_speaker_sim_threshold
        self.switch_margin = switch_margin
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.min_split_segment_sec = min_split_segment_sec
        self.min_update_dur_sec = min_update_dur_sec
        self.min_window_voiced_sec = min_window_voiced_sec
        self.min_seed_voiced_sec = min_seed_voiced_sec

        self._lock = threading.Lock()
        self._centroids: dict[SpeakerRole, np.ndarray | None] = {
            "interviewer": None,
            "candidate": None,
        }
        self._counts: dict[SpeakerRole, int] = {"interviewer": 0, "candidate": 0}
        self._last_speaker: SpeakerRole = "interviewer"
        self._first_speaker: SpeakerRole = "interviewer"
        self._manual_override: bool = False
        # Clean voiced turn history for cold-start self-healing: (emb, voiced_sec, speaker)
        self._voiced_samples: list[tuple[np.ndarray, float, SpeakerRole]] = []
        # utt_id -> (embedding, dur_sec, speaker) for manual correction feedback
        self._utt_history: dict[str, tuple[np.ndarray, float, SpeakerRole]] = {}

    def reset(self) -> None:
        with self._lock:
            self._centroids = {"interviewer": None, "candidate": None}
            self._counts = {"interviewer": 0, "candidate": 0}
            self._last_speaker = "interviewer"
            self._first_speaker = "interviewer"
            self._manual_override = False
            self._voiced_samples.clear()
            self._utt_history.clear()

    @property
    def has_manual_override(self) -> bool:
        with self._lock:
            return self._manual_override

    def swap_roles(self, *, mark_manual: bool = False) -> bool:
        """
        Atomically swaps the acoustic centroids and historical speaker bindings between
        `interviewer` and `candidate` (e.g. when the candidate spoke first at t=00:00 and
        seeded the primary cluster under the inverted role label).
        """
        def _flip(role: SpeakerRole) -> SpeakerRole:
            return "candidate" if role == "interviewer" else "interviewer"

        with self._lock:
            has_any = any(v is not None for v in self._centroids.values())
            self._centroids["interviewer"], self._centroids["candidate"] = (
                self._centroids["candidate"],
                self._centroids["interviewer"],
            )
            self._counts["interviewer"], self._counts["candidate"] = (
                self._counts["candidate"],
                self._counts["interviewer"],
            )
            self._last_speaker = _flip(self._last_speaker)
            self._first_speaker = _flip(self._first_speaker)
            self._voiced_samples = [
                (emb, v_s, _flip(spk)) for emb, v_s, spk in self._voiced_samples
            ]
            self._utt_history = {
                uid: (emb, dur, _flip(spk))
                for uid, (emb, dur, spk) in self._utt_history.items()
            }
            if mark_manual:
                self._manual_override = True
            logger.info(
                "Swapped diarizer speaker roles (manual=%s, counts=%s)",
                mark_manual,
                dict(self._counts),
            )
            return has_any

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            enrolled = [k for k, v in self._centroids.items() if v is not None]
            sep = None
            ci, cc = self._centroids["interviewer"], self._centroids["candidate"]
            if ci is not None and cc is not None:
                sep = round(float(ci @ cc), 3)
            return {
                **self.embedder.status_dict(),
                "enrolled_speakers": enrolled,
                "centroid_cosine_sim": sep,
            }

    def classify_partial(
        self,
        audio_f32: np.ndarray,
        *,
        default_speaker: SpeakerRole = "interviewer",
    ) -> SpeakerRole:
        """Non-mutating fast speaker read for live draft partials (~25ms, only if model already loaded)."""
        if not self.embedder.is_ready or len(audio_f32) < int(0.45 * SAMPLE_RATE):
            return default_speaker
        # Inspect the most recent 2.0s so live partials react immediately when the other person starts talking
        tail = audio_f32[-int(2.0 * SAMPLE_RATE) :]
        if _active_speech_sec(tail) < 0.35:
            with self._lock:
                return self._last_speaker
        try:
            emb = self.embedder.embed_one(tail)
        except Exception:
            return default_speaker
        with self._lock:
            spk, _conf, _si, _sc = self._score_embedding_locked(
                emb, default_speaker=default_speaker, allow_seed=False
            )
            return spk

    def diarize_segment(
        self,
        audio_f32: np.ndarray,
        start_ms: int,
        end_ms: int,
        *,
        default_speaker: SpeakerRole = "interviewer",
    ) -> list[DiarizedSubSegment]:
        """
        Classifies a finalized single-mic VAD segment and splits it into single-speaker PCM
        sub-segments if a rapid mid-segment speaker handoff is detected acoustically.
        """
        arr = np.asarray(audio_f32, dtype=np.float32)
        if arr.size == 0:
            return []
        dur_s = len(arr) / float(SAMPLE_RATE)
        voiced_s = _active_speech_sec(arr)

        # Very short utterance (< 2.8s): single fast embedding
        if dur_s < self.min_split_segment_sec:
            padded = arr if len(arr) >= 4000 else np.pad(arr, (0, 4000 - len(arr)))
            emb = self.embedder.embed_one(padded)
            with self._lock:
                if not self._voiced_samples and self._centroids["interviewer"] is None and self._centroids["candidate"] is None:
                    self._first_speaker = default_speaker
                spk, conf, si, sc = self._score_embedding_locked(
                    emb,
                    default_speaker=default_speaker,
                    allow_seed=(voiced_s >= self.min_seed_voiced_sec),
                )
                self._maybe_update_centroid_locked(spk, emb, dur_s, voiced_s, conf)
                self._record_voiced_sample_locked(emb, voiced_s, spk)
                self._maybe_recluster_history_locked()
                self._last_speaker = spk
            logger.info(
                "Diarized %d..%dms (%.2fs, voiced=%.2fs) -> %s (si=%s sc=%s conf=%.2f splits=1)",
                start_ms,
                end_ms,
                dur_s,
                voiced_s,
                spk,
                f"{si:.2f}" if si is not None else "-",
                f"{sc:.2f}" if sc is not None else "-",
                conf,
            )
            return [
                DiarizedSubSegment(
                    speaker=spk,
                    audio=arr,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=conf,
                    embedding=emb,
                )
            ]

        # Longer segment (>= 2.8s): batched sliding-window voice embeddings for handoff detection
        win_n = int(self.window_sec * SAMPLE_RATE)
        hop_n = int(self.hop_sec * SAMPLE_RATE)
        starts = list(range(0, max(1, len(arr) - win_n + 1), hop_n))
        if len(arr) - (starts[-1] + win_n) > int(0.45 * SAMPLE_RATE):
            starts.append(len(arr) - win_n)
        win_active_secs = [_active_speech_sec(arr[st : st + win_n]) for st in starts]
        if hasattr(self.embedder, "embed_sliding_windows"):
            win_embs = self.embedder.embed_sliding_windows(arr, starts, win_n)
        else:
            win_embs = self.embedder.embed_batch([arr[st : st + win_n] for st in starts])

        with self._lock:
            if not self._voiced_samples and self._centroids["interviewer"] is None and self._centroids["candidate"] is None:
                self._first_speaker = default_speaker
            win_speakers, win_confs = self._label_window_sequence_locked(
                win_embs,
                win_active_secs,
                default_speaker=default_speaker,
            )

            # Smooth isolated 1-window interior spikes to avoid false splits on laughs/filler
            smoothed = list(win_speakers)
            for k in range(1, len(smoothed) - 1):
                if smoothed[k] != smoothed[k - 1] and smoothed[k] != smoothed[k + 1]:
                    smoothed[k] = smoothed[k - 1]

            # Build contiguous speaker runs
            runs: list[tuple[SpeakerRole, int, int]] = []
            run_start = 0
            for k in range(1, len(smoothed)):
                if smoothed[k] != smoothed[run_start]:
                    runs.append((smoothed[run_start], run_start, k - 1))
                    run_start = k
            runs.append((smoothed[run_start], run_start, len(smoothed) - 1))

            # Single speaker throughout the whole VAD chunk
            if len(runs) == 1:
                spk = runs[0][0]
                voiced_mask = [i for i, a in enumerate(win_active_secs) if a >= self.min_window_voiced_sec]
                if voiced_mask:
                    mean_emb = _l2_normalize(win_embs[voiced_mask].mean(axis=0))
                    avg_conf = float(np.mean([win_confs[i] for i in voiced_mask]))
                else:
                    mean_emb = _l2_normalize(win_embs.mean(axis=0))
                    avg_conf = float(np.mean(win_confs))
                self._maybe_update_centroid_locked(spk, mean_emb, dur_s, voiced_s, avg_conf)
                self._record_voiced_sample_locked(mean_emb, voiced_s, spk)
                self._maybe_recluster_history_locked()
                self._last_speaker = spk
                ci, cc = self._centroids["interviewer"], self._centroids["candidate"]
                si = float(mean_emb @ ci) if ci is not None else None
                sc = float(mean_emb @ cc) if cc is not None else None
                logger.info(
                    "Diarized %d..%dms (%.2fs, voiced=%.2fs) -> %s (si=%s sc=%s conf=%.2f splits=1)",
                    start_ms,
                    end_ms,
                    dur_s,
                    voiced_s,
                    spk,
                    f"{si:.2f}" if si is not None else "-",
                    f"{sc:.2f}" if sc is not None else "-",
                    avg_conf,
                )
                return [
                    DiarizedSubSegment(
                        speaker=spk,
                        audio=arr,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        confidence=round(avg_conf, 3),
                        embedding=mean_emb,
                    )
                ]

            # Multi-speaker handoff detected: compute exact low-RMS energy-trough split boundaries
            split_samples: list[int] = [0]
            for run_idx in range(len(runs) - 1):
                _, _r_s, left_end_w = runs[run_idx]
                _, right_start_w, _r_e = runs[run_idx + 1]
                # Search between center of left_end_w and center of right_start_w
                search_lo = starts[left_end_w] + int(0.35 * win_n)
                search_hi = min(len(arr) - 1, starts[right_start_w] + int(0.65 * win_n))
                boundary = self._find_energy_trough(arr, search_lo, search_hi)
                if boundary - split_samples[-1] >= int(0.4 * SAMPLE_RATE):
                    split_samples.append(boundary)
            split_samples.append(len(arr))

            # If a boundary got dropped because it was too close to the edge, align runs accordingly
            out: list[DiarizedSubSegment] = []
            for idx in range(len(split_samples) - 1):
                s0, s1 = split_samples[idx], split_samples[idx + 1]
                sub_audio = arr[s0:s1]
                if len(sub_audio) < int(0.25 * SAMPLE_RATE):
                    continue
                run_spk, w_lo, w_hi = runs[min(idx, len(runs) - 1)]
                run_voiced = [
                    w_i
                    for w_i in range(w_lo, w_hi + 1)
                    if win_active_secs[w_i] >= self.min_window_voiced_sec
                ]
                emb_slice = win_embs[run_voiced] if run_voiced else win_embs[w_lo : w_hi + 1]
                sub_emb = _l2_normalize(emb_slice.mean(axis=0))
                sub_conf = float(np.mean(win_confs[w_lo : w_hi + 1]))
                sub_dur = len(sub_audio) / float(SAMPLE_RATE)
                sub_voiced = _active_speech_sec(sub_audio)
                self._maybe_update_centroid_locked(run_spk, sub_emb, sub_dur, sub_voiced, sub_conf)
                self._record_voiced_sample_locked(sub_emb, sub_voiced, run_spk)
                sub_start_ms = start_ms + int(round(s0 / SAMPLE_RATE * 1000.0))
                sub_end_ms = start_ms + int(round(s1 / SAMPLE_RATE * 1000.0))
                if idx == len(split_samples) - 2:
                    sub_end_ms = max(sub_start_ms, end_ms)
                out.append(
                    DiarizedSubSegment(
                        speaker=run_spk,
                        audio=sub_audio,
                        start_ms=sub_start_ms,
                        end_ms=sub_end_ms,
                        confidence=round(sub_conf, 3),
                        embedding=sub_emb,
                    )
                )
            self._maybe_recluster_history_locked()
            if out:
                self._last_speaker = out[-1].speaker
            logger.info(
                "Diarized %d..%dms (%.2fs, voiced=%.2fs) -> splits=%d (%s)",
                start_ms,
                end_ms,
                dur_s,
                voiced_s,
                len(out),
                ",".join(f"{s.speaker}({(s.end_ms - s.start_ms) / 1000.0:.1f}s)" for s in out),
            )
            return out

    def bind_utterance(self, utterance_id: str, sub: DiarizedSubSegment) -> None:
        if sub.embedding is None:
            return
        dur_s = max(0.1, (sub.end_ms - sub.start_ms) / 1000.0)
        with self._lock:
            self._utt_history[utterance_id] = (sub.embedding.copy(), dur_s, sub.speaker)
            if len(self._utt_history) > 250:
                oldest = next(iter(self._utt_history))
                self._utt_history.pop(oldest, None)

    def record_manual_correction(self, utterance_id: str, new_speaker: SpeakerRole) -> bool:
        """
        Learns from a 1-click UI speaker flip: relabels the utterance's stored voice vector
        and rebuilds centroids from recorded history so subsequent turns immediately align.
        """
        with self._lock:
            entry = self._utt_history.get(utterance_id)
            if entry is None:
                return False
            emb, dur_s, _old = entry
            self._utt_history[utterance_id] = (emb, dur_s, new_speaker)
            self._manual_override = True

            # Recompute speaker centroids weighted by duration from history
            for role in ("interviewer", "candidate"):
                items = [
                    (e, min(6.0, d))
                    for (e, d, spk) in self._utt_history.values()
                    if spk == role and d >= 0.4
                ]
                if items:
                    weighted = sum(e * w for e, w in items)
                    self._centroids[role] = _l2_normalize(weighted)
                    self._counts[role] = len(items)
                else:
                    self._centroids[role] = None
                    self._counts[role] = 0
            return True

    def _score_embedding_locked(
        self,
        emb: np.ndarray,
        *,
        default_speaker: SpeakerRole,
        allow_seed: bool,
    ) -> tuple[SpeakerRole, float, float | None, float | None]:
        ci = self._centroids["interviewer"]
        cc = self._centroids["candidate"]

        # Case 1: Neither speaker enrolled yet -> seed `default_speaker` only on cleanly voiced audio
        if ci is None and cc is None:
            if allow_seed:
                self._centroids[default_speaker] = emb.copy()
                self._counts[default_speaker] = 1
            return default_speaker, 1.0, None, None

        # Case 2: Only one speaker enrolled so far -> check if this is the second voice
        if ci is None or cc is None:
            known_role: SpeakerRole = "interviewer" if ci is not None else "candidate"
            unknown_role: SpeakerRole = "candidate" if known_role == "interviewer" else "interviewer"
            known_vec = ci if ci is not None else cc
            assert known_vec is not None
            sim = float(emb @ known_vec)
            si = sim if known_role == "interviewer" else None
            sc = sim if known_role == "candidate" else None
            if sim < self.new_speaker_sim_threshold:
                if allow_seed:
                    self._centroids[unknown_role] = emb.copy()
                    self._counts[unknown_role] = 1
                    return (
                        unknown_role,
                        round(min(1.0, self.new_speaker_sim_threshold - sim + 0.5), 3),
                        si,
                        sc,
                    )
                # Don't flip to unknown speaker on unvoiced/ultra-short noise when unseeded
                return self._last_speaker, 0.0, si, sc
            return known_role, round(max(0.0, sim), 3), si, sc

        # Case 3: Both speakers enrolled -> cosine nearest centroid
        si = float(emb @ ci)
        sc = float(emb @ cc)
        margin = abs(si - sc)
        if max(si, sc) < 0.25 or margin < 0.04:
            return self._last_speaker, round(margin, 3), si, sc
        chosen: SpeakerRole = "interviewer" if si >= sc else "candidate"
        return chosen, round(margin, 3), si, sc

    def _label_window_sequence_locked(
        self,
        win_embs: np.ndarray,
        win_active_secs: list[float],
        *,
        default_speaker: SpeakerRole,
    ) -> tuple[list[SpeakerRole], list[float]]:
        ci = self._centroids["interviewer"]
        cc = self._centroids["candidate"]
        voiced_idx = [i for i, a in enumerate(win_active_secs) if a >= self.min_window_voiced_sec]

        # Neither speaker enrolled yet: cluster ONLY cleanly voiced windows so a sparse
        # 1-word lead-in (e.g. "Let's" + silence) can NEVER anchor speaker 1.
        if ci is None and cc is None:
            other: SpeakerRole = "candidate" if default_speaker == "interviewer" else "interviewer"
            if not voiced_idx:
                return [default_speaker] * len(win_embs), [0.5] * len(win_embs)
            if len(voiced_idx) == 1:
                v0 = voiced_idx[0]
                if win_active_secs[v0] >= self.min_seed_voiced_sec:
                    self._centroids[default_speaker] = win_embs[v0].copy()
                    self._counts[default_speaker] = 1
                return [default_speaker] * len(win_embs), [1.0] * len(win_embs)

            v_embs = win_embs[voiced_idx]
            labels, c0, c1, c_sim = _farthest_pair_kmeans2(v_embs)
            c0_pos = [voiced_idx[k] for k, lbl in enumerate(labels) if lbl == 0]
            c1_pos = [voiced_idx[k] for k, lbl in enumerate(labels) if lbl == 1]

            # Order clusters chronologically so whichever voice spoke first gets `default_speaker`
            if c0_pos and c1_pos and min(c1_pos) < min(c0_pos):
                c0, c1 = c1, c0
                c0_pos, c1_pos = c1_pos, c0_pos

            # Require both clusters to have solid voiced evidence & clear acoustic separation
            c0_solid = len(c0_pos) >= 2 or (
                len(c0_pos) == 1 and win_active_secs[c0_pos[0]] >= 0.85
            )
            c1_solid = len(c1_pos) >= 2
            if c0_solid and c1_solid and c_sim < self.new_speaker_sim_threshold:
                self._centroids[default_speaker] = c0
                self._counts[default_speaker] = 1
                self._centroids[other] = c1
                self._counts[other] = 1
                return self._decode_hysteresis_locked(win_embs, win_active_secs, voiced_idx)

            # Single speaker (or sparse noise outlier) -> seed `default_speaker` from dominant voiced cluster
            dom_pos = c0_pos if len(c0_pos) >= len(c1_pos) else c1_pos
            self._centroids[default_speaker] = _l2_normalize(win_embs[dom_pos].mean(axis=0))
            self._counts[default_speaker] = 1
            return [default_speaker] * len(win_embs), [1.0] * len(win_embs)

        # Only one speaker enrolled so far
        if ci is None or cc is None:
            known_role: SpeakerRole = "interviewer" if ci is not None else "candidate"
            unknown_role: SpeakerRole = "candidate" if known_role == "interviewer" else "interviewer"
            known_vec = ci if ci is not None else cc
            assert known_vec is not None
            sims = win_embs @ known_vec

            # Only allow cleanly voiced windows to vote for seeding the 2nd speaker
            voiced_new_idx = [
                i for i in voiced_idx if float(sims[i]) < self.new_speaker_sim_threshold
            ]
            can_seed_second = len(voiced_new_idx) >= 2 or (
                len(voiced_new_idx) == 1
                and win_active_secs[voiced_new_idx[0]] >= 0.85
                and float(sims[voiced_new_idx[0]]) < self.new_speaker_sim_threshold - 0.05
            )
            if can_seed_second:
                self._centroids[unknown_role] = _l2_normalize(
                    win_embs[voiced_new_idx].mean(axis=0)
                )
                self._counts[unknown_role] = 1
                return self._decode_hysteresis_locked(win_embs, win_active_secs, voiced_idx)

            # Still one speaker
            confs = [max(0.0, float(s)) for s in sims]
            return [known_role] * len(win_embs), confs

        # Both speakers enrolled -> hysteresis decoding across sliding windows
        return self._decode_hysteresis_locked(win_embs, win_active_secs, voiced_idx)

    def _decode_hysteresis_locked(
        self,
        win_embs: np.ndarray,
        win_active_secs: list[float],
        voiced_idx: list[int],
    ) -> tuple[list[SpeakerRole], list[float]]:
        ci = self._centroids["interviewer"]
        cc = self._centroids["candidate"]
        assert ci is not None and cc is not None
        si = win_embs @ ci
        sc = win_embs @ cc
        init_i = voiced_idx[0] if voiced_idx else 0
        cur: SpeakerRole = "interviewer" if float(si[init_i]) >= float(sc[init_i]) else "candidate"
        spks: list[SpeakerRole] = []
        confs: list[float] = []
        for k, (a, b) in enumerate(zip(si, sc)):
            fa, fb = float(a), float(b)
            # Do not trigger a speaker switch on an unvoiced/sparse window
            if win_active_secs[k] >= self.min_window_voiced_sec * 0.8:
                if cur == "interviewer" and (fb - fa) > self.switch_margin:
                    cur = "candidate"
                elif cur == "candidate" and (fa - fb) > self.switch_margin:
                    cur = "interviewer"
            spks.append(cur)
            confs.append(abs(fa - fb))
        return spks, confs

    def _maybe_update_centroid_locked(
        self,
        speaker: SpeakerRole,
        emb: np.ndarray,
        dur_s: float,
        voiced_s: float,
        confidence: float,
    ) -> None:
        if dur_s < self.min_update_dur_sec or voiced_s < self.min_seed_voiced_sec:
            return
        existing = self._centroids[speaker]
        if existing is None:
            self._centroids[speaker] = emb.copy()
            self._counts[speaker] = 1
            return
        # Only EMA-update on confident single-speaker matches so mixed/noisy audio never drifts centroids
        if confidence < 0.08:
            return
        count = self._counts[speaker]
        alpha = 0.25 if count < 4 else 0.10
        self._centroids[speaker] = _l2_normalize((1.0 - alpha) * existing + alpha * emb)
        self._counts[speaker] = count + 1

    def _record_voiced_sample_locked(
        self,
        emb: np.ndarray,
        voiced_s: float,
        speaker: SpeakerRole,
    ) -> None:
        if voiced_s < self.min_seed_voiced_sec:
            return
        self._voiced_samples.append((emb.copy(), voiced_s, speaker))
        if len(self._voiced_samples) > 40:
            self._voiced_samples.pop(0)

    def _maybe_recluster_history_locked(self) -> None:
        """
        Cold-start self-healing: if an early noisy turn somehow left one speaker bucket with
        <= 1 sample while the other bucket absorbed >= 4 voiced turns containing two distinct
        voices (pairwise min cosine sim < threshold), cleanly 2-means recluster all voiced turns.
        Skipped after any manual UI correction so we never override user feedback.
        """
        if self._manual_override or not (4 <= len(self._voiced_samples) <= 24):
            return
        by_role: dict[SpeakerRole, list[np.ndarray]] = {"interviewer": [], "candidate": []}
        dur_by_role: dict[SpeakerRole, float] = {"interviewer": 0.0, "candidate": 0.0}
        for emb, v_s, spk in self._voiced_samples:
            by_role[spk].append(emb)
            dur_by_role[spk] += v_s

        lopsided = (
            (len(by_role["interviewer"]) <= 1 or dur_by_role["interviewer"] < 1.8)
            and len(by_role["candidate"]) >= 4
        ) or (
            (len(by_role["candidate"]) <= 1 or dur_by_role["candidate"] < 1.8)
            and len(by_role["interviewer"]) >= 4
        )
        if not lopsided:
            return

        crowded_role: SpeakerRole = (
            "candidate" if len(by_role["candidate"]) >= len(by_role["interviewer"]) else "interviewer"
        )
        crowded_mat = np.stack(by_role[crowded_role], axis=0)
        if float(np.min(crowded_mat @ crowded_mat.T)) >= self.new_speaker_sim_threshold:
            return

        all_mat = np.stack([e for e, _, _ in self._voiced_samples], axis=0)
        labels, c0, c1, c_sim = _farthest_pair_kmeans2(all_mat)
        n0 = int(np.sum(labels == 0))
        n1 = int(np.sum(labels == 1))
        if n0 < 2 or n1 < 2 or c_sim >= self.new_speaker_sim_threshold:
            return

        first_idx_0 = int(np.argmax(labels == 0))
        first_idx_1 = int(np.argmax(labels == 1))
        other_first: SpeakerRole = (
            "candidate" if self._first_speaker == "interviewer" else "interviewer"
        )
        if first_idx_0 <= first_idx_1:
            role_for_0, role_for_1 = self._first_speaker, other_first
            self._centroids[role_for_0] = c0
            self._centroids[role_for_1] = c1
            self._counts[role_for_0] = n0
            self._counts[role_for_1] = n1
        else:
            role_for_0, role_for_1 = other_first, self._first_speaker
            self._centroids[role_for_0] = c0
            self._centroids[role_for_1] = c1
            self._counts[role_for_0] = n0
            self._counts[role_for_1] = n1

        self._voiced_samples = [
            (e, v_s, role_for_0 if int(lbl) == 0 else role_for_1)
            for (e, v_s, _old), lbl in zip(self._voiced_samples, labels)
        ]
        logger.info(
            "Diarizer self-healed lopsided centroids from %d voiced samples (c_sim=%.3f, counts=%s)",
            len(self._voiced_samples),
            c_sim,
            dict(self._counts),
        )

    @staticmethod
    def _find_energy_trough(arr: np.ndarray, lo: int, hi: int) -> int:
        """Finds the quietest 40ms frame between [lo, hi] so PCM cuts fall between words."""
        lo = max(0, min(lo, len(arr) - 1))
        hi = max(lo + 1, min(hi, len(arr)))
        win = int(0.04 * SAMPLE_RATE)  # 40ms
        step = int(0.02 * SAMPLE_RATE)  # 20ms
        if hi - lo <= win:
            return (lo + hi) // 2
        best_pos = (lo + hi) // 2
        best_energy = float("inf")
        for pos in range(lo, hi - win + 1, step):
            seg = arr[pos : pos + win]
            energy = float(np.dot(seg, seg))
            if energy < best_energy:
                best_energy = energy
                best_pos = pos + win // 2
        return best_pos
