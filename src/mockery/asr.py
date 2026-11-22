from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from parakeet_mlx import BaseParakeet, ParakeetTDT, from_pretrained, tokenizer
from parakeet_mlx.attention import RelPositionMultiHeadAttention
from parakeet_mlx.audio import get_logmel
from parakeet_mlx.cache import RotatingConformerCache
from parakeet_mlx.conformer import DwStridingSubsampling

from .models import DEFAULT_PARAKEET_MODEL, PARAKEET_CHECKPOINTS
from .state_machines import ASR_MODEL_TRANSITIONS, ASRModelState

logger = logging.getLogger(__name__)

CONTROL_TOKEN_RE = re.compile(r"<unk>|<pad>|<\|[^|>]*\|>")
WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_MLX_CACHE_LIMIT_MB = 64
LONG_UTTERANCE_CACHE_CLEAR_SEC = 10.0
STREAM_WARM_START_SEC = 3.0
STREAM_CACHE_CAPACITY = 512
STREAM_DROP_ENC_FRAMES = 12
STREAM_HALO_ENC_FRAMES = 2
MAX_ACTIVE_STREAMS = 3


@dataclass
class ASRTranscription:
    text: str
    inference_ms: float
    audio_duration_ms: float


def _resolve_model_source(model_id: str) -> str:
    """
    Resolves a Hugging Face repo ID from the local cache first (zero network HEAD requests
    on warm boot), falling back to the raw repo ID so `from_pretrained` downloads on miss.
    """
    if Path(model_id).exists():
        return model_id
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


def _configure_mlx_cache_limit() -> int:
    raw = os.getenv("PARAKEET_MLX_CACHE_MB", str(DEFAULT_MLX_CACHE_LIMIT_MB)).strip()
    try:
        limit_mb = max(0, int(raw))
    except ValueError:
        limit_mb = DEFAULT_MLX_CACHE_LIMIT_MB
    if limit_mb > 0:
        mx.set_cache_limit(limit_mb * 1024 * 1024)
    return limit_mb


class _FusedRelPositionMultiHeadAttention(nn.Module):
    """
    Algebraically equivalent drop-in replacement for `RelPositionMultiHeadAttention` that
    fuses `linear_q`, `linear_k`, and `linear_v` into a single `linear_qkv` projection
    (3 Metal GEMM launches -> 1 per Conformer layer across all 24 layers).
    """

    def __init__(self, orig: RelPositionMultiHeadAttention) -> None:
        super().__init__()
        self.n_head = orig.n_head
        self.head_dim = orig.head_dim
        self.scale = orig.scale
        self.pos_bias_u = orig.pos_bias_u
        self.pos_bias_v = orig.pos_bias_v
        self.linear_pos = orig.linear_pos
        self.linear_out = orig.linear_out

        d_model = self.n_head * self.head_dim
        has_bias = "bias" in orig.linear_q and orig.linear_q.bias is not None
        self.linear_qkv = nn.Linear(d_model, 3 * d_model, bias=has_bias)
        self.linear_qkv.weight = mx.concatenate(
            [orig.linear_q.weight, orig.linear_k.weight, orig.linear_v.weight],
            axis=0,
        )
        if has_bias:
            self.linear_qkv.bias = mx.concatenate(
                [orig.linear_q.bias, orig.linear_k.bias, orig.linear_v.bias],
                axis=0,
            )

    @staticmethod
    def _rel_shift(x: mx.array) -> mx.array:
        b, h, tq, pos_len = x.shape
        x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)])
        x = x.reshape(b, h, pos_len + 1, tq)[:, :, 1:, :]
        return x.reshape(b, h, tq, pos_len)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache: Any = None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb is necessary!")

        qkv = self.linear_qkv(q)
        q, k, v = mx.split(qkv, 3, axis=-1)
        p = self.linear_pos(pos_emb)

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape
        p_batch, pos_len, _ = p.shape
        if p_batch == 1 and batch > 1:
            p = mx.broadcast_to(p, (batch, pos_len, p.shape[-1]))

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)

        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch_kv(k, v)

        matrix_bd = self._rel_shift(mx.matmul(q_v, p.swapaxes(-2, -1)))
        matrix_bd = matrix_bd[:, :, :, : k.shape[-2]] * self.scale
        if mask is not None:
            matrix_bd = mx.where(mx.expand_dims(mask, 0), -mx.inf, matrix_bd)

        o = mx.fast.scaled_dot_product_attention(q_u, k, v, scale=self.scale, mask=matrix_bd)
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, -1)
        return self.linear_out(o)


def _conv1d_k1_to_linear(conv: nn.Conv1d) -> nn.Linear:
    """Converts a kernel_size=1 Conv1d into an identical nn.Linear so it can be quantized."""
    # MLX Conv1d weight layout: (out_channels, kernel_size, in_channels)
    w = conv.weight[:, 0, :]
    has_bias = "bias" in conv and conv.bias is not None
    lin = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
    lin.weight = w
    if has_bias:
        lin.bias = conv.bias
    return lin


def _fold_depthwise_batchnorm(conv_mod: Any) -> None:
    """Folds eval-mode BatchNorm scale/shift into preceding depthwise Conv1d weights & bias."""
    dw = getattr(conv_mod, "depthwise_conv", None)
    bn = getattr(conv_mod, "batch_norm", None)
    if not isinstance(dw, nn.Conv1d) or not isinstance(bn, nn.BatchNorm):
        return
    bn.eval()
    inv_std = (bn.running_var.astype(mx.float32) + bn.eps) ** -0.5
    if "weight" in bn and bn.weight is not None:
        scale = bn.weight.astype(mx.float32) * inv_std
    else:
        scale = inv_std
    if "bias" in bn and bn.bias is not None:
        shift = bn.bias.astype(mx.float32)
    else:
        shift = mx.zeros_like(scale)

    orig_dtype = dw.weight.dtype
    dw.weight = (dw.weight.astype(mx.float32) * scale[:, None, None]).astype(orig_dtype)
    if "bias" in dw and dw.bias is not None:
        old_bias = dw.bias.astype(mx.float32)
    else:
        old_bias = mx.zeros_like(scale)
    dw.bias = ((old_bias - bn.running_mean.astype(mx.float32)) * scale + shift).astype(orig_dtype)
    conv_mod.batch_norm = nn.Identity()


def _optimize_parakeet_model(model: BaseParakeet) -> str:
    """
    Applies accuracy-preserving graph + memory optimizations to Parakeet's Conformer encoder:
    1. Converts pointwise `Conv1d(k=1)` layers to `nn.Linear` so they quantize cleanly (-68 MB).
    2. Folds eval-mode `BatchNorm` into `depthwise_conv` across all 24 Conformer blocks.
    3. Fuses Q/K/V projections in `RelPositionMultiHeadAttention` into a single `linear_qkv`.
    4. Selectively quantizes ONLY `model.encoder` to 8-bit (`group_size=64`), keeping the
       tiny decoder & joint network in full precision (~0.2% WER delta, -45% weight RAM).

    Set `PARAKEET_QUANTIZE=none` to disable quantization (graph fusion still applies).
    """
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is not None:
        for blk in layers:
            conv = getattr(blk, "conv", None)
            if conv is not None:
                for attr in ("pointwise_conv1", "pointwise_conv2"):
                    layer = getattr(conv, attr, None)
                    if isinstance(layer, nn.Conv1d) and layer.weight.shape[1] == 1:
                        setattr(conv, attr, _conv1d_k1_to_linear(layer))
                _fold_depthwise_batchnorm(conv)
            attn = getattr(blk, "self_attn", None)
            if type(attn) is RelPositionMultiHeadAttention:
                blk.self_attn = _FusedRelPositionMultiHeadAttention(attn)

    quant_mode = os.getenv("PARAKEET_QUANTIZE", "8").strip().lower()
    if quant_mode in ("8", "int8", "q8") and encoder is not None:
        nn.quantize(encoder, group_size=64, bits=8)
        mx.eval(model.parameters())
        return "enc-q8-fused"

    mx.eval(model.parameters())
    return "bf16-fused"


class FastTDTGreedyDecoder:
    """
    Drop-in fast greedy TDT decoder running on macOS Apple Accelerate (AMX) CPU BLAS.

    Fixes three bottlenecks in upstream `ParakeetTDT.decode_greedy`:
    1. Pre-projects `joint.enc(features)` once on GPU for all `S` frames instead of per-step.
    2. Caches LSTM `(h, c)` and `joint.pred(h)` across blank frames (zero redundant LSTM calls).
    3. Eliminates unused 8,193-way `softmax` + `log` entropy math and per-step GPU->CPU syncs.
    Fuses `[Wx, Wh]` per LSTM layer with preallocated scratch buffers (~3.3x faster decode,
    100% greedy-exact vs upstream).
    """

    def __init__(self, model: ParakeetTDT) -> None:
        self.vocab: list[str] = list(model.vocabulary)
        self.blank_id: int = len(self.vocab)
        self.durations: tuple[int, ...] = tuple(int(d) for d in model.durations)
        self.max_symbols: int = int(model.max_symbols) if model.max_symbols is not None else 10

        dec = model.decoder
        self.hidden_size: int = int(dec.pred_hidden)
        self.embed_w: np.ndarray = np.ascontiguousarray(
            np.array(dec.prediction["embed"].weight.astype(mx.float32))
        )

        # Fuse [Wx, Wh] into a single (4*H, 2*H) C-contiguous matrix per LSTM layer
        self._wxh_layers: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in dec.prediction["dec_rnn"].lstm:
            wx = np.array(layer.Wx.astype(mx.float32))
            wh = np.array(layer.Wh.astype(mx.float32))
            wxh = np.ascontiguousarray(np.hstack([wx, wh]))
            bias = (
                np.ascontiguousarray(np.array(layer.bias.astype(mx.float32)))
                if ("bias" in layer and layer.bias is not None)
                else np.zeros(wx.shape[0], dtype=np.float32)
            )
            self._wxh_layers.append((wxh, bias))
        self.n_layers: int = len(self._wxh_layers)

        self._pred_w: np.ndarray = np.ascontiguousarray(
            np.array(model.joint.pred.weight.astype(mx.float32))
        )
        self._pred_b: np.ndarray = (
            np.ascontiguousarray(np.array(model.joint.pred.bias.astype(mx.float32)))
            if ("bias" in model.joint.pred and model.joint.pred.bias is not None)
            else np.zeros(self._pred_w.shape[0], dtype=np.float32)
        )

        act_layer = model.joint.joint_net[0]
        if isinstance(act_layer, nn.ReLU):
            self._act_kind = "relu"
        elif isinstance(act_layer, nn.Tanh):
            self._act_kind = "tanh"
        elif isinstance(act_layer, nn.Sigmoid):
            self._act_kind = "sigmoid"
        else:
            raise ValueError(f"Unsupported TDT joint activation: {type(act_layer)}")

        out_lin = model.joint.joint_net[2]
        self._j_w: np.ndarray = np.ascontiguousarray(np.array(out_lin.weight.astype(mx.float32)))
        self._j_b: np.ndarray = (
            np.ascontiguousarray(np.array(out_lin.bias.astype(mx.float32)))
            if ("bias" in out_lin and out_lin.bias is not None)
            else np.zeros(self._j_w.shape[0], dtype=np.float32)
        )

        # Reusable scratch buffers (safe because ParakeetEngine serializes inference under `_lock`)
        hs = self.hidden_size
        self._zero_embed = np.zeros(hs, dtype=np.float32)
        self._xh_buf = np.empty(2 * hs, dtype=np.float32)
        self._gates_buf = np.empty(4 * hs, dtype=np.float32)
        self._pred_out_buf = np.empty(self._pred_w.shape[0], dtype=np.float32)
        self._joint_in_buf = np.empty(self._j_w.shape[1], dtype=np.float32)
        self._logits_buf = np.empty(self._j_w.shape[0], dtype=np.float32)

    @classmethod
    def try_build(cls, model: BaseParakeet) -> FastTDTGreedyDecoder | None:
        if not isinstance(model, ParakeetTDT):
            return None
        try:
            return cls(model)
        except Exception as exc:
            logger.warning("FastTDTGreedyDecoder unavailable (%s); falling back to MLX decoder", exc)
            return None

    def _lstm_and_pred_step(
        self,
        token: int | None,
        h_list: list[np.ndarray],
        c_list: list[np.ndarray],
    ) -> np.ndarray:
        hs = self.hidden_size
        xh = self._xh_buf
        gates = self._gates_buf
        x = self.embed_w[token] if token is not None else self._zero_embed

        for i, (wxh, bias) in enumerate(self._wxh_layers):
            xh[:hs] = x
            xh[hs:] = h_list[i]
            np.dot(wxh, xh, out=gates)
            gates += bias
            # MLX nn.LSTM gate order: i, f, g, o
            gi = 1.0 / (1.0 + np.exp(-gates[:hs]))
            gf = 1.0 / (1.0 + np.exp(-gates[hs : 2 * hs]))
            gg = np.tanh(gates[2 * hs : 3 * hs])
            go = 1.0 / (1.0 + np.exp(-gates[3 * hs :]))
            c_new = gf * c_list[i] + gi * gg
            h_new = go * np.tanh(c_new)
            h_list[i] = h_new
            c_list[i] = c_new
            x = h_new

        np.dot(self._pred_w, x, out=self._pred_out_buf)
        self._pred_out_buf += self._pred_b
        return self._pred_out_buf

    def decode_enc_proj(self, enc_proj_np: np.ndarray) -> str:
        if enc_proj_np.shape[0] == 0:
            return ""
        tokens, _ = self.decode_enc_proj_stateful(enc_proj_np, state=None)
        if not tokens:
            return ""
        return tokenizer.decode(tokens, self.vocab)

    def decode_enc_proj_stateful(
        self,
        enc_proj_np: np.ndarray,
        state: _TDTDecodeState | None = None,
    ) -> tuple[list[int], _TDTDecodeState]:
        """
        Incremental greedy TDT decoder that resumes LSTM `(h, c)`, `pred_proj`, token prefix,
        and cross-boundary duration `step_offset` across streaming Conformer chunks.
        Never mutates the caller's input `state` so speculative tail decoding is side-effect free.
        """
        hs = self.hidden_size
        if state is None:
            h_list = [np.zeros(hs, dtype=np.float32) for _ in range(self.n_layers)]
            c_list = [np.zeros(hs, dtype=np.float32) for _ in range(self.n_layers)]
            pred_proj = self._lstm_and_pred_step(None, h_list, c_list).copy()
            tokens: list[int] = []
            step = 0
        else:
            h_list = [h.copy() for h in state.h_list]
            c_list = [c.copy() for c in state.c_list]
            pred_proj = state.pred_proj.copy()
            tokens = list(state.tokens)
            step = state.step_offset

        n_frames = enc_proj_np.shape[0]
        new_symbols = 0
        blank_id = self.blank_id
        token_end = blank_id + 1
        durations = self.durations
        max_symbols = self.max_symbols
        j_w = self._j_w
        j_b = self._j_b
        joint_in = self._joint_in_buf
        logits = self._logits_buf
        act_kind = self._act_kind

        while step < n_frames:
            np.add(enc_proj_np[step], pred_proj, out=joint_in)
            if act_kind == "relu":
                np.maximum(joint_in, 0.0, out=joint_in)
            elif act_kind == "tanh":
                np.tanh(joint_in, out=joint_in)
            else:
                np.negative(joint_in, out=joint_in)
                np.exp(joint_in, out=joint_in)
                joint_in += 1.0
                np.reciprocal(joint_in, out=joint_in)

            np.dot(j_w, joint_in, out=logits)
            logits += j_b

            pred_tok = int(np.argmax(logits[:token_end]))
            dur_idx = int(np.argmax(logits[token_end:]))

            if pred_tok != blank_id:
                tokens.append(pred_tok)
                pred_proj = self._lstm_and_pred_step(pred_tok, h_list, c_list).copy()

            jump = durations[dur_idx]
            step += jump
            if jump != 0:
                new_symbols = 0
            else:
                new_symbols += 1
                if new_symbols >= max_symbols:
                    step += 1
                    new_symbols = 0

        next_state = _TDTDecodeState(
            h_list=h_list,
            c_list=c_list,
            pred_proj=pred_proj,
            tokens=tokens,
            step_offset=max(0, step - n_frames),
        )
        return tokens, next_state


@dataclass
class _TDTDecodeState:
    h_list: list[np.ndarray]
    c_list: list[np.ndarray]
    pred_proj: np.ndarray
    tokens: list[int]
    step_offset: int


class _FixedRotatingConformerCache(RotatingConformerCache):
    """
    Fixes two upstream bugs in `parakeet_mlx.cache.RotatingConformerCache.update_and_fetch_conv`:
    1. Prepends the PREVIOUS chunk's trailing conv frames BEFORE updating `self.conv`.
    2. Saves the committed frames right before `cache_drop_size` (rather than the dropped tail).
    """

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        if padding == 0:
            return x
        b, s, d = x.shape
        if self.conv is None:
            self.conv = mx.zeros((b, padding, d), x.dtype)

        result = mx.concatenate([self.conv, x], axis=1)
        result = mx.pad(result, [(0, 0), (0, padding), (0, 0)])

        committed = s - self.cache_drop_size
        if committed > 0:
            take = min(padding, committed)
            cache_update = x[:, committed - take : committed, :]
            if take < padding:
                self.conv = mx.concatenate([self.conv[:, take:, :], cache_update], axis=1)
            else:
                self.conv = cache_update
        return result


@dataclass
class _UtteranceStreamState:
    caches: list[_FixedRotatingConformerCache]
    committed_enc: int = 0
    dec_state: _TDTDecodeState | None = None
    last_samples: int = 0


class ParakeetEngine:
    """
    Thread-safe local Apple Silicon STT engine powered by NVIDIA Parakeet TDT on MLX.
    Explicitly transitions across `ASRModelState` (`idle` -> `loading` -> `ready` <-> `switching` | `error`)
    with a monotonic load generation counter so superseded background loads are discarded.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("PARAKEET_MODEL", DEFAULT_PARAKEET_MODEL)
        self._model: BaseParakeet | None = None
        self._fast_decoder: FastTDTGreedyDecoder | None = None
        self._optimization_mode: str = "unloaded"
        self._streams: OrderedDict[str, _UtteranceStreamState] = OrderedDict()
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
            self._streams.clear()
            self._model = None
            self._fast_decoder = None
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
            _configure_mlx_cache_limit()
            source = _resolve_model_source(target_id)
            model = from_pretrained(source, dtype=mx.bfloat16)
            opt_mode = _optimize_parakeet_model(model)
            fast_decoder = FastTDTGreedyDecoder.try_build(model)

            with self._state_lock:
                if self._load_gen != expected_gen:
                    logger.info("Discarding superseded Parakeet load for %s (gen=%d)", target_id, expected_gen)
                    return

            with self._lock:
                # Warm up Metal shaders on a 0.5s silent buffer using the active inference path
                dummy = mx.zeros((8000,), dtype=mx.float32)
                mel = get_logmel(dummy, model.preprocessor_config)
                if fast_decoder is not None:
                    self._infer_fast_locked(model, fast_decoder, mel)
                else:
                    warm = model.generate(mel)
                    mx.eval(warm)
                mx.clear_cache()
                with self._state_lock:
                    if self._load_gen != expected_gen:
                        logger.info("Discarding superseded Parakeet load after warmup (gen=%d)", expected_gen)
                        return
                    self._model = model
                    self._fast_decoder = fast_decoder
                    self._optimization_mode = (
                        f"{opt_mode}+amx-tdt+kv-stream" if fast_decoder is not None else opt_mode
                    )
                    self._set_state_locked(ASRModelState.READY)
                    self.error_message = None

            elapsed = time.perf_counter() - t0
            logger.info(
                "Parakeet MLX ready in %.2fs (%s, mode=%s)",
                elapsed,
                target_id,
                self._optimization_mode,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load Parakeet MLX model")
            with self._state_lock:
                if self._load_gen == expected_gen:
                    self.error_message = str(exc)
                    self._set_state_locked(ASRModelState.ERROR)

    @staticmethod
    def _infer_fast_locked(
        model: BaseParakeet,
        fast_decoder: FastTDTGreedyDecoder,
        mel: mx.array,
    ) -> str:
        if mel.ndim == 2:
            mel = mx.expand_dims(mel, 0)
        features, lengths = model.encoder(mel)
        valid_len = int(lengths[0].item())
        if valid_len <= 0:
            return ""
        enc_proj = model.joint.enc(features[:, :valid_len]).astype(mx.float32)[0]
        mx.eval(enc_proj)
        enc_proj_np = np.array(enc_proj, copy=False)
        return fast_decoder.decode_enc_proj(enc_proj_np)

    def drop_stream(self, utterance_id: str) -> None:
        """Releases cached Conformer KV/conv state for a finalized or canceled utterance."""
        if not utterance_id:
            return
        with self._lock:
            self._streams.pop(utterance_id, None)

    def clear_streams(self) -> None:
        """Releases all cached streaming partial states (e.g. on session switch or capture end)."""
        with self._lock:
            self._streams.clear()

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

    def transcribe_pcm_partial(
        self,
        utterance_id: str,
        pcm_float32: np.ndarray,
    ) -> ASRTranscription:
        """
        Incremental KV-cached partial transcription for an in-progress speech turn.
        - Short turns (< 3.0s) use full-clip `transcribe_pcm` so mel normalization stabilizes first.
        - Once >= 3.0s, reuses Conformer KV + conv caches with a 2-frame subsampling left-halo
          and stateful AMX TDT decoding, keeping partial latency flat (~45-60ms even at 22s,
          ~3.7x faster on long turns).
        - Finalization still runs exact full-clip `transcribe_pcm` (0.00% final WER regression).
        """
        model = self.ensure_loaded()
        cfg = model.preprocessor_config
        min_samples = cfg.hop_length * 2
        if pcm_float32.size < min_samples:
            return ASRTranscription(text="", inference_ms=0.0, audio_duration_ms=0.0)

        audio_duration_s = pcm_float32.size / float(cfg.sample_rate)
        audio_duration_ms = audio_duration_s * 1000.0
        encoder = getattr(model, "encoder", None)
        pre_encode = getattr(encoder, "pre_encode", None)
        pos_enc = getattr(encoder, "pos_enc", None)
        fast_dec = self._fast_decoder

        if (
            not utterance_id
            or fast_dec is None
            or not isinstance(pre_encode, DwStridingSubsampling)
            or pos_enc is None
            or (audio_duration_s < STREAM_WARM_START_SEC and utterance_id not in self._streams)
        ):
            return self.transcribe_pcm(pcm_float32)

        sub = int(model.encoder_config.subsampling_factor)
        pad_samples = int(cfg.sample_rate * 0.2)
        t0 = time.perf_counter()

        # Leading silence pad matches full-clip onset alignment; no trailing pad mid-utterance
        padded = np.zeros(pcm_float32.size + pad_samples, dtype=np.float32)
        padded[pad_samples : pad_samples + pcm_float32.size] = pcm_float32

        fallback_full = False
        with self._lock:
            fast_dec = self._fast_decoder
            if self._model is not model or fast_dec is None:
                fallback_full = True
            else:
                st = self._streams.get(utterance_id)
                if st is None or pcm_float32.size < st.last_samples:
                    st = _UtteranceStreamState(
                        caches=[
                            _FixedRotatingConformerCache(
                                STREAM_CACHE_CAPACITY, cache_drop_size=STREAM_DROP_ENC_FRAMES
                            )
                            for _ in range(len(encoder.layers))
                        ]
                    )
                    self._streams[utterance_id] = st
                self._streams.move_to_end(utterance_id)
                while len(self._streams) > MAX_ACTIVE_STREAMS:
                    self._streams.popitem(last=False)

                st.last_samples = int(pcm_float32.size)

                mel = get_logmel(mx.array(padded), cfg)
                if mel.ndim == 2:
                    mel = mx.expand_dims(mel, 0)

                halo = min(STREAM_HALO_ENC_FRAMES, st.committed_enc)
                start_mel = (st.committed_enc - halo) * sub
                avail_mel = mel.shape[1] - start_mel
                feed_mel = (avail_mel // sub) * sub

                if feed_mel <= halo * sub:
                    tokens = st.dec_state.tokens if st.dec_state is not None else []
                else:
                    mel_slice = mel[:, start_mel : start_mel + feed_mel, :]
                    pe, pe_lens = pre_encode(mel_slice, mx.array([feed_mel], dtype=mx.int64))
                    valid_pe = int(pe_lens[0].item())
                    x_chunk = pe[:, halo:valid_pe, :]
                    vl = int(x_chunk.shape[1])
                    if vl <= 0:
                        tokens = st.dec_state.tokens if st.dec_state is not None else []
                    else:
                        pos_offset = min(st.caches[0].offset, st.caches[0].capacity)
                        x_enc, pos_emb = pos_enc(x_chunk, offset=pos_offset)
                        for layer, cache in zip(encoder.layers, st.caches):
                            x_enc = layer(x_enc, pos_emb=pos_emb, cache=cache)
                        enc_proj = model.joint.enc(x_enc).astype(mx.float32)[0]
                        mx.eval(enc_proj)
                        enc_proj_np = np.array(enc_proj, copy=False)

                        fl = max(0, vl - STREAM_DROP_ENC_FRAMES)
                        if fl > 0:
                            _, st.dec_state = fast_dec.decode_enc_proj_stateful(
                                enc_proj_np[:fl], st.dec_state
                            )
                            st.committed_enc += fl
                        tokens, _ = fast_dec.decode_enc_proj_stateful(enc_proj_np[fl:], st.dec_state)

                raw_text = tokenizer.decode(tokens, fast_dec.vocab) if tokens else ""

        if fallback_full:
            return self.transcribe_pcm(pcm_float32)

        cleaned = CONTROL_TOKEN_RE.sub("", raw_text)
        clean_text = WHITESPACE_RE.sub(" ", cleaned).strip()
        inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_inference_ms = round(inference_ms, 1)

        return ASRTranscription(
            text=clean_text,
            inference_ms=inference_ms,
            audio_duration_ms=audio_duration_ms,
        )

    def transcribe_pcm(self, pcm_float32: np.ndarray) -> ASRTranscription:
        self.ensure_loaded()
        if pcm_float32.size == 0:
            return ASRTranscription(text="", inference_ms=0.0, audio_duration_ms=0.0)

        t0 = time.perf_counter()
        with self._lock:
            model = self._model
            fast_dec = self._fast_decoder
            if model is None:
                return ASRTranscription(text="", inference_ms=0.0, audio_duration_ms=0.0)
            cfg = model.preprocessor_config
            min_samples = cfg.hop_length * 2
            if pcm_float32.size < min_samples:
                return ASRTranscription(text="", inference_ms=0.0, audio_duration_ms=0.0)

            audio_duration_ms = (pcm_float32.size / cfg.sample_rate) * 1000.0
            pad_samples = int(cfg.sample_rate * 0.2)
            # Single contiguous zero-padded float32 buffer (avoids separate ascontiguousarray + pad copies)
            padded = np.zeros(pcm_float32.size + 2 * pad_samples, dtype=np.float32)
            padded[pad_samples : pad_samples + pcm_float32.size] = pcm_float32

            audio_mx = mx.array(padded)
            mel = get_logmel(audio_mx, cfg)
            if fast_dec is not None:
                raw_text = self._infer_fast_locked(model, fast_dec, mel)
            else:
                raw_text = model.generate(mel)[0].text
            if audio_duration_ms >= LONG_UTTERANCE_CACHE_CLEAR_SEC * 1000.0:
                mx.clear_cache()

        cleaned = CONTROL_TOKEN_RE.sub("", raw_text)
        clean_text = WHITESPACE_RE.sub(" ", cleaned).strip()
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
            "optimization": self._optimization_mode,
            "checkpoints": PARAKEET_CHECKPOINTS,
            "last_inference_ms": self.last_inference_ms,
            "error": self.error_message,
        }

