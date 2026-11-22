"""Add-On Packs, Pyodide Assets, Local TTS, and Local ASR HTTP router."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from ..state_machines import InvalidStateTransitionError

MAX_ASR_AUDIO_BYTES = 4 * 1024 * 1024


class AddonModulesUpdateRequest(BaseModel):
    modules: dict[str, bool]


class TTSSpeakRequest(BaseModel):
    text: str
    voice: str | None = None
    rate: float = 1.0
    engine: str | None = Field(default=None, max_length=24)


def build_addons_router(
    runtime: Any,
    *,
    decode_audio_fn: Callable[[bytes, str], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/addons")
    async def get_addons_status() -> dict[str, Any]:
        return runtime.addons.status_dict()

    @router.post("/api/addons/modules")
    async def update_addon_modules(body: AddonModulesUpdateRequest) -> dict[str, Any]:
        runtime.addons.update_modules(body.modules)
        return runtime.addons.status_dict()

    async def _run_pack_action(pack_id: str, event: str, action_verb: str, op_fn: Callable[[str], Any]) -> dict[str, Any]:
        pack = runtime.addons.get_pack_spec(pack_id)
        if pack is not None and event not in pack["allowed_events"]:
            raise HTTPException(
                status_code=409,
                detail=InvalidStateTransitionError(
                    machine="addon_pack",
                    current_state=pack["state"],
                    event=event,
                    allowed_events=pack["allowed_events"],
                    remediation=pack["blocked_reasons"].get(
                        event,
                        f"Cannot {action_verb} '{pack_id}' while in state '{pack['state']}'.",
                    ),
                ).to_dict(),
            )
        try:
            status = await op_fn(pack_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if event == "install":
                raise HTTPException(status_code=500, detail=f"Failed to install {pack_id}: {exc}") from exc
            raise
        if pack_id == "rampart_pii" and event == "install" and runtime.profiles.state.resume is not None:
            await asyncio.to_thread(
                runtime.profiles.update_redaction_prefs,
                runtime.profiles.state.resume.prefs,
            )
        return {
            "addons": status,
            "profile": runtime.profiles.public_dict(),
            "tts": runtime.tts.status_dict(),
        }

    @router.post("/api/addons/{pack_id}/install")
    async def install_addon_pack(pack_id: str) -> dict[str, Any]:
        return await _run_pack_action(pack_id, "install", "install", runtime.addons.install_pack)

    @router.delete("/api/addons/{pack_id}")
    async def uninstall_addon_pack(pack_id: str) -> dict[str, Any]:
        return await _run_pack_action(pack_id, "uninstall", "remove", runtime.addons.uninstall_pack)

    @router.get("/api/addons/pyodide/{asset_path:path}")
    async def serve_pyodide_asset(asset_path: str, request: Request) -> Response:
        try:
            opened = runtime.pyodide.open_asset(asset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if opened is None:
            raise HTTPException(
                status_code=404,
                detail="Offline Python runner (Pyodide) is not installed or asset not found.",
            )
        file_obj, size_bytes, media_type, etag = opened
        if_none_match = (request.headers.get("if-none-match") or "").strip()
        if if_none_match and if_none_match == etag:
            file_obj.close()
            return Response(
                status_code=304,
                headers={
                    "ETag": etag,
                    "Cache-Control": "no-cache",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        def _stream_and_close():
            try:
                while True:
                    chunk = file_obj.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                file_obj.close()

        return StreamingResponse(
            _stream_and_close(),
            media_type=media_type,
            headers={
                "Content-Length": str(size_bytes),
                "ETag": etag,
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.get("/api/tts/status")
    async def get_tts_status() -> dict[str, Any]:
        if runtime.tts._cached_macos_voices is None:
            await asyncio.to_thread(runtime.tts.list_macos_voices)
        return runtime.tts.status_dict()

    @router.post("/api/tts/speak")
    async def synthesize_tts(body: TTSSpeakRequest) -> Response:
        if not (body.text or "").strip():
            raise HTTPException(status_code=400, detail="Text to speak cannot be empty.")
        try:
            await asyncio.wait_for(runtime._tts_sem.acquire(), timeout=0.05)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429,
                detail="Local voice synthesis is busy — try again in a moment.",
            ) from exc
        try:
            try:
                wav_bytes, engine = await runtime.tts.synthesize_wav(
                    body.text,
                    voice=body.voice,
                    rate=body.rate,
                    engine=body.engine,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            runtime._tts_sem.release()
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-TTS-Engine": engine,
                "Cache-Control": "no-store",
            },
        )

    @router.post("/api/tts/stream")
    async def stream_tts(body: TTSSpeakRequest) -> StreamingResponse:
        if not (body.text or "").strip():
            raise HTTPException(status_code=400, detail="Text to speak cannot be empty.")
        try:
            await asyncio.wait_for(runtime._tts_sem.acquire(), timeout=0.05)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429,
                detail="Local voice synthesis is busy — try again in a moment.",
            ) from exc
        try:
            engine, sample_rate, chunk_iter = await runtime.tts.open_pcm_stream(
                body.text,
                voice=body.voice,
                rate=body.rate,
                engine=body.engine,
            )
        except ValueError as exc:
            runtime._tts_sem.release()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            runtime._tts_sem.release()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except BaseException:
            runtime._tts_sem.release()
            raise

        cleaned_up = False

        async def _cleanup_once() -> None:
            nonlocal cleaned_up
            if cleaned_up:
                return
            cleaned_up = True
            try:
                close_fn = getattr(chunk_iter, "close", None)
                if callable(close_fn):
                    close_fn()
                else:
                    aclose = getattr(chunk_iter, "aclose", None)
                    if callable(aclose):
                        await aclose()
            except BaseException:
                pass
            finally:
                runtime._tts_sem.release()

        async def _guarded_stream():
            try:
                async for chunk in chunk_iter:
                    if chunk:
                        yield chunk
            finally:
                await _cleanup_once()

        class _PermitGuardedStreamingResponse(StreamingResponse):
            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    await _cleanup_once()

        return _PermitGuardedStreamingResponse(
            _guarded_stream(),
            media_type="application/octet-stream",
            headers={
                "X-TTS-Engine": engine,
                "X-TTS-Sample-Rate": str(sample_rate),
                "X-TTS-Channels": "1",
                "X-TTS-Sample-Format": "pcm_s16le",
                "Cache-Control": "no-store",
            },
        )

    @router.post("/api/asr/transcribe")
    async def transcribe_local_audio(request: Request) -> dict[str, Any]:
        try:
            await asyncio.wait_for(runtime._asr_http_sem.acquire(), timeout=0.05)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429,
                detail="Local transcription is already running — wait for the previous clip to finish.",
            ) from exc
        try:
            chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_ASR_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="Audio upload exceeds 4 MB limit.")
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="Audio payload is empty.")
            content_type = request.headers.get("content-type", "")
            try:
                pcm = await asyncio.to_thread(decode_audio_fn, raw_bytes, content_type)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            try:
                res = await asyncio.to_thread(runtime.asr.transcribe_pcm, pcm)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Local speech transcription unavailable: {exc}") from exc
            return {
                "text": res.text,
                "inference_ms": round(res.inference_ms, 1),
                "audio_duration_ms": round(res.audio_duration_ms, 1),
            }
        finally:
            runtime._asr_http_sem.release()

    return router
