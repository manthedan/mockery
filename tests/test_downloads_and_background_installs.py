from __future__ import annotations

import asyncio
import email.utils
import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mockery.downloads import (
    DownloadCancelledError,
    _hf_auth_headers_for_url,
    _parse_retry_after,
    download_verified_file,
    format_download_error,
    make_hf_progress_tqdm_class,
)
from mockery.pyodide_pack import PyodidePackManager
from mockery.server import create_app


def test_download_retries_on_502_and_reports_progress(tmp_path: Path) -> None:
    payload = b"phonon-model-bytes-" * 1024
    expected_sha = hashlib.sha256(payload).hexdigest()
    attempts = {"count": 0}
    retries: list[tuple[int, int, str]] = []
    progress_events: list[tuple[int, int | None]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers.get("accept-encoding") == "identity"
        attempts["count"] += 1
        if attempts["count"] <= 2:
            return httpx.Response(502, text="Bad Gateway", request=req)
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Length": str(len(payload))},
            request=req,
        )

    dest = tmp_path / "default-voice.safetensors"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path, size = download_verified_file(
            "https://example.test/default-voice.safetensors",
            dest,
            expected_sha,
            client=client,
            max_attempts=4,
            backoff_base_s=0.0,
            progress_cb=lambda done, total: progress_events.append((done, total)),
            retry_cb=lambda att, mx, note: retries.append((att, mx, note)),
            sleep_fn=lambda _s: None,
        )

    assert path == dest
    assert size == len(payload)
    assert dest.read_bytes() == payload
    assert attempts["count"] == 3
    assert len(retries) == 2
    assert "502" in retries[0][2]
    assert progress_events[-1] == (len(payload), len(payload))
    assert list(tmp_path.glob("*.part")) == []


def test_download_resumes_partial_file_with_http_range(tmp_path: Path) -> None:
    payload = b"0123456789abcdef" * 4096
    split_at = len(payload) // 2
    expected_sha = hashlib.sha256(payload).hexdigest()
    seen_ranges: list[str | None] = []

    class _DroppingStream(httpx.SyncByteStream):
        def __init__(self, first_half: bytes) -> None:
            self._first = first_half

        def __iter__(self):
            yield self._first
            raise httpx.ReadError("simulated mid-stream disconnect")

    def handler(req: httpx.Request) -> httpx.Response:
        rng = req.headers.get("Range")
        seen_ranges.append(rng)
        if rng is None:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(payload))},
                stream=_DroppingStream(payload[:split_at]),
                request=req,
            )
        assert rng == f"bytes={split_at}-"
        return httpx.Response(
            206,
            content=payload[split_at:],
            headers={
                "Content-Range": f"bytes {split_at}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload) - split_at),
            },
            request=req,
        )

    dest = tmp_path / "model.q8.gguf"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path, size = download_verified_file(
            "https://example.test/model.q8.gguf",
            dest,
            expected_sha,
            client=client,
            max_attempts=3,
            backoff_base_s=0.0,
            sleep_fn=lambda _s: None,
        )

    assert path == dest
    assert size == len(payload)
    assert dest.read_bytes() == payload
    assert seen_ranges == [None, f"bytes={split_at}-"]


def test_download_rejects_malformed_206_range_and_restarts_cleanly(tmp_path: Path) -> None:
    payload = b"abcdefgh" * 2048
    split_at = len(payload) // 2
    expected_sha = hashlib.sha256(payload).hexdigest()
    call_idx = {"n": 0}

    class _DropHalf(httpx.SyncByteStream):
        def __iter__(self):
            yield payload[:split_at]
            raise httpx.ReadError("drop")

    def handler(req: httpx.Request) -> httpx.Response:
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(payload))},
                stream=_DropHalf(),
                request=req,
            )
        if call_idx["n"] == 2:
            # Server returns wrong start offset in Content-Range
            return httpx.Response(
                206,
                content=payload[10:],
                headers={
                    "Content-Range": f"bytes 10-{len(payload) - 1}/{len(payload)}",
                    "Content-Length": str(len(payload) - 10),
                },
                request=req,
            )
        # Attempt 3 restarts from 0 with clean 200
        assert req.headers.get("Range") is None
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Length": str(len(payload))},
            request=req,
        )

    dest = tmp_path / "weights.bin"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path, size = download_verified_file(
            "https://example.test/weights.bin",
            dest,
            expected_sha,
            client=client,
            max_attempts=4,
            backoff_base_s=0.0,
            sleep_fn=lambda _s: None,
        )

    assert path == dest
    assert size == len(payload)
    assert dest.read_bytes() == payload
    assert call_idx["n"] == 3


def test_download_416_commits_complete_verified_partial(tmp_path: Path) -> None:
    payload = b"complete-partial-payload" * 512
    expected_sha = hashlib.sha256(payload).hexdigest()
    calls = {"n": 0}

    class _DropAfterFull(httpx.SyncByteStream):
        def __iter__(self):
            yield payload
            raise httpx.ReadError("eof drop after all bytes written")

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(payload))},
                stream=_DropAfterFull(),
                request=req,
            )
        assert req.headers.get("Range") == f"bytes={len(payload)}-"
        return httpx.Response(
            416,
            headers={"Content-Range": f"bytes */{len(payload)}"},
            request=req,
        )

    dest = tmp_path / "full.bin"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path, size = download_verified_file(
            "https://example.test/full.bin",
            dest,
            expected_sha,
            client=client,
            max_attempts=3,
            backoff_base_s=0.0,
            sleep_fn=lambda _s: None,
        )

    assert path == dest
    assert size == len(payload)
    assert dest.read_bytes() == payload
    assert calls["n"] == 2


def test_download_enforces_max_bytes_on_cache_hit_and_stream(tmp_path: Path) -> None:
    payload = b"z" * 4096
    sha = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "cached.bin"
    dest.write_bytes(payload)

    with pytest.raises(RuntimeError, match="safety size limit"):
        download_verified_file(
            "https://example.test/cached.bin",
            dest,
            sha,
            max_bytes=1024,
        )


def test_format_download_error_strips_signed_cdn_query_and_mdn_link() -> None:
    signed_url = (
        "https://us.aws.cdn.hf.co/xet-bridge-us/6ad606ed/d4c21581"
        "?xip=mdpI28aP9io&Expires=1795414522&Policy=eyJTdGF0ZW1lbnQi&Signature=MEQCIG757"
    )
    req = httpx.Request("GET", signed_url)
    resp = httpx.Response(502, request=req)
    exc = httpx.HTTPStatusError(
        f"Server error '502 Bad Gateway' for url '{signed_url}' "
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/502",
        request=req,
        response=resp,
    )
    cleaned = format_download_error(exc, label="default-voice.safetensors")
    assert "502" in cleaned
    assert "us.aws.cdn.hf.co" in cleaned
    assert "default-voice.safetensors" in cleaned
    assert "Signature=" not in cleaned
    assert "Policy=" not in cleaned
    assert "developer.mozilla.org" not in cleaned


def test_hf_auth_headers_strict_https_origin_and_implicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token")
    assert _hf_auth_headers_for_url("https://huggingface.co/gradium/phonon-31m-en/resolve/main/config.json") == {
        "Authorization": "Bearer hf_test_secret_token"
    }
    # Reject HTTP downgrade, non-default ports, and third-party hosts
    assert _hf_auth_headers_for_url("http://huggingface.co/gradium/phonon-31m-en/resolve/main/config.json") == {}
    assert _hf_auth_headers_for_url("https://huggingface.co:8443/gradium/phonon-31m-en/resolve/main/config.json") == {}
    assert _hf_auth_headers_for_url("https://github.com/thewh1teagle/kokoro-onnx/releases/download/v1/a.bin") == {}
    assert _hf_auth_headers_for_url("https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js") == {}

    # Respect HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    assert _hf_auth_headers_for_url("https://huggingface.co/gradium/phonon-31m-en/resolve/main/config.json") == {}


def test_retry_after_http_date_and_over_budget_fail_fast(tmp_path: Path) -> None:
    now = 1_700_000_000.0
    http_date = email.utils.formatdate(now + 7, usegmt=True)
    assert _parse_retry_after("12", now_epoch=now) == pytest.approx(12.0, abs=0.1)
    assert _parse_retry_after(http_date, now_epoch=now) == pytest.approx(7.0, abs=1.0)
    assert _parse_retry_after("Infinity", now_epoch=now) is None
    assert _parse_retry_after("NaN", now_epoch=now) is None

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "120"}, request=req)

    dest = tmp_path / "rate_limited.bin"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="wait 120s"):
            download_verified_file(
                "https://example.test/rate_limited.bin",
                dest,
                "0" * 64,
                client=client,
                max_attempts=4,
                backoff_base_s=0.0,
            )
    assert list(tmp_path.glob("*.part")) == []


def test_download_cancellation_during_stream_and_backoff_cleans_up_staging(tmp_path: Path) -> None:
    # 1. Cancel during streaming
    cancel_ev = threading.Event()
    chunks_sent = {"n": 0}

    class _SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):
                chunks_sent["n"] += 1
                if chunks_sent["n"] >= 2:
                    cancel_ev.set()
                yield b"x" * 65536

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Length": str(65536 * 10)}, stream=_SlowStream(), request=req)

    dest = tmp_path / "big.bin"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadCancelledError):
            download_verified_file(
                "https://example.test/big.bin",
                dest,
                "0" * 64,
                client=client,
                cancel_event=cancel_ev,
            )

    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []

    # 2. Cancel during backoff wait after a mid-stream disconnect still cleans up staging file
    cancel_backoff = threading.Event()

    class _DropOnce(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial-data" * 100
            raise httpx.ReadError("drop")

    def on_retry(_att: int, _mx: int, _note: str) -> None:
        cancel_backoff.set()

    dest2 = tmp_path / "backoff.bin"
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=_DropOnce(), request=req))
    ) as client:
        with pytest.raises(DownloadCancelledError):
            download_verified_file(
                "https://example.test/backoff.bin",
                dest2,
                "0" * 64,
                client=client,
                max_attempts=3,
                backoff_base_s=5.0,
                retry_cb=on_retry,
                cancel_event=cancel_backoff,
            )

    assert not dest2.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_pyodide_install_retries_transient_502(tmp_path: Path) -> None:
    fake_assets = {
        "pyodide.js": b"console.log('pyodide');",
        "pyodide.asm.js": b"var asm = 1;",
    }
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in fake_assets.items()}
    calls = {"pyodide.js": 0, "pyodide.asm.js": 0}
    updates: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = req.url.path.rsplit("/", 1)[-1]
        calls[name] = calls.get(name, 0) + 1
        if name == "pyodide.js" and calls[name] == 1:
            return httpx.Response(502, request=req)
        return httpx.Response(
            200,
            content=fake_assets[name],
            headers={"Content-Length": str(len(fake_assets[name]))},
            request=req,
        )

    mgr = PyodidePackManager(
        tmp_path / "pyodide",
        base_url="https://example.test/pyodide",
        expected_hashes=hashes,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        mgr.install(
            client=client,
            on_progress=updates.append,
            max_attempts=3,
            backoff_base_s=0.0,
        )

    assert mgr.is_installed() is True
    assert calls["pyodide.js"] == 2
    assert any(u.get("phase") == "retrying" for u in updates)


def test_hf_progress_tqdm_supports_full_hf_api_avoids_double_count_and_cancels() -> None:
    events: list[dict[str, Any]] = []
    cancel_ev = threading.Event()
    Tqdm = make_hf_progress_tqdm_class(
        events.append,
        approx_total_bytes=100_000,
        cancel_event=cancel_ev,
        throttle_s=0.0,
    )
    with (
        Tqdm(
            total=100_000,
            unit="B",
            desc="Downloading bytes",
            name="huggingface_hub.snapshot_download.transfer",
        ) as transfer_bar,
        Tqdm(
            total=100_000,
            unit="B",
            desc="Reconstructing (incomplete total...)",
            name="huggingface_hub.snapshot_download",
        ) as reconstruct_bar,
    ):
        # Exercise methods used by huggingface_hub 2.4.0 snapshot_download and Xet paths
        transfer_bar.set_description_str("Fetching 2 files: 100%")
        transfer_bar.set_postfix_str("xet")
        _ = transfer_bar.format_dict
        transfer_bar.update(40_000)
        reconstruct_bar.update(40_000)
        # Reconstruction bar must NOT double-count transfer bytes
        assert events[-1]["bytes_done"] == 40_000
        assert events[-1]["percent"] == pytest.approx(40.0, abs=0.5)
        transfer_bar.reset(total=100_000)
        transfer_bar.update(10_000)
        assert events[-1]["bytes_done"] == 10_000
        cancel_ev.set()
        with pytest.raises(DownloadCancelledError):
            transfer_bar.update(10_000)


def test_background_install_cancel_and_ws_terminal_error_events(tmp_path: Path) -> None:
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime

    gate = threading.Event()
    fail_mode = {"on": False}
    installed = {"on": False}

    def _fake_install(*, on_progress=None, cancel_event=None) -> bool:
        if on_progress is not None:
            on_progress({"phase": "downloading", "bytes_done": 30_000_000, "bytes_total": 62_000_000, "percent": 48.4})
        deadline = time.monotonic() + 3.0
        while not gate.is_set() and time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelledError("Download cancelled.")
            time.sleep(0.01)
        if fail_mode["on"]:
            raise RuntimeError("Download failed (HTTP 502 Bad Gateway from us.aws.cdn.hf.co) — please retry.")
        installed["on"] = True
        return True

    async def _fake_install_async(fn, *, on_progress=None, cancel_event=None):
        return await asyncio.to_thread(fn, on_progress=on_progress, cancel_event=cancel_event)

    runtime.addons.register_pack_handler(
        "phonon_tts",
        is_installed=lambda: installed["on"],
        install=lambda on_progress=None, cancel_event=None: _fake_install_async(
            _fake_install, on_progress=on_progress, cancel_event=cancel_event
        ),
        uninstall=lambda: (installed.__setitem__("on", False) or True),
        available=True,
    )

    def _recv_json_bounded(ws: Any, timeout_s: float = 3.0) -> dict[str, Any]:
        box: dict[str, Any] = {}
        err: list[BaseException] = []

        def _reader() -> None:
            try:
                box["msg"] = ws.receive_json()
            except BaseException as exc:
                err.append(exc)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise TimeoutError(f"Timed out waiting {timeout_s}s for WebSocket JSON message")
        if err:
            raise err[0]
        return box["msg"]

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/session") as ws:
                init_state = _recv_json_bounded(ws)
                assert init_state["type"] == "session_state"
                assert "addons" in init_state
                assert isinstance(init_state["addons"]["revision"], int)
                assert isinstance(init_state["addons"]["server_epoch"], str)
                _ = _recv_json_bounded(ws)  # asr_status

                # 1. Start in background -> 202 immediately with monotonic revision (no stale profile)
                r_start = client.post("/api/addons/phonon_tts/install?background=true")
                assert r_start.status_code == 202
                body = r_start.json()
                assert body["background"] is True
                assert "profile" not in body
                start_rev = body["addons"]["revision"]
                pk = next(p for p in body["addons"]["packs"] if p["id"] == "phonon_tts")
                assert pk["state"] == "installing"

                # 2. Cancel cleanly returns to not_installed and emits WS terminal event
                r_cancel = client.post("/api/addons/phonon_tts/cancel")
                assert r_cancel.status_code == 200
                cancelled_msg = None
                for _ in range(12):
                    m = _recv_json_bounded(ws)
                    if m.get("type") == "addons_status" and m.get("event") == "cancelled":
                        cancelled_msg = m
                        break
                assert cancelled_msg is not None
                assert cancelled_msg["addons"]["revision"] > start_rev
                c_pk = next(p for p in cancelled_msg["addons"]["packs"] if p["id"] == "phonon_tts")
                assert c_pk["state"] == "not_installed"
                assert "error" not in c_pk

                # 3. Failed background install MUST emit terminal WS error event (never stuck on Downloading)
                fail_mode["on"] = True
                gate.set()
                r_fail = client.post("/api/addons/phonon_tts/install?background=true")
                assert r_fail.status_code == 202
                error_msg = None
                for _ in range(12):
                    m = _recv_json_bounded(ws)
                    if m.get("type") == "addons_status" and m.get("event") == "error":
                        error_msg = m
                        break
                assert error_msg is not None
                err_pk = next(p for p in error_msg["addons"]["packs"] if p["id"] == "phonon_tts")
                assert err_pk["state"] == "error"
                assert "502" in err_pk["error"]

                # 4. Retry succeeds and emits terminal WS installed event with higher revision
                fail_mode["on"] = False
                r_ok = client.post("/api/addons/phonon_tts/install?background=true")
                assert r_ok.status_code == 202
                ok_msg = None
                for _ in range(12):
                    m = _recv_json_bounded(ws)
                    if m.get("type") == "addons_status" and m.get("event") == "installed":
                        ok_msg = m
                        break
                assert ok_msg is not None
                assert ok_msg["addons"]["revision"] > error_msg["addons"]["revision"]
                ok_pk = next(p for p in ok_msg["addons"]["packs"] if p["id"] == "phonon_tts")
                assert ok_pk["state"] == "installed"
                assert ok_pk["installed"] is True
    finally:
        gate.set()


def test_early_cancel_before_worker_runs_is_honored(tmp_path: Path) -> None:
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    invoked = {"count": 0}

    def _never_called(*, on_progress=None, cancel_event=None) -> bool:
        invoked["count"] += 1
        return True

    runtime.addons.register_pack_handler(
        "phonon_tts",
        is_installed=lambda: False,
        install=_never_called,
        uninstall=lambda: True,
        available=True,
    )

    async def _scenario() -> None:
        runtime.addons.start_install_pack("phonon_tts")
        # Cancel synchronously on the same event loop tick before _run_install_locked starts
        runtime.addons.cancel_install("phonon_tts")
        task = runtime.addons._install_tasks.get("phonon_tts")
        if task is not None:
            await task
        pk = runtime.addons.get_pack_spec("phonon_tts")
        assert pk is not None
        assert pk["state"] == "not_installed"
        assert invoked["count"] == 0

    asyncio.run(_scenario())


def test_rampart_install_hook_preserves_concurrent_user_privacy_prefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.profiles.set_resume(
        filename="resume.txt",
        raw_text="Jordan Rivera\njordan@example.com\nSenior Engineer at Acme Corp\n",
    )

    def _fake_rampart_download(*, on_progress=None, cancel_event=None) -> dict[str, Any]:
        # Simulate user updating privacy prefs concurrently while Rampart finishes downloading
        cur = runtime.profiles.state.resume
        assert cur is not None
        updated = cur.prefs.model_copy(update={"custom_terms": ["Acme Corp"]})
        runtime.profiles.update_redaction_prefs(updated)
        return {"installed": True}

    monkeypatch.setattr(runtime.redactor, "ensure_downloaded", _fake_rampart_download)
    asyncio.run(runtime.addons.install_pack("rampart_pii"))

    final_resume = runtime.profiles.state.resume
    assert final_resume is not None
    assert "Acme Corp" in final_resume.prefs.custom_terms
    assert "Acme Corp" not in final_resume.redacted_text


def test_uninstall_fails_if_pack_remains_installed(tmp_path: Path) -> None:
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.addons.register_pack_handler(
        "phonon_tts",
        is_installed=lambda: True,
        install=lambda: True,
        uninstall=lambda: False,
        available=True,
    )
    with pytest.raises(RuntimeError, match="Could not completely remove"):
        asyncio.run(runtime.addons.uninstall_pack("phonon_tts"))
    pk = runtime.addons.get_pack_spec("phonon_tts")
    assert pk is not None
    assert pk["state"] == "error"
