from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .paths import is_loopback_host, resolve_data_root


def _is_loopback_host(host: str) -> bool:
    return is_loopback_host(host)


def _format_url_host(host: str) -> str:
    clean = (host or "").strip().strip("[]")
    if clean in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    if ":" in clean:
        return f"[{clean}]"
    return clean


def _open_browser_when_ready(
    *,
    probe_url: str,
    open_url: str,
    stop_event: threading.Event,
    timeout_s: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    step = 0.15
    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=0.5) as resp:
                if getattr(resp, "status", 0) == 200:
                    if not stop_event.is_set():
                        webbrowser.open(open_url)
                    return
        except Exception:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0 or stop_event.wait(min(step, remaining)):
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mockery",
        description="Mockery — Local-first Apple Silicon mock interview studio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve the local web studio (default: MOCKERY_PORT or 8765)",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Directory for sessions and .env.local (default: auto-detected repo or ~/Library/Application Support/Mockery)",
    )
    parser.add_argument(
        "--lan-token",
        default=None,
        help="Required auth token when binding a non-loopback --host (auto-generated if omitted)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web UI in your browser",
    )
    parser.add_argument(
        "--dump-machines",
        action="store_true",
        help="Print JSON specification of all application state machines and exit",
    )
    args = parser.parse_args()

    if args.dump_machines:
        from .state_machines import export_machine_specs

        print(json.dumps(export_machine_specs(), indent=2))
        return

    resolved_root = resolve_data_root(Path(args.data_dir) if args.data_dir else None)
    load_dotenv(resolved_root / ".env", override=False)
    load_dotenv(resolved_root / ".env.local", override=False)

    port = (
        int(args.port)
        if args.port is not None
        else int(os.getenv("MOCKERY_PORT") or os.getenv("MOCK_HELPER_PORT", "8765"))
    )
    raw_lan_token = args.lan_token if args.lan_token is not None else os.getenv("MOCKERY_LAN_TOKEN", "")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress HTTP client INFO logs and strip signed query strings from Hub logs.
    for _noisy_logger in ("httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
    from .downloads import sanitize_error_text

    class _RedactSignedUrlFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
                cleaned = sanitize_error_text(msg)
                if cleaned != msg:
                    record.msg = cleaned
                    record.args = ()
            except Exception:
                pass
            return True

    logging.getLogger("huggingface_hub").addFilter(_RedactSignedUrlFilter())

    lan_token: str | None = None
    if not _is_loopback_host(args.host):
        lan_token = (raw_lan_token or "").strip() or secrets.token_urlsafe(24)

    display_host = _format_url_host(args.host)
    base_url = f"http://{display_host}:{port}"
    url = f"{base_url}/?token={lan_token}" if lan_token else base_url
    probe_url = f"{base_url}/api/machines" + (f"?token={lan_token}" if lan_token else "")
    if lan_token:
        print(f"\n  Mockery listening on {args.host}:{port} (LAN token auth enabled)")
        print(f"  Open: {url}")
        print(f"  Data: {resolved_root}\n")
    else:
        print(f"\n  Mockery running at {url}")
        print(f"  Data: {resolved_root}\n")

    stop_browser_probe = threading.Event()
    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            kwargs={
                "probe_url": probe_url,
                "open_url": url,
                "stop_event": stop_browser_probe,
            },
            name="mockery-browser-open",
            daemon=True,
        ).start()

    from .server import create_app

    bind_host = args.host.strip().strip("[]") or "127.0.0.1"
    app = create_app(project_root=resolved_root, auth_token=lan_token)
    try:
        uvicorn.run(app, host=bind_host, port=port, log_level="info")
    finally:
        stop_browser_probe.set()


if __name__ == "__main__":
    main()
