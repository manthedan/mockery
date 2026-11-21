from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import threading
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .paths import resolve_data_root

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    data_root = resolve_data_root()
    load_dotenv(data_root / ".env", override=False)
    load_dotenv(data_root / ".env.local", override=False)

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
        default=int(os.getenv("MOCKERY_PORT") or os.getenv("MOCK_HELPER_PORT", "8765")),
        help="Port to serve the local web studio (default: 8765)",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Directory for sessions and .env.local (default: auto-detected repo or ~/Library/Application Support/Mockery)",
    )
    parser.add_argument(
        "--lan-token",
        default=os.getenv("MOCKERY_LAN_TOKEN", ""),
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    resolved_root = resolve_data_root(Path(args.data_dir) if args.data_dir else None)

    lan_token: str | None = None
    if args.host not in LOOPBACK_HOSTS:
        lan_token = (args.lan_token or "").strip() or secrets.token_urlsafe(24)

    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}"
    if lan_token:
        url = f"{url}/?token={lan_token}"
        print(f"\n  Mockery listening on {args.host}:{args.port} (LAN token auth enabled)")
        print(f"  Open: {url}")
        print(f"  Data: {resolved_root}\n")
    else:
        print(f"\n  Mockery running at {url}")
        print(f"  Data: {resolved_root}\n")

    if not args.no_browser:
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()

    from .server import create_app

    app = create_app(project_root=resolved_root, auth_token=lan_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
