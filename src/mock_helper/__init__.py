from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path.cwd() / ".env.local", override=False)

    parser = argparse.ArgumentParser(
        description="Mock Helper — Local Apple Silicon Mock Interview Copilot"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MOCK_HELPER_PORT", "8765")),
        help="Port to serve the local web studio (default: 8765)",
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

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Mock Helper running at {url}\n")

    if not args.no_browser:
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()

    from .server import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
