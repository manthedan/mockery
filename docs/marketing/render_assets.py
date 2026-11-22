"""Regenerate Mockery's marketing assets from the real app.

Run from the repo root (Pillow is only needed here, so it is pulled in ad hoc):

    uv run python scripts/verify_ui.py snapshots --out-dir docs/assets/screenshots
    uv run --with pillow python docs/marketing/render_assets.py all

Steps (each is also a subcommand):

    shots   Capture the screens `verify_ui.py snapshots` does not cover: the
            practice launchpad and the Live Interviewer Copilot workspace
            (follow-up probes + blind calibration), Dark + Light.
    render  Rasterise the SVG sources in docs/assets (social card, hero
            showcases, icons) to PNG with headless Chrome.
    site    Copy everything the landing page needs into site/assets so that
            site/ stays self-contained for GitHub Pages.

Not part of `all`:

    stage   Serve the same synthetic demo data at http://127.0.0.1:8765 until
            Ctrl-C, for screen recordings (see docs/marketing/video-brief.md).

Every name, company and transcript line below is synthetic demo data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "docs" / "assets"
SHOTS = ASSETS / "screenshots"
SITE_ASSETS = REPO / "site" / "assets"

sys.path.insert(0, str(REPO / "scripts"))

import verify_ui as V  # noqa: E402

COPILOT_SESSION_ID = "20261120_101500_fin2"

# (id, speaker, start mm:ss, end mm:ss, text)
TURNS = [
    (
        "utt_01",
        "interviewer",
        (0, 6),
        (0, 19),
        "Let's design a webhook delivery service for a payments platform. Merchants register endpoints, and we owe them at-least-once delivery with retries and a dead-letter queue.",
    ),
    (
        "utt_02",
        "candidate",
        (0, 24),
        (1, 2),
        "Before I draw anything — what's the volume, and do merchants need ordering per account, or is best-effort ordering acceptable as long as every event carries a sequence number?",
    ),
    (
        "utt_03",
        "interviewer",
        (1, 6),
        (1, 21),
        "Call it forty thousand events a second at peak. Ordering per merchant account matters for balance events, nothing else.",
    ),
    (
        "utt_04",
        "candidate",
        (1, 27),
        (2, 48),
        "Then I'd scope it as: durable ingest, per-account ordered partitions for balance events only, signed payloads, and a delivery SLO of ninety-nine percent under thirty seconds.",
    ),
    (
        "utt_05",
        "interviewer",
        (4, 52),
        (5, 3),
        "Good. Walk me through the write path from the ledger commit to the first delivery attempt.",
    ),
    (
        "utt_06",
        "candidate",
        (5, 9),
        (6, 44),
        "The ledger commit writes an outbox row in the same transaction. A relay tails the outbox into a partitioned log keyed by merchant account, so ingestion never depends on a merchant endpoint being healthy.",
    ),
    (
        "utt_07",
        "candidate",
        (6, 50),
        (8, 31),
        "Delivery workers pull per partition, sign the payload, and POST with a five-second timeout. Failures go to a retry topic with exponential backoff and full jitter, capped at twenty-four hours.",
    ),
    (
        "utt_08",
        "interviewer",
        (8, 36),
        (8, 47),
        "What stops one slow merchant from stalling everyone else on that partition?",
    ),
    (
        "utt_09",
        "candidate",
        (8, 53),
        (10, 40),
        "Head-of-line blocking is the real risk. I'd give each endpoint a circuit breaker and park its events in a per-endpoint retry lane, so the shared partition keeps draining.",
    ),
    (
        "utt_10",
        "interviewer",
        (19, 2),
        (19, 14),
        "Say an endpoint is down for six hours. What does the merchant see when they come back?",
    ),
    (
        "utt_11",
        "candidate",
        (19, 20),
        (21, 5),
        "After the retry budget is spent, events land in a dead-letter store with the last response code. The merchant gets a replay API scoped by time range, and replays reuse the original event ID so their handler stays idempotent.",
    ),
    (
        "utt_12",
        "interviewer",
        (21, 10),
        (21, 19),
        "How would you know the dead-letter queue is growing before a merchant tells you?",
    ),
    (
        "utt_13",
        "candidate",
        (21, 25),
        (22, 58),
        "I'd alert on dead-letter depth per endpoint and on age of the oldest undelivered event. Honestly I haven't sized the storage for a multi-day outage — I'd want to model that.",
    ),
    (
        "utt_14",
        "candidate",
        (23, 4),
        (24, 12),
        "The tradeoff I'm making is latency for isolation: per-endpoint lanes cost more queues, but one bad merchant can't take down delivery for the rest.",
    ),
]

# (id, index, title, main question, utterance ids)
CARDS = [
    (
        "qa_p1",
        1,
        "Requirements & Delivery Guarantees",
        "Design a webhook delivery service with retries and a dead-letter queue.",
        ["utt_01", "utt_02", "utt_03", "utt_04"],
    ),
    (
        "qa_p2",
        2,
        "Ingestion Path & Retry Lanes",
        "Walk me through the write path from the ledger commit to the first delivery attempt.",
        ["utt_05", "utt_06", "utt_07", "utt_08", "utt_09"],
    ),
    (
        "qa_p3",
        3,
        "Dead-Letter Queue & Replay",
        "An endpoint is down for six hours. What does the merchant see when they come back?",
        ["utt_10", "utt_11", "utt_12", "utt_13", "utt_14"],
    ),
]

# card id -> (quick read, [(angle, question, why)])
PROBES = {
    "qa_p2": (
        "Clean outbox-to-log write path and a real answer to head-of-line blocking. Backpressure and retry-storm behaviour are still unexplored.",
        [
            (
                "Failure mode",
                "When ten thousand endpoints recover at the same minute, what keeps the retry lanes from stampeding them?",
                "Backoff with jitter was named, but not what happens when it all expires together.",
            ),
            (
                "Scale",
                "How many per-endpoint lanes can one worker pool hold before the queue count itself becomes the bottleneck?",
                "Isolation was chosen without a cost estimate.",
            ),
            (
                "Tradeoff",
                "What would you give up to keep strict per-account ordering during a partial outage?",
                "Ordering was promised for balance events but never stress-tested.",
            ),
        ],
    ),
    "qa_p3": (
        "Replay reuses the original event ID, which is the right instinct. Storage sizing was openly left unmodelled.",
        [
            (
                "Depth",
                "Size the dead-letter store for a three-day outage at forty thousand events a second. Where does it live?",
                "The candidate flagged this gap themselves at 21:25.",
            ),
            (
                "Operations",
                "Who gets paged when the oldest undelivered event is two hours old — you or the merchant?",
                "Alerting was described, ownership was not.",
            ),
            (
                "Tradeoff",
                "When would you drop events instead of dead-lettering them?",
                "Tests whether at-least-once is a principle or a default.",
            ),
        ],
    ),
}

CALIBRATION_MD = """### 1. Calibration Check: Your Blind Call vs. Transcript Evidence
Your blind call was `lean_hire`. The transcript evidence supports **Lean Hire for Senior**, so your gut read is well calibrated. The candidate scoped before designing `[00:24, utt_02]`, chose a transactional outbox unprompted `[05:09, utt_06]`, and named head-of-line blocking before being led to it `[08:53, utt_09]`. The one explored gap is capacity: dead-letter storage was never sized `[21:25, utt_13]`. Retry-storm behaviour went unprobed, which is a pacing note for you rather than a mark against the candidate.

### 2. Candidate Signal & Evidence Audit (By Card)
- **P1: Requirements & Delivery Guarantees — Strong**
  Asked for volume and ordering scope before drawing `[00:24, utt_02]`, then committed to a measurable delivery SLO `[01:27, utt_04]`.
- **P2: Ingestion Path & Retry Lanes — Strong**
  Outbox in the ledger transaction keeps ingest independent of merchant health `[05:09, utt_06]`. Per-endpoint circuit breakers answer the slow-merchant question directly `[08:53, utt_09]`.
- **P3: Dead-Letter Queue & Replay — Mixed**
  Replay reuses the original event ID so handlers stay idempotent `[19:20, utt_11]`, but storage for a multi-day outage was left unmodelled `[21:25, utt_13]`.

### 3. Interviewer Probing & Pacing Retro
- Your talk share stayed low, which left the candidate room to drive every card.
- **Missed Probing Moments:** At `[06:50, utt_07]` the candidate capped backoff at twenty-four hours — ask what happens when every lane expires in the same minute. At `[23:04, utt_14]` they named the isolation tradeoff — ask for the queue-count cost of that choice.

### 4. Constructive Takeaways for Candidate
- Keep opening with scope questions; it set up every later decision.
- Put a number on storage before the interviewer asks for one.
- Say the failure mode out loud for each component as you draw it.
"""

# SVG source -> (PNG name, CSS width, CSS height, device scale, transparent background)
RENDERS = [
    ("og-social-card.svg", "og-social-card.png", 1280, 640, 1, False),
    ("hero-dark.svg", "hero-dark.png", 1280, 848, 2, True),
    ("hero-light.svg", "hero-light.png", 1280, 848, 2, True),
    ("logo-mark.svg", "logo-mark-180.png", 180, 180, 1, True),
    ("logo-mark.svg", "logo-mark-32.png", 32, 32, 1, True),
]

# Screens the landing page shows, exported as WebP at 1.5x of the 1440px capture.
SITE_SHOTS = [
    "04_behavioral",
    "07_debrief",
    "05_whiteboard",
    "06_coding",
    "02_profile",
    "08_copilot_probes",
    "09_copilot_calibration",
]
SITE_SHOT_WIDTH = 2160
SITE_COPIES = [
    "logo-mark.svg",
    "logo-mark-180.png",
    "logo-mark-32.png",
    "wordmark-on-dark.svg",
    "wordmark-on-light.svg",
    "og-social-card.png",
]


def _ms(minutes: int, seconds: int) -> int:
    return (minutes * 60 + seconds) * 1000


def build_copilot_session():
    from mockery.models import (
        DiveDeeperResponse,
        InterviewerSelfRating,
        InterviewSession,
        ProbeSuggestion,
        QAGroup,
        SessionConfig,
        Utterance,
    )

    utterances = [
        Utterance(id=uid, speaker=speaker, text=text, start_ms=_ms(*start), end_ms=_ms(*end))
        for uid, speaker, start, end, text in TURNS
    ]
    by_id = {u.id: u for u in utterances}
    groups = []
    for qa_id, index, title, question, utt_ids in CARDS:
        history = []
        if qa_id in PROBES:
            quick_read, probes = PROBES[qa_id]
            history.append(
                DiveDeeperResponse(
                    qa_id=qa_id,
                    quick_read=quick_read,
                    probes=[ProbeSuggestion(angle=a, question=q, why=w) for a, q, w in probes],
                    model_used="local",
                    based_on_utt_count=len(utt_ids),
                )
            )
        groups.append(
            QAGroup(
                id=qa_id,
                index=index,
                card_kind="phase",
                title=title,
                main_question=question,
                utterance_ids=utt_ids,
                dive_deeper_history=history,
                started_ms=by_id[utt_ids[0]].start_ms,
                updated_ms=by_id[utt_ids[-1]].end_ms,
            )
        )
    return InterviewSession(
        id=COPILOT_SESSION_ID,
        created_at_iso="2026-11-20T10:15:00-0800",
        elapsed_capture_ms=_ms(24, 30),
        config=SessionConfig(
            role_title="Senior Backend Engineer (L5)",
            interview_type="system_design",
            mode="remote_meet",
            rubric_notes="ownership, failure modes, scale",
        ),
        utterances=utterances,
        qa_groups=groups,
        self_rating=InterviewerSelfRating(
            hire_signal="lean_hire",
            perceived_level="Senior (L5)",
            gut_rationale="Structured, scoped first, strong on isolation. Thin on capacity numbers",
            biggest_concern="Never sized the dead-letter store.",
        ),
        calibration_markdown=CALIBRATION_MD,
    )


async def capture_shots() -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    chrome_proc, chrome_profile, ws_url = await V.launch_chrome()
    seeded_dir = Path(tempfile.mkdtemp(prefix="mockery_marketing_"))
    server = None
    try:
        V.seed_demo_data_dir(seeded_dir, offline_llm=False)
        V.SessionStore(seeded_dir / "sessions").save(build_copilot_session())
        port = V.find_free_port()
        server = V.start_mockery_server(seeded_dir, port)

        async with V.CDPClient(ws_url, SHOTS) as cdp:
            await cdp.set_viewport(1440, 900)
            await cdp.navigate(f"http://127.0.0.1:{port}")
            await asyncio.sleep(0.9)

            async def both_themes(name: str) -> None:
                # Same UI state for both captures, so Dark and Light line up pixel for pixel.
                for theme in ("dark", "light"):
                    await cdp.eval(f"document.documentElement.setAttribute('data-theme', {json.dumps(theme)})")
                    await asyncio.sleep(0.35)
                    await cdp.screenshot(f"{name}_{theme}.png")

            async def click_mode(pattern: str) -> None:
                await cdp.eval(
                    f"[...document.querySelectorAll('#homeModeSwitch button')].find(b => /{pattern}/i.test(b.innerText))?.click()"
                )

            await cdp.press_key("Escape", "Escape")
            await click_mode("practice")
            await asyncio.sleep(0.5)
            await both_themes("00_home_practice")

            await click_mode("copilot")
            await asyncio.sleep(0.3)
            await cdp.eval(f"""document.querySelector('[data-load-session="{COPILOT_SESSION_ID}"]')?.click()""")
            await asyncio.sleep(1.0)
            await cdp.eval("document.getElementById('tabCalibration')?.click()")
            await asyncio.sleep(0.3)
            await cdp.eval("""document.querySelector('.cite-link[data-cite-utt="utt_02"]')?.click()""")
            await asyncio.sleep(0.6)
            await both_themes("09_copilot_calibration")

            await cdp.eval("document.getElementById('tabProbes')?.click()")
            await asyncio.sleep(0.3)
            await cdp.eval("""document.querySelector('[data-jump-qa="qa_p3"]')?.click()""")
            await asyncio.sleep(0.9)
            await both_themes("08_copilot_probes")

            problems = cdp.exceptions + cdp.console_errors
            if problems:
                print(f"Capture hit page errors: {problems}", file=sys.stderr)
                return 1
    finally:
        if server:
            server.terminate()
        chrome_proc.terminate()
        shutil.rmtree(chrome_profile, ignore_errors=True)
        shutil.rmtree(seeded_dir, ignore_errors=True)
    print(f"Captured practice launchpad + Copilot screens (Dark + Light) in {SHOTS}")
    return 0


def stage_demo(port: int) -> int:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"Port {port} is already in use. Quit the running Mockery first, or pass --port.", file=sys.stderr)
            return 1
    seeded_dir = Path(tempfile.mkdtemp(prefix="mockery_stage_"))
    server = None
    try:
        V.seed_demo_data_dir(seeded_dir, offline_llm=False)
        V.SessionStore(seeded_dir / "sessions").save(build_copilot_session())
        server = V.start_mockery_server(seeded_dir, port)
        print(
            f"Synthetic demo studio at http://127.0.0.1:{port} (data in {seeded_dir}). Ctrl-C to stop and delete it.",
            flush=True,
        )
        server.wait()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if server:
            server.terminate()
        shutil.rmtree(seeded_dir, ignore_errors=True)
    return 0


def render_svgs() -> int:
    from PIL import Image

    chrome = V.find_chrome()
    for src, dest, width, height, scale, transparent in RENDERS:
        out = ASSETS / dest
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={scale}",
            f"--window-size={width},{height}",
            f"--screenshot={out}",
        ]
        if transparent:
            cmd.append("--default-background-color=00000000")
        cmd.append((ASSETS / src).as_uri())
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with Image.open(out) as im:
            expected = (width * scale, height * scale)
            if im.size != expected:
                print(f"{dest}: rendered {im.size}, expected {expected}", file=sys.stderr)
                return 1
            im.save(out, optimize=True)
        print(f"{dest}: {expected[0]}x{expected[1]}, {out.stat().st_size // 1024} KB")
    return 0


def sync_site() -> int:
    from PIL import Image

    shots_out = SITE_ASSETS / "shots"
    shots_out.mkdir(parents=True, exist_ok=True)
    for name in SITE_COPIES:
        shutil.copyfile(ASSETS / name, SITE_ASSETS / name)
    total = 0
    for stem in SITE_SHOTS:
        for theme in ("dark", "light"):
            with Image.open(SHOTS / f"{stem}_{theme}.png") as im:
                height = round(im.height * SITE_SHOT_WIDTH / im.width)
                resized = im.convert("RGB").resize((SITE_SHOT_WIDTH, height), Image.LANCZOS)
                out = shots_out / f"{stem}_{theme}.webp"
                resized.save(out, "WEBP", quality=90, method=6)
                total += out.stat().st_size
    print(f"site/assets: {len(SITE_COPIES)} brand files + {len(SITE_SHOTS) * 2} WebP screens ({total // 1024} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Mockery marketing assets.")
    parser.add_argument("step", choices=("shots", "render", "site", "all", "stage"))
    parser.add_argument("--port", type=int, default=8765, help="Port for `stage` (default 8765).")
    args = parser.parse_args()
    step = args.step
    if step == "stage":
        return stage_demo(args.port)
    if step in ("shots", "all") and (code := asyncio.run(capture_shots())):
        return code
    if step in ("render", "all") and (code := render_svgs()):
        return code
    if step in ("site", "all") and (code := sync_site()):
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
