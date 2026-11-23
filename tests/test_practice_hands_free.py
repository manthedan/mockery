"""Hands-free Practice answers: long takes are cut at pauses, and the UI contract holds."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "mockery" / "static"


def _run_node(script: str) -> dict:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_long_take_is_cut_at_the_pause_and_loses_no_audio() -> None:
    res = _run_node(
        """
import { splitClipForAsr } from "./src/mockery/static/voice_clip.js";
const rate = 16000;
const take = new Int16Array(rate * 250).fill(9000);
// One second of silence shortly before the 110-second limit, and another before 220.
take.fill(0, rate * 105, rate * 106);
take.fill(0, rate * 212, rate * 213);
const pieces = splitClipForAsr(take);
let offset = 0;
const cuts = [];
let contiguous = true;
for (const piece of pieces) {
  if (piece.byteOffset !== offset * 2) contiguous = false;
  offset += piece.length;
  cuts.push(offset / rate);
}
console.log(JSON.stringify({
  count: pieces.length,
  longest: Math.max(...pieces.map((p) => p.length)) / rate,
  total: offset / rate,
  contiguous,
  cuts,
  short: splitClipForAsr(new Int16Array(rate * 30)).length,
}));
"""
    )
    assert res["count"] == 3
    assert res["longest"] <= 110
    assert res["total"] == 250
    assert res["contiguous"] is True
    assert 105 <= res["cuts"][0] <= 106
    assert 212 <= res["cuts"][1] <= 213
    assert res["short"] == 1


def test_each_piece_fits_the_transcriber_upload_limit() -> None:
    from mockery.server import MAX_ASR_AUDIO_BYTES

    res = _run_node(
        """
import { ASR_PIECE_SECONDS, ASR_SAMPLE_RATE } from "./src/mockery/static/voice_clip.js";
console.log(JSON.stringify({ bytes: ASR_PIECE_SECONDS * ASR_SAMPLE_RATE * 2 }));
"""
    )
    assert res["bytes"] <= MAX_ASR_AUDIO_BYTES


def test_hands_free_controls_are_wired() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    studio = (STATIC / "practice_studio.js").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    for dom_id in (
        "practiceHandsFreeCheckbox",
        "practiceAnswerModeBtn",
        "practiceDiscardTakeBtn",
        "practiceRetryDiscardBtn",
    ):
        assert f'id="{dom_id}"' in html
        assert f'el("{dom_id}")' in studio

    # Space starts and sends a take, but never while typing in a field.
    assert 'ev.code === "Space"' in app
    # Only interviewer turns hand the floor to the mic; previews and replays do not.
    assert studio.count("thenListen: true") == 3
    assert ".is-voice-first .composer-dock .practice-textarea" in css
