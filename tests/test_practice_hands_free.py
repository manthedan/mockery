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


def test_a_long_message_is_read_in_pieces_and_nothing_is_dropped() -> None:
    res = _run_node(
        """
import { splitForSpeech } from "./src/mockery/static/voice_clip.js";
const sentence = "Consider the team, the commute, the pay and the work itself before you answer. ";
const text = "Welcome. " + sentence.repeat(40) + "Use `self.capacity` within 3.5 ms. Final constraint: one minute.";
const pieces = splitForSpeech(text, 800);
console.log(JSON.stringify({
  count: pieces.length,
  longest: Math.max(...pieces.map((p) => p.length)),
  complete: pieces.join(" ") === text.replace(/\\s+/g, " ").trim(),
  last: pieces.at(-1),
  midSentenceCuts: pieces.slice(0, -1).filter((p) => !/[.!?]$/.test(p)).length,
  short: splitForSpeech("One short question?", 800),
  unbroken: splitForSpeech("x".repeat(2000), 800).map((p) => p.length),
}));
"""
    )
    assert res["count"] > 1 and res["longest"] <= 800
    assert res["complete"] is True
    assert res["last"].endswith("Use `self.capacity` within 3.5 ms. Final constraint: one minute.")
    assert res["midSentenceCuts"] == 0
    assert res["short"] == ["One short question?"]
    assert res["unbroken"] == [800, 800, 400]


def test_the_voice_reports_its_text_limit_so_pieces_fit(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app
    from mockery.tts import MAX_TTS_CHARS

    client = TestClient(create_app(project_root=tmp_path, preload_asr=False))
    assert client.get("/api/status").json()["tts"]["max_chars"] == MAX_TTS_CHARS


def test_only_a_press_sends_a_take() -> None:
    studio = (STATIC / "practice_studio.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    # Reaching the take limit transcribes into the box; it never sends on its own.
    recorder = studio[studio.index("processor.onaudioprocess = ") : studio.index("source.connect(processor)")]
    assert "stopMicAndTranscribe()" in recorder
    assert "finishTake(" not in recorder and "submitTurn(" not in recorder

    # An automatic mic request is dropped if the tab was hidden or a dialog opened before access was granted.
    assert "(auto && !autoListenWelcome())" in studio
    assert 'toggleMicRecording("answer", { auto: true })' in studio

    # The retry drill can be typed too, and a failed transcription brings its box back.
    assert 'id="practiceRetryModeBtn"' in html and 'el("practiceRetryModeBtn")' in studio
    assert "transcriptionFailed" in studio
