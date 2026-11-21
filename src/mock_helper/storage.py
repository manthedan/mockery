from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any, BinaryIO

from .models import InterviewSession, Utterance
from .sources import CHANNEL_CANDIDATE, CHANNEL_INTERVIEWER, CHANNEL_ROOM_MIC
from .vad import SAMPLE_RATE

WAV_HEADER_BYTES = 44


def _make_wav_header(data_size: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Builds a standard 44-byte RIFF PCM 16-bit mono WAV header."""
    byte_rate = sample_rate * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        1,  # mono
        sample_rate,
        byte_rate,
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )


class SessionAudioArchive:
    """
    Crash-safe resumable WAV recorder (16kHz mono PCM per stream).
    Opens existing `.wav` files in `r+b` append mode and updates the RIFF/data length
    headers in place, so Stop->Start cycles or WebSocket reconnects NEVER truncate prior audio.
    """

    def __init__(self, sessions_dir: Path, session_id: str) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._handles: dict[str, BinaryIO] = {}

    def append_frame(self, payload: bytes) -> None:
        if len(payload) < 3:
            return
        ch = payload[0]
        pcm = payload[1:]
        if len(pcm) % 2 != 0:
            pcm = pcm[:-1]
        if not pcm:
            return

        stream_name = (
            "interviewer"
            if ch == CHANNEL_INTERVIEWER
            else ("candidate" if ch == CHANNEL_CANDIDATE else ("room" if ch == CHANNEL_ROOM_MIC else None))
        )
        if stream_name is None:
            return

        fh = self._handles.get(stream_name)
        if fh is None:
            wav_path = self.sessions_dir / f"{self.session_id}_{stream_name}.wav"
            if wav_path.exists() and wav_path.stat().st_size >= WAV_HEADER_BYTES:
                fh = wav_path.open("r+b")
            else:
                fh = wav_path.open("w+b")
                fh.write(_make_wav_header(0, SAMPLE_RATE))
            self._handles[stream_name] = fh

        fh.seek(0, 2)
        fh.write(pcm)
        total_size = fh.tell()
        data_size = max(0, total_size - WAV_HEADER_BYTES)
        # Update RIFF chunk size (offset 4) and data chunk size (offset 40) in O(1)
        fh.seek(4)
        fh.write(struct.pack("<I", 36 + data_size))
        fh.seek(40)
        fh.write(struct.pack("<I", data_size))
        fh.flush()

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._handles.clear()


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


class SessionStore:
    """
    Persists sessions with:
    1. Append-only crash-safe `<session_id>.events.jsonl` log written on every utterance upsert.
    2. Atomic full state snapshot `<session_id>.json` + interviewer & candidate `.md` exports.
    """

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def append_utterance_event(self, session_id: str, utt: Utterance) -> Path:
        safe_id = Path(session_id).name
        jsonl_path = self.sessions_dir / f"{safe_id}.events.jsonl"
        record = {
            "logged_at_ms": int(time.time() * 1000),
            "type": "utterance_upsert",
            "utterance": utt.model_dump(),
        }
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
        return jsonl_path

    def save(self, session: InterviewSession) -> tuple[Path, Path]:
        safe_id = Path(session.id).name
        json_path = self.sessions_dir / f"{safe_id}.json"
        md_path = self.sessions_dir / f"{safe_id}.md"
        cand_md_path = self.sessions_dir / f"{safe_id}.candidate.md"
        _atomic_write_text(json_path, session.model_dump_json(indent=2))
        _atomic_write_text(md_path, session.to_markdown(audience="interviewer"))
        _atomic_write_text(cand_md_path, session.to_markdown(audience="candidate"))
        return json_path, md_path

    def _recover_from_jsonl(self, safe_id: str) -> InterviewSession | None:
        jsonl_path = self.sessions_dir / f"{safe_id}.events.jsonl"
        if not jsonl_path.exists():
            return None
        from .grouper import QAGrouper

        session = InterviewSession(id=safe_id)
        grouper = QAGrouper()
        recovered_any = False
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "utterance_upsert" and rec.get("utterance"):
                    utt = Utterance.model_validate(rec["utterance"])
                    grouper.upsert_and_route(session, utt)
                    recovered_any = True
            except Exception:
                continue
        return session if recovered_any else None

    def load(self, session_id: str) -> InterviewSession | None:
        safe_id = Path(session_id).name
        json_path = self.sessions_dir / f"{safe_id}.json"
        if json_path.exists():
            try:
                return InterviewSession.model_validate_json(json_path.read_text(encoding="utf-8"))
            except Exception:
                return self._recover_from_jsonl(safe_id)
        return self._recover_from_jsonl(safe_id)

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sess = InterviewSession.model_validate(data)
                metrics = sess.compute_metrics()
                items.append(
                    {
                        "id": sess.id,
                        "created_at_iso": sess.created_at_iso,
                        "role_title": sess.config.role_title,
                        "interview_type": sess.config.interview_type,
                        "mode": sess.config.mode,
                        "qa_count": metrics.qa_count,
                        "duration_seconds": metrics.duration_seconds,
                        "has_calibration": bool(sess.calibration_markdown),
                    }
                )
            except Exception:
                continue
        return items
