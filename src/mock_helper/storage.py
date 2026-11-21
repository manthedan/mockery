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
    headers periodically (every `header_sync_interval_s`) and unconditionally on `close()`,
    avoiding 4 seeks + disk flushes on every 128ms WebSocket chunk while never truncating prior audio.
    """

    def __init__(
        self,
        sessions_dir: Path,
        session_id: str,
        *,
        header_sync_interval_s: float = 2.0,
    ) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.header_sync_interval_s = header_sync_interval_s
        self._handles: dict[str, BinaryIO] = {}
        self._data_sizes: dict[str, int] = {}
        self._dirty: dict[str, bool] = {}
        self._last_sync: dict[str, float] = {}

    def _sync_header(self, stream_name: str) -> None:
        fh = self._handles.get(stream_name)
        if fh is None or not self._dirty.get(stream_name, False):
            return
        data_size = self._data_sizes.get(stream_name, 0)
        fh.seek(4)
        fh.write(struct.pack("<I", 36 + data_size))
        fh.seek(40)
        fh.write(struct.pack("<I", data_size))
        fh.flush()
        fh.seek(0, 2)
        self._dirty[stream_name] = False
        self._last_sync[stream_name] = time.monotonic()

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
                fh.seek(0, 2)
                self._data_sizes[stream_name] = max(0, fh.tell() - WAV_HEADER_BYTES)
            else:
                fh = wav_path.open("w+b")
                fh.write(_make_wav_header(0, SAMPLE_RATE))
                self._data_sizes[stream_name] = 0
            self._handles[stream_name] = fh
            self._dirty[stream_name] = False
            self._last_sync[stream_name] = time.monotonic()

        fh.write(pcm)
        self._data_sizes[stream_name] += len(pcm)
        self._dirty[stream_name] = True

        if (time.monotonic() - self._last_sync[stream_name]) >= self.header_sync_interval_s:
            self._sync_header(stream_name)

    def close(self) -> None:
        for stream_name, fh in list(self._handles.items()):
            try:
                self._sync_header(stream_name)
                fh.close()
            except Exception:
                pass
        self._handles.clear()
        self._data_sizes.clear()
        self._dirty.clear()
        self._last_sync.clear()


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
        metrics = session.compute_metrics()
        _atomic_write_text(json_path, session.model_dump_json(indent=2))
        _atomic_write_text(md_path, session.to_markdown(audience="interviewer", metrics=metrics))
        _atomic_write_text(cand_md_path, session.to_markdown(audience="candidate", metrics=metrics))
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
                if not isinstance(data, dict) or not data.get("id"):
                    continue
                cfg = data.get("config") or {}
                utts = data.get("utterances") or []
                if utts:
                    min_s = min(int(u.get("start_ms", 0)) for u in utts)
                    max_e = max(int(u.get("end_ms", 0)) for u in utts)
                    dur_s = round(max(0.0, (max_e - min_s) / 1000.0), 1)
                else:
                    dur_s = 0.0
                items.append(
                    {
                        "id": str(data["id"]),
                        "created_at_iso": str(data.get("created_at_iso", "")),
                        "role_title": str(cfg.get("role_title", "Senior Software Engineer")),
                        "interview_type": str(cfg.get("interview_type", "behavioral")),
                        "mode": str(cfg.get("mode", "remote_meet")),
                        "qa_count": len(data.get("qa_groups") or []),
                        "duration_seconds": dur_s,
                        "has_calibration": bool(data.get("calibration_markdown")),
                    }
                )
            except Exception:
                continue
        return items
