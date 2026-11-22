from __future__ import annotations

import json
import logging
import struct
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

from .models import InterviewSession, Utterance
from .sources import CHANNEL_CANDIDATE, CHANNEL_INTERVIEWER, CHANNEL_ROOM_MIC
from .vad import SAMPLE_RATE

logger = logging.getLogger(__name__)

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
    1. Append-only crash-safe `<session_id>.events.jsonl` log written on utterance upserts.
    2. Atomic full state snapshot `<session_id>.json` + interviewer & candidate `.md` exports.
    3. Fast non-blocking coalesced `save_live()` for per-utterance live capture (compact JSON
       written on a single background thread; Markdown exports written on milestones/export).
    """

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._bg_cond = threading.Condition()
        self._bg_pending: dict[str, tuple[int, str]] = {}
        self._bg_seq: dict[str, int] = {}
        self._bg_in_flight: bool = False
        self._bg_closed: bool = False
        self._bg_thread: threading.Thread | None = None
        self.bg_write_errors: int = 0
        self.last_bg_write_error: str | None = None

    def _current_journal_offset(self, safe_id: str) -> int:
        jsonl_path = self.sessions_dir / f"{safe_id}.events.jsonl"
        try:
            return jsonl_path.stat().st_size
        except FileNotFoundError:
            return 0

    def _ensure_bg_worker_locked(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._bg_closed = False
        self._bg_thread = threading.Thread(
            target=self._bg_writer_loop,
            name="session-store-writer",
            daemon=True,
        )
        self._bg_thread.start()

    def _bg_writer_loop(self) -> None:
        while True:
            with self._bg_cond:
                while not self._bg_pending and not self._bg_closed:
                    self._bg_cond.wait(timeout=5.0)
                    if not self._bg_pending:
                        # Exit cleanly when idle so short-lived stores never linger
                        self._bg_thread = None
                        return
                if not self._bg_pending and self._bg_closed:
                    self._bg_thread = None
                    return
                safe_id, (seq, payload) = self._bg_pending.popitem()
                self._bg_in_flight = True

            try:
                with self._write_lock:
                    with self._bg_cond:
                        latest_seq = self._bg_seq.get(safe_id, 0)
                    if seq == latest_seq:
                        json_path = self.sessions_dir / f"{safe_id}.json"
                        _atomic_write_text(json_path, payload)
            except Exception as exc:
                self.bg_write_errors += 1
                self.last_bg_write_error = f"{safe_id}: {exc}"
                logger.exception("Background live save failed for session %s", safe_id)
            finally:
                with self._bg_cond:
                    self._bg_in_flight = False
                    if not self._bg_pending:
                        self._bg_cond.notify_all()

    def flush_pending(self) -> None:
        """Blocks until all coalesced background `save_live()` snapshots have hit disk."""
        with self._bg_cond:
            while self._bg_pending or self._bg_in_flight:
                self._bg_cond.wait(timeout=0.05)

    def close(self) -> None:
        with self._bg_cond:
            self._bg_closed = True
            self._bg_cond.notify_all()
            worker = self._bg_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    def append_utterance_event(self, session_id: str, utt: Utterance) -> Path:
        safe_id = Path(session_id).name
        jsonl_path = self.sessions_dir / f"{safe_id}.events.jsonl"
        record = {
            "logged_at_ms": int(time.time() * 1000),
            "type": "utterance_upsert",
            "utterance": utt.model_dump(),
        }
        line_bytes = (json.dumps(record) + "\n").encode("utf-8")
        with jsonl_path.open("ab") as fh:
            fh.write(line_bytes)
            fh.flush()
        return jsonl_path

    def save_live(self, session: InterviewSession) -> None:
        """
        Non-blocking live autosave: snapshots compact JSON immediately on the caller thread
        (race-free) and coalesces the atomic disk write onto a single background worker thread,
        skipping expensive dual-Markdown regeneration until session milestones/export.
        """
        safe_id = Path(session.id).name
        session.journal_offset = self._current_journal_offset(safe_id)
        payload = session.model_dump_json()
        with self._bg_cond:
            seq = self._bg_seq.get(safe_id, 0) + 1
            self._bg_seq[safe_id] = seq
            self._bg_pending[safe_id] = (seq, payload)
            self._ensure_bg_worker_locked()
            self._bg_cond.notify_all()

    def save(
        self,
        session: InterviewSession,
        *,
        write_markdown: bool = True,
        indent: int | None = 2,
    ) -> tuple[Path, Path]:
        safe_id = Path(session.id).name
        json_path = self.sessions_dir / f"{safe_id}.json"
        md_path = self.sessions_dir / f"{safe_id}.md"
        cand_md_path = self.sessions_dir / f"{safe_id}.candidate.md"
        session.journal_offset = self._current_journal_offset(safe_id)
        json_payload = session.model_dump_json(indent=indent)
        md_interviewer: str | None = None
        md_candidate: str | None = None
        if write_markdown:
            metrics = session.compute_metrics()
            md_interviewer = session.to_markdown(audience="interviewer", metrics=metrics)
            md_candidate = session.to_markdown(audience="candidate", metrics=metrics)

        with self._bg_cond:
            # Supersede any queued background snapshot for this session
            self._bg_seq[safe_id] = self._bg_seq.get(safe_id, 0) + 1
            self._bg_pending.pop(safe_id, None)

        with self._write_lock:
            _atomic_write_text(json_path, json_payload)
            if md_interviewer is not None and md_candidate is not None:
                _atomic_write_text(md_path, md_interviewer)
                _atomic_write_text(cand_md_path, md_candidate)
        return json_path, md_path

    def _replay_journal_into(self, session: InterviewSession, safe_id: str, *, offset: int = 0) -> int:
        jsonl_path = self.sessions_dir / f"{safe_id}.events.jsonl"
        if not jsonl_path.exists():
            return 0
        file_size = jsonl_path.stat().st_size
        if offset < 0 or offset >= file_size:
            return 0
        from .grouper import QAGrouper

        grouper = QAGrouper()
        replayed = 0
        with jsonl_path.open("rb") as fh:
            fh.seek(offset)
            raw_tail = fh.read()
            new_offset = fh.tell()

        for raw_line in raw_tail.splitlines():
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "utterance_upsert" and rec.get("utterance"):
                    utt = Utterance.model_validate(rec["utterance"])
                    grouper.upsert_and_route(session, utt)
                    replayed += 1
            except Exception:
                continue
        if replayed > 0:
            grouper.sync_card_kinds(session)
            session.journal_offset = new_offset
        return replayed

    def _recover_from_jsonl(self, safe_id: str) -> InterviewSession | None:
        jsonl_path = self.sessions_dir / f"{safe_id}.events.jsonl"
        if not jsonl_path.exists():
            return None
        session = InterviewSession(id=safe_id)
        replayed = self._replay_journal_into(session, safe_id, offset=0)
        return session if replayed > 0 else None

    def load(self, session_id: str) -> InterviewSession | None:
        self.flush_pending()
        safe_id = Path(session_id).name
        json_path = self.sessions_dir / f"{safe_id}.json"
        if json_path.exists():
            try:
                session = InterviewSession.model_validate_json(json_path.read_text(encoding="utf-8"))
                if session.journal_offset is not None:
                    self._replay_journal_into(session, safe_id, offset=session.journal_offset)
                return session
            except Exception:
                return self._recover_from_jsonl(safe_id)
        return self._recover_from_jsonl(safe_id)

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]:
        self.flush_pending()
        # Union snapshot (*.json) and journal-only (*.events.jsonl) IDs so crashed sessions stay recoverable.
        candidate_ids: set[str] = set()
        for path in self.sessions_dir.glob("*.json"):
            candidate_ids.add(path.stem)
        for path in self.sessions_dir.glob("*.events.jsonl"):
            name = path.name
            if name.endswith(".events.jsonl"):
                candidate_ids.add(name[: -len(".events.jsonl")])

        items: list[dict[str, Any]] = []
        for sid in sorted(candidate_ids, reverse=True):
            if len(items) >= limit:
                break
            json_path = self.sessions_dir / f"{sid}.json"
            data: dict[str, Any] | None = None
            recovered = False
            if json_path.is_file():
                try:
                    parsed = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict) and parsed.get("id"):
                        data = parsed
                except Exception:
                    data = None
            if data is None:
                sess = self._recover_from_jsonl(sid)
                if sess is None:
                    continue
                data = sess.model_dump()
                recovered = True
            try:
                cfg = data.get("config") or {}
                utts = data.get("utterances") or []
                if utts:
                    min_s = min(int(u.get("start_ms", 0)) for u in utts)
                    max_e = max(int(u.get("end_ms", 0)) for u in utts)
                    dur_s = round(max(0.0, (max_e - min_s) / 1000.0), 1)
                else:
                    dur_s = 0.0
                entry: dict[str, Any] = {
                    "id": str(data["id"]),
                    "created_at_iso": str(data.get("created_at_iso", "")),
                    "role_title": str(cfg.get("role_title", "Senior Software Engineer")),
                    "interview_type": str(cfg.get("interview_type", "behavioral")),
                    "mode": str(cfg.get("mode", "remote_meet")),
                    "qa_count": len(data.get("qa_groups") or []),
                    "duration_seconds": dur_s,
                    "has_calibration": bool(data.get("calibration_markdown")),
                }
                if recovered:
                    entry["recovered"] = True
                items.append(entry)
            except Exception:
                continue
        return items
