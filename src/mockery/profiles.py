from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import short_id
from .redactor import RampartRedactor, RedactionResult


def _mask_pii_preview(value: str, label: str) -> str:
    """Return a recognizable but privacy-safe masked preview for an actively redacted PII span."""
    raw = re.sub(r"\s+", " ", (value or "").strip())
    if not raw:
        return "••••"
    if label == "EMAIL" and "@" in raw:
        local, domain = raw.split("@", 1)
        local_m = f"{local[0]}•••" if local else "•••"
        dom_parts = domain.split(".")
        dom_m = f"{dom_parts[0][:1]}•••.{dom_parts[-1]}" if len(dom_parts) >= 2 else "•••"
        return f"{local_m}@{dom_m}"
    if label == "PHONE":
        digits = re.sub(r"\D", "", raw)
        return f"•••-•••-{digits[-4:]}" if len(digits) >= 4 else "•••-••••"
    if label == "URL":
        m = re.match(r"^(https?://)?([^/]+)", raw, flags=re.IGNORECASE)
        host = m.group(2) if m else raw[:14]
        return f"{host}/•••"
    if label in ("SSN", "CREDIT_CARD"):
        digits = re.sub(r"\D", "", raw)
        return f"••••{digits[-4:]}" if len(digits) >= 4 else "••••"
    words = raw.split(" ")
    masked_words = [f"{w[0]}{'•' * min(4, max(1, len(w) - 1))}" for w in words[:4] if w]
    suffix = " …" if len(words) > 4 else ""
    return " ".join(masked_words) + suffix


class ResumeRedactionPrefs(BaseModel):
    keep_location: bool = True
    hide_employers_and_schools: bool = False
    custom_terms: list[str] = Field(default_factory=list)
    unredacted_values: list[str] = Field(default_factory=list)
    use_rampart: bool = True


class ResumeProfile(BaseModel):
    filename: str = ""
    raw_text: str = ""
    redacted_text: str = ""
    spans: list[dict[str, Any]] = Field(default_factory=list)
    engine: str = "rules_only"
    entity_counts: dict[str, int] = Field(default_factory=dict)
    redacted_count: int = 0
    total_detected: int = 0
    prefs: ResumeRedactionPrefs = Field(default_factory=ResumeRedactionPrefs)
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class JobListingProfile(BaseModel):
    id: str = Field(default_factory=lambda: short_id("jd"))
    title: str = "Target Role"
    company: str = ""
    url: str = ""
    text: str = ""
    target_level: str = "L5"
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class CandidateProfileState(BaseModel):
    resume: ResumeProfile | None = None
    job_listings: list[JobListingProfile] = Field(default_factory=list)
    active_jd_id: str | None = None


class CandidateProfileStore:
    """
    Stores candidate resume (raw kept strictly on disk with exclusive 0600 perms; only redacted text
    is sent to cloud LLMs and raw hidden PII is never returned by public APIs) and target job listings
    under `<data_root>/profiles/profile.json`.
    """

    def __init__(self, profiles_dir: Path, redactor: RampartRedactor) -> None:
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.profiles_dir, 0o700)
        except OSError:
            pass
        self.profile_path = self.profiles_dir / "profile.json"
        self.redactor = redactor
        self._lock = threading.Lock()
        self._resume_revision = 0
        self._state = self._load()

    def _load(self) -> CandidateProfileState:
        if not self.profile_path.is_file():
            return CandidateProfileState()
        try:
            raw = json.loads(self.profile_path.read_text(encoding="utf-8"))
            return CandidateProfileState.model_validate(raw)
        except Exception:
            return CandidateProfileState()

    def _save_locked(self) -> None:
        """Atomically persist state using an exclusively-created 0600 temp file (caller holds self._lock)."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        payload = self._state.model_dump_json(indent=2).encode("utf-8")
        tmp_path = self.profiles_dir / f".profile.{secrets.token_hex(8)}.tmp"
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(self.profile_path))
        except Exception:
            try:
                os.unlink(str(tmp_path))
            except OSError:
                pass
            raise

    @property
    def state(self) -> CandidateProfileState:
        return self._state

    def active_jd(self) -> JobListingProfile | None:
        with self._lock:
            if self._state.active_jd_id:
                for jd in self._state.job_listings:
                    if jd.id == self._state.active_jd_id:
                        return jd
            return self._state.job_listings[0] if self._state.job_listings else None

    def resume_for_llm(self, *, is_local_llm: bool = False) -> str:
        """
        Return the privacy-safe resume text to pass into LLM prompts.
        Always uses `redacted_text` so raw PII never leaves the Mac.
        """
        with self._lock:
            r = self._state.resume
            if not r or not r.raw_text.strip():
                return ""
            return (r.redacted_text or "")[:12_000]

    def set_resume(
        self,
        *,
        filename: str,
        raw_text: str,
        prefs: ResumeRedactionPrefs | None = None,
    ) -> ResumeProfile | None:
        with self._lock:
            self._resume_revision += 1
            rev = self._resume_revision
            effective_prefs = prefs or (
                self._state.resume.prefs if self._state.resume is not None else ResumeRedactionPrefs()
            )

        res: RedactionResult = self.redactor.redact(
            raw_text,
            keep_location=effective_prefs.keep_location,
            hide_employers_and_schools=effective_prefs.hide_employers_and_schools,
            custom_terms=effective_prefs.custom_terms,
            unredacted_values=effective_prefs.unredacted_values,
            use_rampart=effective_prefs.use_rampart,
        )
        profile = ResumeProfile(
            filename=filename or "resume.txt",
            raw_text=raw_text,
            redacted_text=res.redacted_text,
            spans=res.spans,
            engine=res.engine,
            entity_counts=res.entity_counts,
            redacted_count=res.redacted_count,
            total_detected=res.total_detected,
            prefs=effective_prefs,
            updated_at_ms=int(time.time() * 1000),
        )
        with self._lock:
            if self._resume_revision != rev:
                # A newer upload or delete superseded this redaction pass; do not overwrite/resurrect.
                return self._state.resume
            self._state.resume = profile
            self._save_locked()
            return profile

    def update_redaction_prefs(self, prefs: ResumeRedactionPrefs) -> ResumeProfile | None:
        with self._lock:
            if self._state.resume is None:
                return None
            rev = self._resume_revision
            filename = self._state.resume.filename
            raw_text = self._state.resume.raw_text

        res: RedactionResult = self.redactor.redact(
            raw_text,
            keep_location=prefs.keep_location,
            hide_employers_and_schools=prefs.hide_employers_and_schools,
            custom_terms=prefs.custom_terms,
            unredacted_values=prefs.unredacted_values,
            use_rampart=prefs.use_rampart,
        )
        profile = ResumeProfile(
            filename=filename or "resume.txt",
            raw_text=raw_text,
            redacted_text=res.redacted_text,
            spans=res.spans,
            engine=res.engine,
            entity_counts=res.entity_counts,
            redacted_count=res.redacted_count,
            total_detected=res.total_detected,
            prefs=prefs,
            updated_at_ms=int(time.time() * 1000),
        )
        with self._lock:
            if self._resume_revision != rev or self._state.resume is None:
                return self._state.resume
            self._state.resume = profile
            self._save_locked()
            return profile

    def toggle_span_redaction(self, span_id: str) -> ResumeProfile | None:
        """Toggle redaction on/off for a detected span server-side without exposing hidden PII to HTTP callers."""
        with self._lock:
            r = self._state.resume
            if r is None:
                return None
            target_span = next((sp for sp in r.spans if sp.get("id") == span_id), None)
            if target_span is None:
                return None
            orig = str(target_span.get("original") or "")
            target_norm = re.sub(r"\s+", " ", orig.strip()).lower()
            cur_unredacted = list(r.prefs.unredacted_values)
            idx = next(
                (i for i, v in enumerate(cur_unredacted) if re.sub(r"\s+", " ", v.strip()).lower() == target_norm),
                -1,
            )
            if idx >= 0:
                cur_unredacted.pop(idx)
            else:
                cur_unredacted.append(orig)
            next_prefs = r.prefs.model_copy(update={"unredacted_values": cur_unredacted})
        return self.update_redaction_prefs(next_prefs)

    def clear_resume(self) -> None:
        with self._lock:
            self._resume_revision += 1
            self._state.resume = None
            self._save_locked()

    def add_job_listing(
        self,
        *,
        title: str,
        company: str = "",
        url: str = "",
        text: str,
        target_level: str = "L5",
    ) -> JobListingProfile:
        clean_url = url.strip()
        with self._lock:
            # Update existing JD if same URL was re-imported
            if clean_url:
                for idx, existing in enumerate(self._state.job_listings):
                    if existing.url == clean_url:
                        updated = JobListingProfile(
                            id=existing.id,
                            title=title.strip() or existing.title,
                            company=company.strip(),
                            url=clean_url,
                            text=text.strip()[:20_000],
                            target_level=target_level or existing.target_level,
                            created_at_ms=int(time.time() * 1000),
                        )
                        self._state.job_listings[idx] = updated
                        self._state.active_jd_id = updated.id
                        self._save_locked()
                        return updated

            jd = JobListingProfile(
                title=title.strip() or "Target Role",
                company=company.strip(),
                url=clean_url,
                text=text.strip()[:20_000],
                target_level=target_level or "L5",
            )
            self._state.job_listings.insert(0, jd)
            self._state.job_listings = self._state.job_listings[:20]
            self._state.active_jd_id = jd.id
            self._save_locked()
            return jd

    def select_job_listing(self, jd_id: str | None) -> JobListingProfile | None:
        with self._lock:
            if not jd_id:
                self._state.active_jd_id = None
                self._save_locked()
                return None
            for jd in self._state.job_listings:
                if jd.id == jd_id:
                    self._state.active_jd_id = jd.id
                    self._save_locked()
                    return jd
            return None

    def delete_job_listing(self, jd_id: str) -> bool:
        with self._lock:
            before = len(self._state.job_listings)
            self._state.job_listings = [j for j in self._state.job_listings if j.id != jd_id]
            if len(self._state.job_listings) == before:
                return False
            if self._state.active_jd_id == jd_id:
                self._state.active_jd_id = self._state.job_listings[0].id if self._state.job_listings else None
            self._save_locked()
            return True

    def public_dict(self) -> dict[str, Any]:
        with self._lock:
            resume_dict: dict[str, Any] | None = None
            if self._state.resume is not None:
                r = self._state.resume
                safe_spans = []
                for sp in r.spans:
                    is_active = bool(sp.get("active", True))
                    orig = str(sp.get("original") or "")
                    label = str(sp.get("label") or "PII")
                    safe_spans.append(
                        {
                            "id": sp.get("id", ""),
                            "placeholder": sp.get("placeholder", ""),
                            "label": label,
                            "source": sp.get("source", ""),
                            "active": is_active,
                            # Never return raw hidden PII over HTTP; mask actively redacted spans
                            "display_value": _mask_pii_preview(orig, label) if is_active else orig,
                        }
                    )
                resume_dict = {
                    "filename": r.filename,
                    "char_count": len(r.raw_text),
                    "redacted_text": r.redacted_text,
                    "spans": safe_spans,
                    "engine": r.engine,
                    "entity_counts": dict(r.entity_counts),
                    "redacted_count": r.redacted_count,
                    "total_detected": r.total_detected,
                    "prefs": r.prefs.model_dump(),
                    "updated_at_ms": r.updated_at_ms,
                }
            return {
                "resume": resume_dict,
                "job_listings": [j.model_dump() for j in self._state.job_listings],
                "active_jd_id": self._state.active_jd_id,
                "rampart": self.redactor.status_dict(),
            }
