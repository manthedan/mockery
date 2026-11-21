from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mockery.banks")

BUNDLED_BANKS_DIR = Path(__file__).resolve().parent
_JD_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]{1,24}")
_Q_WORD_RE = re.compile(r"[a-z0-9]{4,20}")


@dataclass(frozen=True)
class _IndexedQuestion:
    question_id: str
    raw: dict[str, Any]
    levels_upper: frozenset[str]
    competency_lower: str
    jd_tags_lower: tuple[str, ...]
    text_tokens: tuple[str, ...]


@dataclass
class _CachedPack:
    mtime_ns: int
    size: int
    meta: dict[str, Any]
    category_lower: str
    indexed_questions: list[_IndexedQuestion]
    by_id: dict[str, _IndexedQuestion]


class QuestionBankStore:
    """
    Loads bundled JSON question bank packs (`behavioral`, `system_design`, `coding`) plus
    any user-contributed `*.json` packs dropped into `<data_root>/banks/`.
    Caches parsed packs and pre-tokenized JD indices keyed by `(mtime_ns, size)`.
    """

    def __init__(self, user_banks_dir: Path | None = None) -> None:
        self.bundled_dir = BUNDLED_BANKS_DIR
        self.user_dir = user_banks_dir
        self._lock = threading.Lock()
        self._pack_cache: dict[tuple[str, bool], _CachedPack] = {}

    def _iter_pack_files(self) -> list[tuple[Path, bool]]:
        files: list[tuple[Path, bool]] = []
        if self.bundled_dir.is_dir():
            for p in sorted(self.bundled_dir.glob("*.json")):
                files.append((p, True))
        if self.user_dir and self.user_dir.is_dir():
            for p in sorted(self.user_dir.glob("*.json")):
                files.append((p, False))
        return files

    def _load_cached_packs(self) -> list[_CachedPack]:
        file_list = self._iter_pack_files()
        live_keys: set[tuple[str, bool]] = set()
        loaded: list[_CachedPack] = []

        with self._lock:
            for path, is_bundled in file_list:
                key = (str(path), is_bundled)
                try:
                    st = os.stat(path)
                except OSError:
                    self._pack_cache.pop(key, None)
                    continue
                live_keys.add(key)
                cached = self._pack_cache.get(key)
                if cached is not None and cached.mtime_ns == st.st_mtime_ns and cached.size == st.st_size:
                    loaded.append(cached)
                    continue

                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
                        self._pack_cache.pop(key, None)
                        continue
                    pack_id = str(data.get("pack_id") or path.stem)
                    category = str(data.get("category") or "general")
                    raw_questions = data["questions"]
                    meta = {
                        "pack_id": pack_id,
                        "title": str(data.get("title") or path.stem),
                        "category": category,
                        "version": str(data.get("version") or "1.0"),
                        "description": str(data.get("description") or ""),
                        "bundled": is_bundled,
                        "question_count": len(raw_questions),
                        "questions": raw_questions,
                    }
                    indexed: list[_IndexedQuestion] = []
                    by_id: dict[str, _IndexedQuestion] = {}
                    for q in raw_questions:
                        if not isinstance(q, dict) or not q.get("id"):
                            continue
                        qid = str(q["id"])
                        levels_upper = frozenset(str(l).upper() for l in (q.get("levels") or []))
                        comp_lower = str(q.get("competency") or "").lower()
                        jd_tags_lower = tuple(str(tag).lower() for tag in (q.get("jd_tags") or []))
                        text_blob = f"{q.get('title', '')} {q.get('competency', '')} {q.get('prompt', '')}".lower()
                        text_tokens = tuple(_Q_WORD_RE.findall(text_blob))
                        iq = _IndexedQuestion(
                            question_id=qid,
                            raw=q,
                            levels_upper=levels_upper,
                            competency_lower=comp_lower,
                            jd_tags_lower=jd_tags_lower,
                            text_tokens=text_tokens,
                        )
                        indexed.append(iq)
                        by_id.setdefault(qid, iq)

                    cp = _CachedPack(
                        mtime_ns=st.st_mtime_ns,
                        size=st.st_size,
                        meta=meta,
                        category_lower=category.lower(),
                        indexed_questions=indexed,
                        by_id=by_id,
                    )
                    self._pack_cache[key] = cp
                    loaded.append(cp)
                except Exception as exc:
                    self._pack_cache.pop(key, None)
                    logger.warning("Failed to parse question bank %s: %s", path, exc)

            if len(self._pack_cache) > len(live_keys):
                for stale in list(self._pack_cache.keys()):
                    if stale not in live_keys:
                        self._pack_cache.pop(stale, None)

        return loaded

    def list_packs(self) -> list[dict[str, Any]]:
        packs: list[dict[str, Any]] = []
        for cp in self._load_cached_packs():
            pack_copy = dict(cp.meta)
            pack_copy["questions"] = [dict(q) if isinstance(q, dict) else q for q in cp.meta["questions"]]
            packs.append(pack_copy)
        return packs

    def list_questions(
        self,
        *,
        category: str | None = None,
        level: str | None = None,
        competency: str | None = None,
        jd_text: str | None = None,
    ) -> list[dict[str, Any]]:
        cat_norm = (category or "").strip().lower()
        lvl_norm = (level or "").strip().upper()
        comp_norm = (competency or "").strip().lower()
        has_jd = bool(jd_text and jd_text.strip())
        jd_lower = jd_text.lower() if has_jd and jd_text is not None else ""
        jd_tokens = set(_JD_TOKEN_RE.findall(jd_lower)) if has_jd else set()

        results: list[dict[str, Any]] = []
        for cp in self._load_cached_packs():
            if cat_norm and cp.category_lower != cat_norm:
                continue
            pack_id = cp.meta["pack_id"]
            pack_cat = cp.meta["category"]
            for iq in cp.indexed_questions:
                if lvl_norm and iq.levels_upper and lvl_norm not in iq.levels_upper:
                    continue
                if comp_norm and comp_norm not in iq.competency_lower:
                    continue
                item = dict(iq.raw)
                item["pack_id"] = pack_id
                item["category"] = pack_cat
                if has_jd:
                    score = 0
                    for tag in iq.jd_tags_lower:
                        if tag in jd_lower or tag in jd_tokens:
                            score += 3
                    for tok in iq.text_tokens:
                        if tok in jd_tokens:
                            score += 1
                    item["jd_relevance_score"] = score
                results.append(item)

        if has_jd:
            results.sort(key=lambda item: int(item.get("jd_relevance_score", 0)), reverse=True)
        return results

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        for cp in self._load_cached_packs():
            iq = cp.by_id.get(question_id)
            if iq is not None:
                item = dict(iq.raw)
                item["pack_id"] = cp.meta["pack_id"]
                item["category"] = cp.meta["category"]
                return item
        return None
