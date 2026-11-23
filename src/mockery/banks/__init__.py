from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..levels import normalize_level
from ..models import short_id

logger = logging.getLogger("mockery.banks")

BUNDLED_BANKS_DIR = Path(__file__).resolve().parent
_JD_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]{1,24}")
_Q_WORD_RE = re.compile(r"[a-z0-9]{4,20}")

# The candidate's own saved questions: one hand-editable pack per interview format.
MY_QUESTION_CATEGORIES = ("behavioral", "general", "system_design", "coding")
MY_PACK_TITLE = "My questions"
MAX_MY_QUESTIONS = 300
MAX_MY_QUESTION_CHARS = 6_000
MAX_QUESTION_TITLE_CHARS = 64
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
_LIST_MARKER_RE = re.compile(r"^(?:[-*•–]|\(?\d{1,3}[.)])[ \t]+")
_WS_RUN_RE = re.compile(r"\s+")


class MyQuestionsError(ValueError):
    """A saved-question change that cannot be applied; the message is written for the candidate."""


def my_pack_id(category: str) -> str:
    return f"my_{category}"


def title_from_prompt(prompt: str) -> str:
    """A one-line title for a question the candidate wrote: its first line, shortened if long."""
    lines = prompt.strip().splitlines()
    first_line = lines[0].strip() if lines else ""
    if len(first_line) <= MAX_QUESTION_TITLE_CHARS:
        return first_line
    return first_line[: MAX_QUESTION_TITLE_CHARS - 1].rstrip() + "…"


def split_pasted_questions(text: str) -> list[str]:
    """
    Split a pasted list into questions. Blank lines separate questions when the paste has any, so a
    question may run over several lines; otherwise every line is one question. Leading bullets and
    list numbers are dropped.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    blocks = _BLANK_LINE_RE.split(cleaned) if _BLANK_LINE_RE.search(cleaned) else cleaned.split("\n")
    questions: list[str] = []
    for block in blocks:
        question = _LIST_MARKER_RE.sub("", block.strip(), count=1).strip()
        if question:
            questions.append(question)
    return questions


def _question_key(prompt: str) -> str:
    return _WS_RUN_RE.sub(" ", prompt.strip()).lower()


@dataclass(frozen=True)
class _IndexedQuestion:
    question_id: str
    raw: dict[str, Any]
    levels: tuple[str, ...]
    levels_lower: frozenset[str]
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
    Loads bundled JSON question bank packs (`behavioral`, `general`, `system_design`, `coding`) plus
    any user-contributed `*.json` packs dropped into `<data_root>/banks/`.
    Caches parsed packs and pre-tokenized JD indices keyed by `(mtime_ns, size)`.
    """

    def __init__(self, user_banks_dir: Path | None = None) -> None:
        self.bundled_dir = BUNDLED_BANKS_DIR
        self.user_dir = user_banks_dir
        self._lock = threading.Lock()
        self._my_questions_lock = threading.Lock()
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
                        levels = tuple(dict.fromkeys(normalize_level(lvl) for lvl in (q.get("levels") or [])))
                        comp_lower = str(q.get("competency") or "").lower()
                        jd_tags_lower = tuple(str(tag).lower() for tag in (q.get("jd_tags") or []))
                        text_blob = f"{q.get('title', '')} {q.get('competency', '')} {q.get('prompt', '')}".lower()
                        text_tokens = tuple(_Q_WORD_RE.findall(text_blob))
                        iq = _IndexedQuestion(
                            question_id=qid,
                            raw=q,
                            levels=levels,
                            levels_lower=frozenset(lvl.lower() for lvl in levels),
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
        lvl_norm = normalize_level(level).lower()
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
                if lvl_norm and iq.levels_lower and lvl_norm not in iq.levels_lower:
                    continue
                if comp_norm and comp_norm not in iq.competency_lower:
                    continue
                item = dict(iq.raw)
                item["levels"] = list(iq.levels)
                item["pack_id"] = pack_id
                item["category"] = pack_cat
                item["mine"] = _is_my_pack(cp)
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
                item["levels"] = list(iq.levels)
                item["pack_id"] = cp.meta["pack_id"]
                item["category"] = cp.meta["category"]
                item["mine"] = _is_my_pack(cp)
                return item
        return None

    def _my_pack_path(self, category: str) -> Path:
        if category not in MY_QUESTION_CATEGORIES:
            raise MyQuestionsError(f"Unknown interview format: {category}")
        if self.user_dir is None:
            raise MyQuestionsError("This install has no folder for saved questions.")
        return self.user_dir / f"{my_pack_id(category)}.json"

    def _read_my_pack(self, path: Path, category: str) -> dict[str, Any]:
        if not path.exists():
            return {
                "pack_id": my_pack_id(category),
                "title": MY_PACK_TITLE,
                "category": category,
                "version": "1.0",
                "description": "Questions you saved in Mockery. Edit this file freely; each question needs an id and a prompt.",
                "questions": [],
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MyQuestionsError(f"{path.name} could not be read, so it was left untouched: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
            raise MyQuestionsError(f"{path.name} is not a question pack, so it was left untouched.")
        return data

    @staticmethod
    def _write_my_pack(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.chmod(tmp_path, 0o600)
        tmp_path.replace(path)

    def add_my_questions(self, category: str, prompts: list[str]) -> tuple[list[dict[str, Any]], int]:
        """
        Save the candidate's own questions for one interview format.
        Returns the questions added and how many were skipped because they were already saved.
        """
        path = self._my_pack_path(category)
        with self._my_questions_lock:
            data = self._read_my_pack(path, category)
            questions: list[Any] = data["questions"]
            known = {_question_key(str(q.get("prompt") or "")) for q in questions if isinstance(q, dict)}
            added: list[dict[str, Any]] = []
            already_saved = 0
            for prompt in prompts:
                text = prompt.strip()
                if not text:
                    continue
                if len(text) > MAX_MY_QUESTION_CHARS:
                    raise MyQuestionsError(
                        f"A question is longer than {MAX_MY_QUESTION_CHARS:,} characters: {title_from_prompt(text)}"
                    )
                key = _question_key(text)
                if key in known:
                    already_saved += 1
                    continue
                known.add(key)
                added.append(
                    {
                        "id": short_id("my"),
                        "title": title_from_prompt(text),
                        "competency": "Your question",
                        "prompt": text,
                    }
                )
            if len(questions) + len(added) > MAX_MY_QUESTIONS:
                raise MyQuestionsError(
                    f"My questions holds up to {MAX_MY_QUESTIONS} per format. Remove some before adding more."
                )
            if added:
                questions.extend(added)
                self._write_my_pack(path, data)
        return added, already_saved

    def delete_my_question(self, question_id: str) -> bool:
        if self.user_dir is None:
            return False
        with self._my_questions_lock:
            for category in MY_QUESTION_CATEGORIES:
                path = self._my_pack_path(category)
                if not path.exists():
                    continue
                data = self._read_my_pack(path, category)
                kept = [q for q in data["questions"] if not (isinstance(q, dict) and str(q.get("id")) == question_id)]
                if len(kept) != len(data["questions"]):
                    data["questions"] = kept
                    self._write_my_pack(path, data)
                    return True
        return False


def _is_my_pack(pack: _CachedPack) -> bool:
    return not pack.meta["bundled"] and pack.meta["pack_id"] == my_pack_id(pack.category_lower)
