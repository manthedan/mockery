from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

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
MY_QUESTIONS_LOCK_FILE = ".my_questions.lock"
MY_PACK_WRITE_ATTEMPTS = 3
_T = TypeVar("_T")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
_LIST_MARKER_RE = re.compile(r"^(?:[-*•–]|\(?\d{1,3}[.)])[ \t]+")
_WS_RUN_RE = re.compile(r"\s+")


class MyQuestionsError(ValueError):
    """A saved-question change that cannot be applied; the message is written for the candidate."""


def my_pack_id(category: str) -> str:
    return f"my_{category}"


_MY_PACK_FILES = {f"{my_pack_id(category)}.json": category for category in MY_QUESTION_CATEGORIES}


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
    question may run over several lines; otherwise every line is one question. Inside a block,
    numbered or bulleted lines each start a question. Leading bullets and list numbers are dropped.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    blocks = _BLANK_LINE_RE.split(cleaned) if _BLANK_LINE_RE.search(cleaned) else cleaned.split("\n")
    questions: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n")]
        if sum(1 for line in lines if _LIST_MARKER_RE.match(line)) < 2:
            items = ["\n".join(lines)]
        else:
            items = []
            for line in lines:
                if _LIST_MARKER_RE.match(line) or not items:
                    items.append(line)
                else:
                    items[-1] += f"\n{line}"
        for item in items:
            question = _LIST_MARKER_RE.sub("", item, count=1).strip()
            if re.search(r"[^\W_]", question):
                questions.append(question)
    return questions


def _question_key(prompt: str) -> str:
    """Two saved questions are the same when only their spacing differs; case can change the answer."""
    return _WS_RUN_RE.sub(" ", prompt.strip())


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
    mine: bool = False


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
                    # A saved-questions file is known by its name, whatever its contents say, so a
                    # hand edit can never turn the candidate's own questions into bank questions.
                    my_category = None if is_bundled else _MY_PACK_FILES.get(path.name)
                    pack_id = my_pack_id(my_category) if my_category else str(data.get("pack_id") or path.stem)
                    category = my_category or str(data.get("category") or "general")
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
                        mine=my_category is not None,
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
                item["mine"] = cp.mine
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

    def get_question(self, question_id: str, *, category: str | None = None) -> dict[str, Any] | None:
        """The question with this id; when two packs share an id, the one in ``category`` wins."""
        wanted = (category or "").strip().lower()
        packs = sorted(self._load_cached_packs(), key=lambda cp: cp.category_lower != wanted)
        for cp in packs:
            iq = cp.by_id.get(question_id)
            if iq is not None:
                item = dict(iq.raw)
                item["levels"] = list(iq.levels)
                item["pack_id"] = cp.meta["pack_id"]
                item["category"] = cp.meta["category"]
                item["mine"] = cp.mine
                return item
        return None

    def _my_pack_path(self, category: str) -> Path:
        if category not in MY_QUESTION_CATEGORIES:
            raise MyQuestionsError(f"Unknown interview format: {category}")
        if self.user_dir is None:
            raise MyQuestionsError("This install has no folder for saved questions.")
        return self.user_dir / f"{my_pack_id(category)}.json"

    @staticmethod
    def _read_my_pack(path: Path, category: str) -> tuple[dict[str, Any], tuple[int, int] | None]:
        """The pack and the file version it was read from (None when the file does not exist yet)."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {
                "pack_id": my_pack_id(category),
                "title": MY_PACK_TITLE,
                "category": category,
                "version": "1.0",
                "description": "Questions you saved in Mockery. Edit this file freely; each question needs an id and a prompt.",
                "questions": [],
            }, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MyQuestionsError(f"{path.name} could not be read, so it was left untouched: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
            raise MyQuestionsError(f"{path.name} is not a question pack, so it was left untouched.")
        return data, (stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _file_version(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _change_my_pack(self, category: str, change: Callable[[dict[str, Any]], _T]) -> _T:
        """
        Apply ``change`` to one saved-questions pack and write it back only if ``change`` altered it.
        Every Mockery process sharing this folder takes the same file lock, and a file that changed
        after it was read (an edit made in a text editor) is read again instead of being overwritten.
        """
        path = self._my_pack_path(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._my_questions_lock, open(path.parent / MY_QUESTIONS_LOCK_FILE, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            for _attempt in range(MY_PACK_WRITE_ATTEMPTS):
                data, version = self._read_my_pack(path, category)
                before = json.dumps(data, sort_keys=True)
                result = change(data)
                if json.dumps(data, sort_keys=True) == before:
                    return result
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False
                ) as tmp:
                    tmp.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
                tmp_path = Path(tmp.name)
                os.chmod(tmp_path, 0o600)
                if self._file_version(path) != version:
                    tmp_path.unlink()
                    continue
                tmp_path.replace(path)
                return result
        raise MyQuestionsError(f"{path.name} kept changing while it was being saved. Try again in a moment.")

    def add_my_questions(self, category: str, prompts: list[str]) -> tuple[list[dict[str, Any]], int]:
        """
        Save the candidate's own questions for one interview format.
        Returns the questions added and how many were skipped because they were already saved.
        """
        texts = [prompt.strip() for prompt in prompts if prompt.strip()]
        for text in texts:
            if len(text) > MAX_MY_QUESTION_CHARS:
                raise MyQuestionsError(
                    f"A question is longer than {MAX_MY_QUESTION_CHARS:,} characters: {title_from_prompt(text)}"
                )

        def add(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
            questions: list[Any] = data["questions"]
            known = {_question_key(str(q.get("prompt") or "")) for q in questions if isinstance(q, dict)}
            taken_ids = {str(q.get("id")) for q in questions if isinstance(q, dict)}
            added: list[dict[str, Any]] = []
            already_saved = 0
            for text in texts:
                key = _question_key(text)
                if key in known:
                    already_saved += 1
                    continue
                known.add(key)
                question_id = short_id("my")
                while question_id in taken_ids:
                    question_id = short_id("my")
                taken_ids.add(question_id)
                added.append(
                    {"id": question_id, "title": title_from_prompt(text), "competency": "Your question", "prompt": text}
                )
            if len(questions) + len(added) > MAX_MY_QUESTIONS:
                raise MyQuestionsError(
                    f"My questions holds up to {MAX_MY_QUESTIONS} per format. Remove some before adding more."
                )
            questions.extend(added)
            return added, already_saved

        return self._change_my_pack(category, add)

    def delete_my_question(self, category: str, question_id: str) -> bool:
        """Remove one saved question from one format's pack. False when it is not there."""
        if not self._my_pack_path(category).exists():
            return False

        def remove(data: dict[str, Any]) -> bool:
            for index, question in enumerate(data["questions"]):
                if isinstance(question, dict) and str(question.get("id")) == question_id:
                    del data["questions"][index]
                    return True
            return False

        return self._change_my_pack(category, remove)
