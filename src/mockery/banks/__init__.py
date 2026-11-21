from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("mockery.banks")

BUNDLED_BANKS_DIR = Path(__file__).resolve().parent


class QuestionBankStore:
    """
    Loads bundled JSON question bank packs (`behavioral`, `system_design`, `coding`) plus
    any user-contributed `*.json` packs dropped into `<data_root>/banks/`.
    """

    def __init__(self, user_banks_dir: Path | None = None) -> None:
        self.bundled_dir = BUNDLED_BANKS_DIR
        self.user_dir = user_banks_dir

    def _iter_pack_files(self) -> list[tuple[Path, bool]]:
        files: list[tuple[Path, bool]] = []
        if self.bundled_dir.is_dir():
            for p in sorted(self.bundled_dir.glob("*.json")):
                files.append((p, True))
        if self.user_dir and self.user_dir.is_dir():
            for p in sorted(self.user_dir.glob("*.json")):
                files.append((p, False))
        return files

    def list_packs(self) -> list[dict[str, Any]]:
        packs: list[dict[str, Any]] = []
        for path, is_bundled in self._iter_pack_files():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
                    continue
                packs.append(
                    {
                        "pack_id": str(data.get("pack_id") or path.stem),
                        "title": str(data.get("title") or path.stem),
                        "category": str(data.get("category") or "general"),
                        "version": str(data.get("version") or "1.0"),
                        "description": str(data.get("description") or ""),
                        "bundled": is_bundled,
                        "question_count": len(data["questions"]),
                        "questions": data["questions"],
                    }
                )
            except Exception as exc:
                logger.warning("Failed to parse question bank %s: %s", path, exc)
        return packs

    def list_questions(
        self,
        *,
        category: str | None = None,
        level: str | None = None,
        competency: str | None = None,
        jd_text: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cat_norm = (category or "").strip().lower()
        lvl_norm = (level or "").strip().upper()
        comp_norm = (competency or "").strip().lower()

        for pack in self.list_packs():
            if cat_norm and pack["category"].lower() != cat_norm:
                continue
            for q in pack["questions"]:
                if not isinstance(q, dict) or not q.get("id"):
                    continue
                levels = [str(l).upper() for l in (q.get("levels") or [])]
                if lvl_norm and levels and lvl_norm not in levels:
                    continue
                q_comp = str(q.get("competency") or "")
                if comp_norm and comp_norm not in q_comp.lower():
                    continue
                item = dict(q)
                item["pack_id"] = pack["pack_id"]
                item["category"] = pack["category"]
                results.append(item)

        if jd_text and jd_text.strip():
            jd_tokens = set(re.findall(r"[a-z0-9][a-z0-9+#.\-]{1,24}", jd_text.lower()))

            def _jd_score(q: dict[str, Any]) -> int:
                score = 0
                for tag in q.get("jd_tags") or []:
                    t = str(tag).lower()
                    if t in jd_text.lower() or t in jd_tokens:
                        score += 3
                text_blob = f"{q.get('title', '')} {q.get('competency', '')} {q.get('prompt', '')}".lower()
                for tok in re.findall(r"[a-z0-9]{4,20}", text_blob):
                    if tok in jd_tokens:
                        score += 1
                return score

            for item in results:
                item["jd_relevance_score"] = _jd_score(item)
            results.sort(key=lambda item: int(item.get("jd_relevance_score", 0)), reverse=True)
        return results

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        for pack in self.list_packs():
            for q in pack["questions"]:
                if isinstance(q, dict) and q.get("id") == question_id:
                    item = dict(q)
                    item["pack_id"] = pack["pack_id"]
                    item["category"] = pack["category"]
                    return item
        return None
