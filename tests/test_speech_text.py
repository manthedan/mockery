"""Code-flavored interview text is rewritten for the voice only; the screen keeps the original."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mockery.speech_text import speak_code, speak_identifier, speak_prose, speak_text
from mockery.tts import _clean_text_for_speech

BANKS_DIR = Path(__file__).resolve().parents[1] / "src" / "mockery" / "banks"


@pytest.mark.parametrize(
    ("code", "spoken"),
    [
        ("deps: list[tuple[str, str]]", "deps, a list of pairs of strings"),
        ("schedules: list[list[tuple[int, int]]]", "schedules, a list of lists of pairs of integers"),
        ("tasks: list[str]", "tasks, a list of strings"),
        ("dict[str, list[int]]", "a dictionary from strings to lists of integers"),
        ("x: int | None", "x, an optional integer"),
        ("tuple[str, int]", "a tuple of string and integer"),
        ("Map<string, number>", "a dictionary from strings to numbers"),
        ("string[]", "a list of strings"),
        ("get(key, now_ms)", "get of key and now M S"),
        ("put(key, value, ttl_ms, now_ms)", "put of key, value, T T L M S, and now M S"),
        ("allow(user_id, now_ms) -> bool", "allow of user I D and now M S, returning a boolean"),
        ("feed(chunk: str) -> list[dict]", "feed of chunk, a string, returning a list of dictionaries"),
        ("min(limit, len(items))", "min of limit and len of items"),
        ("undo()", "undo"),
        ("get_text() -> str", "get text, returning a string"),
        (
            "(now_ms - window_ms, now_ms]",
            "from now M S minus window M S, exclusive, to now M S, inclusive",
        ),
        ("[0, n)", "from 0, inclusive, to n, exclusive"),
        ("a * (b + c)", "a times the quantity b plus c"),
        ("a * b + c", "a times b plus c"),
        ("(a + b) * c", "the quantity a plus b, times c"),
        ("a[i] >= a[i-1]", "a at i is greater than or equal to a at i minus 1"),
        ("event:", "event"),
        (":", "a colon"),
        ("Redis", "Redis"),
    ],
)
def test_inline_code_is_read_the_way_a_person_says_it(code: str, spoken: str) -> None:
    assert speak_code(code) == spoken


def test_identifiers_split_into_words_and_spell_acronyms() -> None:
    assert speak_identifier("schedule_layers") == "schedule layers"
    assert speak_identifier("runWithLimit") == "run With Limit"
    assert speak_identifier("TTLLRUCache") == "T T L L R U Cache"
    assert speak_identifier("SSEParser") == "S S E Parser"
    assert speak_identifier("TOMBSTONE") == "tombstone"
    assert speak_identifier("JSON") == "json"


def test_prose_and_code_spans_combine_with_natural_pauses() -> None:
    assert (
        speak_text("Given `tasks: list[str]` and pairs `deps: list[tuple[str, str]]` where `(a, b)` means order.")
        == "Given tasks, a list of strings, and pairs deps, a list of pairs of strings, where a, b means order."
    )
    assert speak_text("Runs in O(n log n), O(n^2) at worst.") == "Runs in O of n log n, O of n squared at worst."
    assert speak_text("gaps >= duration; call get_text() per user_id") == (
        "gaps greater than or equal to duration; call get text per user I D"
    )
    plain = "Tell me about a time you disagreed with a colleague (or a manager). What happened?"
    assert speak_text(plain) == plain


def test_tts_cleaner_speaks_code_and_still_strips_markdown() -> None:
    cleaned = _clean_text_for_speech("**Implement** `schedule_layers(tasks, deps)` for `deps: list[tuple[str, str]]`.")
    assert cleaned == "Implement schedule layers of tasks and deps for deps, a list of pairs of strings."


@pytest.mark.parametrize("bank", ["coding", "system_design", "behavioral_leadership", "general"])
def test_no_bundled_prompt_reaches_the_voice_as_raw_code(bank: str) -> None:
    questions = json.loads((BANKS_DIR / f"{bank}.json").read_text(encoding="utf-8"))["questions"]
    for question in questions:
        spoken = _clean_text_for_speech(question["prompt"])
        assert spoken, question["id"]
        assert not re.search(r"[`_\[\]<>=]|->|\w\(", spoken), (question["id"], spoken)


def test_file_names_paths_and_addresses_in_prose_are_left_as_written() -> None:
    spoken = speak_prose("Open Five_Year_Plan.pdf, see docs/api_guide.md, write to first_last@example.com.")
    assert "Five_Year_Plan.pdf" in spoken
    assert "docs/api_guide.md" in spoken
    assert "first_last@example.com" in spoken
    assert speak_prose("Pass now_ms. Then user_id is set.") == "Pass now M S. Then user I D is set."


@pytest.mark.parametrize(
    "hostile",
    [
        "`x: int" + "[]" * 2900 + "`",
        "`" + "list[" * 900 + "int" + "]" * 900 + "`",
        "`" + "f(" * 1400 + "x" + ")" * 1400 + "`",
        "`" + "(" * 2900 + ")" * 2900 + "`",
        "`a: " + "A|" * 2900 + "B`",
    ],
)
def test_deeply_nested_code_within_the_question_limit_is_still_spoken(hostile: str) -> None:
    assert len(hostile) <= 6000
    assert speak_text(hostile)
    assert _clean_text_for_speech(hostile)
