"""Every bundled coding problem ships a Python test suite; these reference solutions prove
each suite can be passed, so a candidate is never graded against an unsatisfiable test."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path

import pytest

BANK_PATH = Path(__file__).resolve().parents[1] / "src" / "mockery" / "banks" / "coding.json"
DIFFICULTIES = {"Easy", "Medium", "Hard"}

REFERENCE_SOLUTIONS: dict[str, str] = {
    "code_ttl_lru_01": '''
from collections import OrderedDict

class TTLLRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, key, now_ms):
        entry = self.items.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if now_ms >= expires_at:
            del self.items[key]
            return None
        self.items.move_to_end(key)
        return value

    def put(self, key, value, ttl_ms, now_ms):
        if self.capacity <= 0:
            return
        self.items.pop(key, None)
        if len(self.items) >= self.capacity:
            for stale in [k for k, (_, exp) in self.items.items() if now_ms >= exp]:
                del self.items[stale]
        while len(self.items) >= self.capacity:
            self.items.popitem(last=False)
        self.items[key] = (value, now_ms + ttl_ms)
''',
    "code_rate_01": '''
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self, limit: int, window_ms: int):
        self.limit = limit
        self.window_ms = window_ms
        self.hits = defaultdict(deque)

    def allow(self, user_id, now_ms):
        hits = self.hits[user_id]
        while hits and hits[0] <= now_ms - self.window_ms:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now_ms)
        return True
''',
    "code_sse_01": '''
class SSEParser:
    def __init__(self):
        self.buffer = ""
        self.event = ""
        self.data = []

    def feed(self, chunk):
        self.buffer += chunk
        out = []
        while True:
            cut = -1
            for i, ch in enumerate(self.buffer):
                if ch == "\\n":
                    cut, width = i, 1
                    break
                if ch == "\\r":
                    if i + 1 == len(self.buffer):
                        break
                    cut, width = i, 2 if self.buffer[i + 1] == "\\n" else 1
                    break
            if cut < 0:
                return out
            line, self.buffer = self.buffer[:cut], self.buffer[cut + width:]
            if line == "":
                if self.data:
                    out.append({"event": self.event or "message", "data": "\\n".join(self.data)})
                self.event, self.data = "", []
                continue
            if line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if name == "event":
                self.event = value
            elif name == "data":
                self.data.append(value)
''',
    "code_dag_01": '''
def schedule_layers(tasks, deps):
    edges = set(deps)
    indegree = {t: 0 for t in tasks}
    children = {t: [] for t in tasks}
    for before, after in edges:
        children[before].append(after)
        indegree[after] += 1
    layer = sorted(t for t in tasks if indegree[t] == 0)
    layers, placed = [], 0
    while layer:
        layers.append(layer)
        placed += len(layer)
        ready = []
        for t in layer:
            for child in children[t]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        layer = sorted(ready)
    if placed != len(tasks):
        raise ValueError("cycle detected")
    return layers
''',
    "code_undo_01": '''
class TextEditor:
    def __init__(self):
        self.text = ""
        self.undo_stack = []
        self.redo_stack = []

    def _apply(self, op):
        kind, pos, chunk = op
        if kind == "insert":
            self.text = self.text[:pos] + chunk + self.text[pos:]
        else:
            self.text = self.text[:pos] + self.text[pos + len(chunk):]

    @staticmethod
    def _inverse(op):
        kind, pos, chunk = op
        return ("delete" if kind == "insert" else "insert", pos, chunk)

    def insert(self, pos, text):
        pos = max(0, min(pos, len(self.text)))
        op = ("insert", pos, text)
        self._apply(op)
        self.undo_stack.append(op)
        self.redo_stack.clear()

    def delete(self, pos, length):
        pos = max(0, min(pos, len(self.text)))
        removed = self.text[pos:pos + max(0, length)]
        op = ("delete", pos, removed)
        self._apply(op)
        self.undo_stack.append(op)
        self.redo_stack.clear()
        return removed

    def undo(self):
        if not self.undo_stack:
            return False
        op = self.undo_stack.pop()
        self._apply(self._inverse(op))
        self.redo_stack.append(op)
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        op = self.redo_stack.pop()
        self._apply(op)
        self.undo_stack.append(op)
        return True

    def get_text(self):
        return self.text
''',
    "code_merge_int_01": '''
def find_free_slots(schedules, day_start, day_end, duration):
    busy = sorted(
        (max(start, day_start), min(end, day_end))
        for person in schedules
        for start, end in person
        if end > day_start and start < day_end
    )
    free, cursor = [], day_start
    for start, end in busy:
        if start - cursor >= duration:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if day_end - cursor >= duration:
        free.append((cursor, day_end))
    return free
''',
    "code_async_pool_01": '''
import asyncio

async def run_with_limit(items, worker, limit):
    items = list(items)
    results = [None] * len(items)
    next_index = 0

    async def drain():
        nonlocal next_index
        while next_index < len(items):
            i = next_index
            next_index += 1
            results[i] = await worker(items[i])

    tasks = [asyncio.ensure_future(drain()) for _ in range(max(1, min(limit, len(items))))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
    return results
''',
    "code_kv_txn_01": '''
class TransactionalKV:
    _DELETED = object()

    def __init__(self):
        self.frames = [{}]

    def set(self, key, value):
        self.frames[-1][key] = value

    def get(self, key):
        for frame in reversed(self.frames):
            if key in frame:
                value = frame[key]
                return None if value is self._DELETED else value
        return None

    def delete(self, key):
        if len(self.frames) == 1:
            self.frames[0].pop(key, None)
        else:
            self.frames[-1][key] = self._DELETED

    def begin(self):
        self.frames.append({})

    def commit(self):
        if len(self.frames) == 1:
            return False
        top = self.frames.pop()
        parent = self.frames[-1]
        for key, value in top.items():
            if value is self._DELETED and len(self.frames) == 1:
                parent.pop(key, None)
            else:
                parent[key] = value
        return True

    def rollback(self):
        if len(self.frames) == 1:
            return False
        self.frames.pop()
        return True
''',
}


def _load_questions() -> list[dict]:
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))["questions"]


def _run_like_pyodide(source: str, namespace: dict) -> None:
    """Run candidate code or a suite the way the Coding Studio does: top-level `await` allowed."""
    code = compile(source, "<bank>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if inspect.iscoroutine(result):
        asyncio.run(result)


def test_every_coding_question_has_a_reference_solution_and_difficulty() -> None:
    questions = _load_questions()
    assert {q["id"] for q in questions} == set(REFERENCE_SOLUTIONS)
    for q in questions:
        assert q.get("difficulty") in DIFFICULTIES, q["id"]


@pytest.mark.parametrize("question", _load_questions(), ids=lambda q: q["id"])
def test_python_suite_passes_reference_and_fails_starter(question: dict, capsys: pytest.CaptureFixture[str]) -> None:
    suite = question["tests"]["python"]

    solved: dict = {}
    _run_like_pyodide(REFERENCE_SOLUTIONS[question["id"]], solved)
    _run_like_pyodide(suite, solved)
    assert "ALL TESTS PASSED" in capsys.readouterr().out

    untouched: dict = {}
    _run_like_pyodide(question["starter_code"]["python"], untouched)
    with pytest.raises(Exception):
        _run_like_pyodide(suite, untouched)
