"""Models that think before they answer get the output room their own replies show they need."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mockery.llm import (
    REASONING_ALLOWANCE_CAP,
    InterviewCopilotLLM,
    LLMRuntimeConfig,
    _reasoning_tokens_spent,
)

MODEL = "local-thinker"


def _response(content: str, finish: str, reasoning_tokens: int | None = None, thinking: str = "") -> Any:
    usage = None
    if reasoning_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=reasoning_tokens + len(content) // 4,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        )
    message = SimpleNamespace(content=content, reasoning_content=thinking or None, model_extra={})
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason=finish, message=message)])


class _ThinkingModel:
    """Thinks for `thinking_tokens`, then answers; a request with too little room returns only thinking."""

    def __init__(self, thinking_tokens: int, answer: str = '{"ok": true}', report_usage: bool = True) -> None:
        self.thinking_tokens = thinking_tokens
        self.answer = answer
        self.report_usage = report_usage
        self.requested: list[int] = []
        self.timeouts: list[float] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        budget = int(kwargs["max_tokens"])
        self.requested.append(budget)
        self.timeouts.append(float(kwargs["timeout"]))
        answer_tokens = len(self.answer) // 4 + 1
        if budget >= self.thinking_tokens + answer_tokens:
            reported = self.thinking_tokens if self.report_usage else None
            return _response(self.answer, "stop", reported, "" if self.report_usage else "t" * self.thinking_tokens * 3)
        thought = min(budget, self.thinking_tokens)
        return _response("", "length", thought if self.report_usage else None, "" if self.report_usage else "t" * thought * 3)


def _llm(tmp_path: Path, client: Any) -> InterviewCopilotLLM:
    cfg = LLMRuntimeConfig(tmp_path)
    cfg.provider = "omlx"
    cfg.base_url = "http://localhost:8100/v1"
    llm = InterviewCopilotLLM(cfg)
    llm._client = lambda: client  # type: ignore[method-assign]
    return llm


def _ask(llm: InterviewCopilotLLM, max_tokens: int) -> str:
    return asyncio.run(
        llm._complete_text(model=MODEL, system="Reply in JSON.", user="Go.", max_tokens=max_tokens, timeout=30.0)
    )


def test_a_reply_lost_to_thinking_is_asked_for_again_with_room(tmp_path: Path) -> None:
    model = _ThinkingModel(thinking_tokens=850)
    llm = _llm(tmp_path, model)

    assert _ask(llm, 480) == '{"ok": true}'
    assert model.requested[0] == 480 and len(model.requested) == 2
    assert model.requested[1] >= 850 + 480 // 2
    assert model.timeouts[1] > model.timeouts[0]

    # The room is remembered for this model, so the next call succeeds on its first request.
    model.requested.clear()
    assert _ask(llm, 160) == '{"ok": true}'
    assert len(model.requested) == 1 and model.requested[0] > 850


def test_thinking_returned_as_text_is_counted_when_no_token_count_is_reported(tmp_path: Path) -> None:
    model = _ThinkingModel(thinking_tokens=600, report_usage=False)
    llm = _llm(tmp_path, model)
    assert _ask(llm, 300) == '{"ok": true}'
    assert len(model.requested) >= 2
    assert llm._reasoning_allowance[MODEL] > 0


def test_a_model_that_does_not_think_is_never_given_extra_or_retried(tmp_path: Path) -> None:
    calls: list[int] = []

    async def _create(**kwargs: Any) -> Any:
        calls.append(int(kwargs["max_tokens"]))
        return _response('{"cut": "off', "length", reasoning_tokens=0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    llm = _llm(tmp_path, client)
    assert _ask(llm, 200) == '{"cut": "off'
    assert calls == [200]
    assert MODEL not in llm._reasoning_allowance


def test_extra_room_is_bounded_and_retries_stop(tmp_path: Path) -> None:
    model = _ThinkingModel(thinking_tokens=10**6)
    llm = _llm(tmp_path, model)
    assert _ask(llm, 480) == ""
    assert len(model.requested) == 3
    assert max(model.requested) <= 480 + REASONING_ALLOWANCE_CAP
    assert llm._reasoning_allowance[MODEL] <= REASONING_ALLOWANCE_CAP


def test_unfinished_inline_thinking_counts_as_thinking() -> None:
    assert _reasoning_tokens_spent(None, "", "<think>Let me work through this " + "step " * 60) > 0
    assert _reasoning_tokens_spent(None, "", "<think>done</think>{\"ok\": true}") == 0
    assert _reasoning_tokens_spent(None, "", '{"ok": true}') == 0
