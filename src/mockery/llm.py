from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from openai import AsyncOpenAI

from .models import (
    DEFAULT_PARAKEET_MODEL,
    LLM_PRESETS,
    DiveDeeperResponse,
    InterviewSession,
    ProbeSuggestion,
    QAGroup,
    TurnSegment,
    TurnTriagePlan,
    Utterance,
    fmt_ms,
)

logger = logging.getLogger(__name__)


PROVIDER_ENV_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

CLI_BACKEND_MAP: dict[str, str] = {
    "claude_cli": "claude",
    "codex_cli": "codex",
    "gemini_cli": "gemini",
    "agy": "agy",
    "pi": "pi",
}

LOCAL_HTTP_PROVIDERS = {"ollama", "lmstudio"}


class LLMRuntimeConfig:
    """Manages OpenAI-compatible, native Anthropic, local Ollama/LM Studio, & local CLI LLM + ASR config with `.env.local` persistence."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.env_local_path = project_root / ".env.local"
        self.env_path = project_root / ".env"

        merged: dict[str, str] = {}
        if self.env_path.exists():
            merged.update({k: v for k, v in dotenv_values(self.env_path).items() if v is not None})
        if self.env_local_path.exists():
            merged.update({k: v for k, v in dotenv_values(self.env_local_path).items() if v is not None})

        self._has_saved_provider: bool = bool(merged.get("LLM_PROVIDER") or os.getenv("LLM_PROVIDER"))
        self.provider: str = merged.get("LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "gemini")
        preset = LLM_PRESETS.get(self.provider, LLM_PRESETS["gemini"])
        self.base_url: str = merged.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL", preset["base_url"])

        provider_env_var = PROVIDER_ENV_KEYS.get(self.provider)
        provider_fallback = (
            (merged.get(provider_env_var) or os.getenv(provider_env_var, ""))
            if provider_env_var
            else ""
        )
        self.api_key: str = merged.get("LLM_API_KEY") or os.getenv("LLM_API_KEY") or provider_fallback or ""

        self.fast_model: str = merged.get("LLM_FAST_MODEL") or os.getenv("LLM_FAST_MODEL", preset["fast_model"])
        self.calibration_model: str = merged.get("LLM_CALIBRATION_MODEL") or os.getenv(
            "LLM_CALIBRATION_MODEL", preset["calibration_model"]
        )
        self.parakeet_model: str = (
            merged.get("PARAKEET_MODEL")
            or os.getenv("PARAKEET_MODEL")
            or DEFAULT_PARAKEET_MODEL
        )

    @property
    def cli_backend(self) -> str | None:
        if self.provider in CLI_BACKEND_MAP:
            return CLI_BACKEND_MAP[self.provider]
        if self.base_url.startswith("cli://"):
            bin_name = self.base_url[len("cli://") :].strip().split("/")[0]
            if bin_name in CLI_BACKEND_MAP.values():
                return bin_name
        return None

    @property
    def is_anthropic_native(self) -> bool:
        return self.provider == "anthropic" or "api.anthropic.com" in self.base_url.lower()

    @property
    def is_configured(self) -> bool:
        if self.provider == "none":
            return False
        cli = self.cli_backend
        if cli is not None:
            return shutil.which(cli) is not None
        if self.provider in LOCAL_HTTP_PROVIDERS or "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return True
        return bool(self.api_key.strip())

    def public_dict(self) -> dict[str, Any]:
        masked = ""
        if self.provider == "none":
            masked = "Transcript-only mode"
        elif self.api_key:
            masked = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "••••••••"
        elif self.cli_backend:
            masked = f"local {self.cli_backend} CLI"
        elif self.provider in LOCAL_HTTP_PROVIDERS:
            masked = "Local server (no key)"
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "fast_model": self.fast_model,
            "calibration_model": self.calibration_model,
            "has_api_key": bool(self.api_key.strip()) or bool(self.cli_backend and self.is_configured),
            "masked_api_key": masked,
            "is_configured": self.is_configured,
            "needs_onboarding": (not self.is_configured) and (not self._has_saved_provider),
            "presets": LLM_PRESETS,
        }

    def update_and_save(
        self,
        *,
        provider: str,
        base_url: str,
        fast_model: str,
        calibration_model: str,
        api_key: str | None = None,
        parakeet_model: str | None = None,
    ) -> None:
        clean_provider = _sanitize_env_val(provider) or "custom"
        provider_changed = clean_provider != self.provider
        self.provider = clean_provider
        self._has_saved_provider = True
        self.base_url = _sanitize_env_val(base_url)
        self.fast_model = _sanitize_env_val(fast_model)
        self.calibration_model = _sanitize_env_val(calibration_model)

        existing: dict[str, str] = {}
        if self.env_local_path.exists():
            existing = {k: v for k, v in dotenv_values(self.env_local_path).items() if v is not None}

        if self.provider == "none":
            self.api_key = ""
        elif api_key is not None and api_key.strip() != "":
            self.api_key = _sanitize_env_val(api_key)
        elif provider_changed:
            # Never carry over an old provider's API key when switching to a different provider
            provider_env_var = PROVIDER_ENV_KEYS.get(self.provider)
            self.api_key = (
                _sanitize_env_val(existing.get(provider_env_var) or os.getenv(provider_env_var, ""))
                if provider_env_var
                else ""
            )

        if parakeet_model:
            self.parakeet_model = _sanitize_env_val(parakeet_model)

        existing.update(
            {
                "LLM_PROVIDER": self.provider,
                "LLM_BASE_URL": self.base_url,
                "LLM_API_KEY": self.api_key,
                "LLM_FAST_MODEL": self.fast_model,
                "LLM_CALIBRATION_MODEL": self.calibration_model,
                "PARAKEET_MODEL": self.parakeet_model,
            }
        )
        lines = [f"{k}={_sanitize_env_val(v)}" for k, v in existing.items()]
        self.env_local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.env_local_path.with_suffix(".local.tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        tmp_path.replace(self.env_local_path)


def _sanitize_env_val(val: str) -> str:
    return val.replace("\r", "").replace("\n", "").strip()


def _is_reasoning_model(model_name: str) -> bool:
    m = model_name.lower()
    if "deepseek-flash" in m:
        return False
    return any(tok in m for tok in ("reasoner", "deepseek-r1", "/r1", "o1", "o3", "o4-mini", "thinking"))


def _fast_extra_body(model_name: str) -> dict[str, Any] | None:
    """
    DeepSeek V4.1 Flash (`deepseek-flash`) enables Chain-of-Thought thinking by default.
    Explicitly disable thinking mode on latency-critical live calls to avoid multi-second TTFT.
    """
    m = model_name.lower()
    if "deepseek-flash" in m:
        return {"thinking": {"type": "disabled"}}
    return None


def build_compact_session_index(
    session: InterviewSession,
    *,
    recent_turns: int = 6,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    """
    Compact summary index of active interview state (inspired by scratch/triage-lab's `buildCompactIndex`).
    Keeps incremental live LLM calls small and fast even 45 minutes into an interview.
    """
    skip = exclude_ids or set()
    recent = [u for u in session.utterances if u.id not in skip][-recent_turns:]
    return {
        "role_title": session.config.role_title,
        "interview_type": session.config.interview_type,
        "capture_mode": session.config.mode,
        "cards": [
            {
                "id": qa.id,
                "badge": f"{qa.badge_prefix()}{qa.index}",
                "title": qa.title,
                "utt_count": len(qa.utterance_ids),
            }
            for qa in session.qa_groups[-8:]
        ],
        "recent_turns": [
            {
                "id": u.id,
                "mm_ss": fmt_ms(u.start_ms),
                "speaker": u.speaker,
                "text": u.text[:260],
            }
            for u in recent
        ],
    }


def _parse_agy_model_spec(model_spec: str) -> tuple[str, str | None]:
    stripped = re.sub(r"^agy/", "", model_spec.strip())
    if ":" in stripped:
        name, effort = stripped.rsplit(":", 1)
        if effort in ("low", "medium", "high"):
            return name.strip(), effort
    return stripped, None


class InterviewCopilotLLM:
    def __init__(self, config: LLMRuntimeConfig) -> None:
        self.config = config
        self._cached_client: AsyncOpenAI | None = None
        self._cached_client_key: tuple[str, str] | None = None

    def _client(self) -> AsyncOpenAI:
        key = self.config.api_key.strip() or "ollama-local"
        base_url = self.config.base_url
        cache_key = (key, base_url)
        if self._cached_client is None or self._cached_client_key != cache_key:
            self._cached_client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=60.0)
            self._cached_client_key = cache_key
        return self._cached_client

    async def aclose(self) -> None:
        if self._cached_client is not None:
            try:
                await self._cached_client.close()
            except Exception:
                pass
            self._cached_client = None
            self._cached_client_key = None

    async def _call_agy_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        model_name, effort = _parse_agy_model_spec(model)
        format_rule = (
            "Do NOT call any tools or read any files. Use ONLY the input below and reply with ONLY raw valid JSON (no markdown fences)."
            if expect_json
            else "Do NOT call any tools or read any files. Use ONLY the input below and reply with ONLY the requested Markdown response."
        )
        combined = f"<INSTRUCTIONS>\n{system}\n</INSTRUCTIONS>\n\n{format_rule}\n\n<INPUT>\n{user}\n</INPUT>"
        use_stream = on_delta is not None and not expect_json
        args = [
            "agy",
            "--model",
            model_name,
            "--disable-slash-commands",
            "--output-format",
            "stream-json" if use_stream else "json",
        ]
        if effort:
            args.extend(["--effort", effort])
        args.append(f"-p={combined}")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if not use_stream:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"agy CLI timed out after {timeout:.0f}s")

            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"agy CLI exited {proc.returncode}: {(stderr or stdout)[:300]}")

            envelope = _extract_json_object(stdout)
            status = envelope.get("status")
            if status and status != "SUCCESS":
                raise RuntimeError(f"agy status={status}: {envelope.get('error', '')}")
            return str(envelope.get("response", stdout)).strip()

        # Stream NDJSON events from `agy --output-format stream-json`
        deadline = time.monotonic() + timeout
        streamed_chunks: list[str] = []
        final_result: dict[str, Any] | None = None
        assert proc.stdout is not None and proc.stderr is not None
        stderr_task = asyncio.create_task(proc.stderr.read())
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                line_b = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev_type = ev.get("event")
                if ev_type == "step_update":
                    su = ev.get("step_update") or {}
                    if su.get("step_type") == "agent_response":
                        delta = su.get("text_delta")
                        if isinstance(delta, str) and delta:
                            streamed_chunks.append(delta)
                            if on_delta is not None:
                                await on_delta(delta)
                elif ev_type == "result":
                    res_obj = ev.get("result")
                    if isinstance(res_obj, dict):
                        final_result = res_obj
            remaining = max(0.1, deadline - time.monotonic())
            await asyncio.wait_for(proc.wait(), timeout=remaining)
            stderr_b = await stderr_task
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stderr_task.cancel()
            raise TimeoutError(f"agy CLI timed out after {timeout:.0f}s")

        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        if final_result is not None:
            status = final_result.get("status")
            if status and status != "SUCCESS":
                raise RuntimeError(f"agy status={status}: {final_result.get('error', '')}")
            resp_text = str(final_result.get("response", "")).strip()
            if resp_text:
                return resp_text
        if proc.returncode != 0:
            raise RuntimeError(f"agy CLI exited {proc.returncode}: {stderr[:300]}")
        return "".join(streamed_chunks).strip()

    async def _call_pi_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        clean_model = re.sub(r"^pi/", "", model.strip())
        args = [
            "pi",
            "-p",
            "--no-session",
            "--no-context-files",
            "--no-tools",
            "--no-extensions",
            "--no-mcp",
            "--model",
            clean_model,
            "--system-prompt",
            system,
            "--",
            user,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if on_delta is None:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"pi CLI timed out after {timeout:.0f}s")

            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"pi CLI exited {proc.returncode}: {(stderr or stdout)[:300]}")
            return stdout

        deadline = time.monotonic() + timeout
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        chunks: list[str] = []
        assert proc.stdout is not None and proc.stderr is not None
        stderr_task = asyncio.create_task(proc.stderr.read())
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                data = await asyncio.wait_for(proc.stdout.read(256), timeout=remaining)
                if not data:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        chunks.append(tail)
                        await on_delta(tail)
                    break
                text_chunk = decoder.decode(data, final=False)
                if text_chunk:
                    chunks.append(text_chunk)
                    await on_delta(text_chunk)
            remaining = max(0.1, deadline - time.monotonic())
            await asyncio.wait_for(proc.wait(), timeout=remaining)
            stderr_b = await stderr_task
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stderr_task.cancel()
            raise TimeoutError(f"pi CLI timed out after {timeout:.0f}s")

        stdout = "".join(chunks).strip()
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"pi CLI exited {proc.returncode}: {(stderr or stdout)[:300]}")
        return stdout

    async def _call_claude_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        model_name, effort = _parse_agy_model_spec(re.sub(r"^claude/", "", model.strip()))
        format_rule = (
            "Reply with ONLY raw valid JSON (no markdown fences)."
            if expect_json
            else "Reply with ONLY the requested Markdown response."
        )
        sys_prompt = f"{system.strip()}\n\n{format_rule}".strip()
        args = [
            "claude",
            "-p",
            "--no-session-persistence",
            "--tools",
            "",
            "--output-format",
            "text",
        ]
        if model_name:
            args.extend(["--model", model_name])
        if effort:
            args.extend(["--effort", effort])
        if sys_prompt:
            args.extend(["--system-prompt", sys_prompt])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await self._communicate_or_stream_cli(
            proc,
            name="claude CLI",
            stdin_text=user,
            timeout=timeout,
            on_delta=on_delta,
        )

    async def _call_codex_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        clean_model = re.sub(r"^codex/", "", model.strip())
        format_rule = (
            "Do NOT run commands or inspect files. Use ONLY the input below and reply with ONLY raw valid JSON (no markdown fences)."
            if expect_json
            else "Do NOT run commands or inspect files. Use ONLY the input below and reply with ONLY the requested Markdown response."
        )
        combined = f"<INSTRUCTIONS>\n{system}\n</INSTRUCTIONS>\n\n{format_rule}\n\n<INPUT>\n{user}\n</INPUT>"
        fd, out_path = tempfile.mkstemp(prefix="mockery_codex_", suffix=".txt")
        os.close(fd)
        try:
            args = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "-s",
                "read-only",
                "-o",
                out_path,
            ]
            if clean_model:
                args.extend(["-m", clean_model])
            args.append("-")

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout = await self._communicate_or_stream_cli(
                proc,
                name="codex CLI",
                stdin_text=combined,
                timeout=timeout,
                on_delta=None,
            )
            file_out = Path(out_path).read_text(encoding="utf-8", errors="replace").strip()
            result = file_out or stdout
            if on_delta is not None and result:
                await on_delta(result)
            return result
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    async def _call_gemini_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        clean_model = re.sub(r"^gemini/", "", model.strip())
        format_rule = (
            "Do NOT call any tools or read any files. Use ONLY the input below and reply with ONLY raw valid JSON (no markdown fences)."
            if expect_json
            else "Do NOT call any tools or read any files. Use ONLY the input below and reply with ONLY the requested Markdown response."
        )
        combined = f"<INSTRUCTIONS>\n{system}\n</INSTRUCTIONS>\n\n{format_rule}\n\n<INPUT>\n{user}\n</INPUT>"
        args = ["gemini", "-o", "text"]
        if clean_model:
            args.extend(["-m", clean_model])
        args.extend(["-p", ""])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await self._communicate_or_stream_cli(
            proc,
            name="gemini CLI",
            stdin_text=combined,
            timeout=timeout,
            on_delta=on_delta,
        )

    async def _communicate_or_stream_cli(
        self,
        proc: asyncio.subprocess.Process,
        *,
        name: str,
        stdin_text: str,
        timeout: float,
        on_delta: Callable[[str], Awaitable[None]] | None,
    ) -> str:
        stdin_bytes = stdin_text.encode("utf-8")
        if on_delta is None:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"{name} timed out after {timeout:.0f}s")
            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"{name} exited {proc.returncode}: {(stderr or stdout)[:300]}")
            return stdout

        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        deadline = time.monotonic() + timeout
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        chunks: list[str] = []
        stderr_task = asyncio.create_task(proc.stderr.read())
        try:
            proc.stdin.write(stdin_bytes)
            await proc.stdin.drain()
            proc.stdin.close()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                data = await asyncio.wait_for(proc.stdout.read(256), timeout=remaining)
                if not data:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        chunks.append(tail)
                        await on_delta(tail)
                    break
                text_chunk = decoder.decode(data, final=False)
                if text_chunk:
                    chunks.append(text_chunk)
                    await on_delta(text_chunk)
            remaining = max(0.1, deadline - time.monotonic())
            await asyncio.wait_for(proc.wait(), timeout=remaining)
            stderr_b = await stderr_task
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stderr_task.cancel()
            raise TimeoutError(f"{name} timed out after {timeout:.0f}s")

        stdout = "".join(chunks).strip()
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"{name} exited {proc.returncode}: {(stderr or stdout)[:300]}")
        return stdout

    async def _call_anthropic_api(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        base = self.config.base_url.rstrip("/")
        url = f"{base}/messages" if not base.endswith("/messages") else base
        headers = {
            "x-api-key": self.config.api_key.strip(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system.strip():
            payload["system"] = system

        async with httpx.AsyncClient(timeout=timeout) as http:
            if on_delta is not None:
                payload["stream"] = True
                parts: list[str] = []
                async with http.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {body[:300]}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            ev = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if ev.get("type") == "content_block_delta":
                            delta_obj = ev.get("delta") or {}
                            txt = delta_obj.get("text")
                            if isinstance(txt, str) and txt:
                                parts.append(txt)
                                await on_delta(txt)
                return "".join(parts).strip()

            resp = await http.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            blocks = data.get("content") or []
            return "".join(
                str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            ).strip()

    async def test_connection(self, timeout: float = 20.0) -> dict[str, Any]:
        """Run a tiny end-to-end ping against the configured provider to verify credentials and model access."""
        if self.config.provider == "none":
            return {
                "ok": True,
                "latency_ms": 0,
                "model": "none",
                "message": "Transcript-only mode active (local speech recognition ready; no AI provider needed).",
            }
        if not self.config.is_configured:
            return {
                "ok": False,
                "latency_ms": 0,
                "model": self.config.fast_model,
                "error": "No API key or local CLI detected for this provider.",
                "hint": "Paste an API key or choose a detected local CLI / Ollama server.",
            }
        t0 = time.perf_counter()
        try:
            reply = await self._complete_text(
                model=self.config.fast_model,
                system="You are a connectivity test. Reply with exactly: OK",
                user="Ping",
                temperature=0.0,
                max_tokens=16,
                timeout=timeout,
                expect_json=False,
            )
            dt_ms = round((time.perf_counter() - t0) * 1000)
            return {
                "ok": True,
                "latency_ms": dt_ms,
                "model": self.config.fast_model,
                "reply": reply[:60],
                "message": f"Connected to {self.config.fast_model} in {dt_ms} ms.",
            }
        except Exception as exc:
            dt_ms = round((time.perf_counter() - t0) * 1000)
            msg = str(exc)
            hint = _friendly_connection_hint(self.config, msg)
            return {
                "ok": False,
                "latency_ms": dt_ms,
                "model": self.config.fast_model,
                "error": msg[:260],
                "hint": hint,
            }

    async def _complete_text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: float = 30.0,
        expect_json: bool = False,
        disable_fast_thinking: bool = True,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        cli = self.config.cli_backend
        if cli == "agy":
            return await self._call_agy_cli(
                model=model,
                system=system,
                user=user,
                timeout=timeout,
                expect_json=expect_json,
                on_delta=on_delta,
            )
        if cli == "pi":
            return await self._call_pi_cli(
                model=model,
                system=system,
                user=user,
                timeout=timeout,
                on_delta=on_delta,
            )
        if cli == "claude":
            return await self._call_claude_cli(
                model=model,
                system=system,
                user=user,
                timeout=timeout,
                expect_json=expect_json,
                on_delta=on_delta,
            )
        if cli == "codex":
            return await self._call_codex_cli(
                model=model,
                system=system,
                user=user,
                timeout=timeout,
                expect_json=expect_json,
                on_delta=on_delta,
            )
        if cli == "gemini":
            return await self._call_gemini_cli(
                model=model,
                system=system,
                user=user,
                timeout=timeout,
                expect_json=expect_json,
                on_delta=on_delta,
            )
        if self.config.is_anthropic_native:
            return await self._call_anthropic_api(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                on_delta=on_delta,
            )

        client = self._client()
        is_reasoning = _is_reasoning_model(model)
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if disable_fast_thinking:
            extra = _fast_extra_body(model)
            if extra:
                kwargs["extra_body"] = extra

        if on_delta is not None:
            stream = await client.chat.completions.create(**kwargs, stream=True, timeout=timeout)
            parts: list[str] = []
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    await on_delta(delta)
            return "".join(parts).strip()

        resp = await client.chat.completions.create(**kwargs, timeout=timeout)
        return (resp.choices[0].message.content or "").strip()

    async def suggest_probes(self, session: InterviewSession, qa: QAGroup) -> DiveDeeperResponse:
        # Include live in-progress (`final=False`) utterances so Dive Deeper works mid-answer!
        utts = session.utterances_for_card(qa, include_drafts=True)
        rev_sum = sum(u.rev for u in utts)
        if not self.config.is_configured:
            return self._heuristic_fallback_probes(session, qa, len(utts), rev_sum)

        transcript_text = "\n".join(
            f"[{fmt_ms(u.start_ms)} {u.speaker.upper()}{'' if u.final else ' (speaking live)'}]: {u.text}"
            for u in utts
        )

        # Pass BOTH questions the interviewer already asked AND prior AI suggestions so none repeat!
        already_asked = [u.text for u in utts if u.speaker == "interviewer"]
        prior_suggested = [p.question for h in qa.dive_deeper_history for p in h.probes]
        do_not_repeat = (already_asked[-6:] + prior_suggested[-6:])[-10:]
        avoid_block = ""
        if do_not_repeat:
            avoid_block = (
                "\nQuestions already asked or suggested (DO NOT repeat or paraphrase these):\n- "
                + "\n- ".join(do_not_repeat)
            )

        probe_guidance = _probe_guidance_for(session.config.interview_type)
        system_prompt = (
            "You are a fast, pragmatic real-time mock interview co-pilot helping an interviewer probe deeper. "
            "Analyze the exchange (including any in-progress candidate answer) and output ONLY valid JSON:\n"
            '{"quick_read": "<1 crisp sentence on what is solid so far vs where to probe next>", '
            '"probes": [{"angle": "<2-3 word label>", "question": "<exact natural follow-up under 24 words>", "why": "<6-12 word reason>"}]}\n'
            f"{probe_guidance}\n"
            "Keep questions natural and human—do NOT police 'I' vs 'we' pronouns or demand exact numbers where qualitative reasoning makes more sense."
        )

        user_prompt = (
            f"Target Role: {session.config.role_title} | Focus: {session.config.interview_type}\n"
            f"Rubric Context: {session.config.rubric_notes or 'Standard pragmatic engineering bar'}\n\n"
            f"Active Card ({qa.badge_prefix()}{qa.index}: {qa.title}):\n{transcript_text}"
            f"{avoid_block}"
        )

        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system=system_prompt,
                user=user_prompt,
                temperature=0.35,
                max_tokens=420,
                timeout=35.0,
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            probes = [
                ProbeSuggestion(
                    angle=str(item.get("angle", "Depth")),
                    question=str(item.get("question", "")).strip(),
                    why=str(item.get("why", "")).strip(),
                )
                for item in parsed.get("probes", [])[:3]
                if item.get("question")
            ]
            if not probes:
                return self._heuristic_fallback_probes(session, qa, len(utts), rev_sum)
            return DiveDeeperResponse(
                qa_id=qa.id,
                quick_read=str(parsed.get("quick_read", "Press for individual scope, numbers, and tradeoffs.")),
                probes=probes,
                model_used=self.config.fast_model,
                based_on_utt_count=len(utts),
                based_on_rev_sum=rev_sum,
            )
        except Exception as exc:
            logger.warning("LLM suggest_probes failed (%s); using local heuristic", exc)
            fb = self._heuristic_fallback_probes(session, qa, len(utts), rev_sum)
            fb.quick_read = f"(Offline heuristic — {exc.__class__.__name__}) {fb.quick_read}"
            return fb

    async def triage_turn(
        self,
        session: InterviewSession,
        utterance: Utterance,
        *,
        diarize: bool = True,
    ) -> TurnTriagePlan | None:
        """
        Scratch-style typed op planner for a single finalized turn using a compact session index.
        Handles: (1) single-mic speaker diarization + splitting rapid <700ms mixed-speaker handoffs,
        (2) light STT proper-noun cleanup without changing meaning, (3) confidence-gated card pivots.
        """
        if not self.config.is_configured:
            return None

        plans = await self.triage_window(session, [utterance], diarize=diarize)
        return plans[0] if plans else None

    async def triage_window(
        self,
        session: InterviewSession,
        utterances: list[Utterance],
        *,
        diarize: bool = True,
    ) -> list[TurnTriagePlan]:
        """
        Batch-friendly sliding-window Turn Triage (used both for single live turns and fast
        multi-turn diarization/grouping evaluation on recorded interviews).
        """
        if not self.config.is_configured or not utterances:
            return []

        index = build_compact_session_index(
            session,
            recent_turns=6,
            exclude_ids={u.id for u in utterances},
        )
        incoming_payload = [
            {
                "id": u.id,
                "mm_ss": fmt_ms(u.start_ms),
                "dur_sec": round(max(0, u.end_ms - u.start_ms) / 1000.0, 1),
                "current_speaker": u.speaker,
                "text": u.text,
            }
            for u in utterances
        ]

        diarize_rules = (
            "1. SPEAKER DIARIZATION & RAPID HANDOFF SPLITTING (CRITICAL):\n"
            "   - The audio comes from a single shared mic/track, so `current_speaker` may be wrong AND a single VAD segment may contain BOTH speakers back-to-back without a pause.\n"
            "   - For each incoming utterance, output `segments`: 1 segment if a single person spoke, OR 2+ segments split cleanly at the exact sentence/clause where the speaker switched (`interviewer` <-> `candidate`).\n"
            "   - Interviewer cues: introduces the mock, asks primary questions or follow-ups, praises ('Cool, that makes sense...'), transitions ('moving on to the next question').\n"
            "   - Candidate cues: answers questions in first person, tells stories about past roles/projects, explains technical work.\n"
            "   - Preserve almost all words verbatim; ONLY fix unmistakable STT proper-noun/domain typos (e.g. 'for deployed' -> 'forward deployed') without paraphrasing.\n"
            if diarize
            else "1. SPEAKER LABELS ARE HARDWARE-SEPARATED:\n"
            "   - Keep `segments` as a single item with the exact `current_speaker` and original `text`.\n"
        )

        system_prompt = (
            "You are a fast, precision Interview Turn Triage planner (inspired by operation-log triage).\n"
            "Given a compact `session_index` and 1+ incoming finalized transcript utterances, emit ONLY valid JSON:\n"
            '{"plans": [{"utterance_id": "<exact id>", '
            '"segments": [{"speaker": "interviewer"|"candidate", "text": "<text>"}], '
            '"card_action": "keep_in_current"|"open_new_card"|"ignore_backchannel", '
            '"suggested_title": "<5-8 word card title ONLY if open_new_card else empty string>", '
            '"confidence": "high"|"medium"|"low", '
            '"rationale": "<short 6-14 word reason>"}]}\n\n'
            "RULES:\n"
            f"{diarize_rules}"
            "2. CARD BOUNDARIES (DO NOT OVER-SPLIT):\n"
            "   - Use `open_new_card` ONLY when the interviewer introduces a BRAND NEW primary interview question or major design phase.\n"
            "   - Keep ALL follow-up probes, clarifying questions, and candidate answers inside the current card (`keep_in_current`).\n"
            "   - Use `ignore_backchannel` only for pure 1-3 word filler ('yeah', 'uh-huh', 'cool') that adds no content.\n"
            "3. CONFIDENCE CALIBRATION:\n"
            "   - `high`: Unambiguous speaker role, clear handoff boundary, or explicit new question ('moving on to...').\n"
            "   - `medium` / `low`: Genuinely ambiguous speaker attribution or borderline new-question vs follow-up pivot.\n"
        )

        user_prompt = json.dumps(
            {"session_index": index, "incoming_utterances": incoming_payload},
            ensure_ascii=False,
        )

        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system=system_prompt,
                user=user_prompt,
                temperature=0.0,
                max_tokens=max(450, 220 * len(utterances)),
                timeout=max(25.0, 15.0 + 4.0 * len(utterances)),
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            raw_plans = parsed.get("plans")
            if not isinstance(raw_plans, list):
                # Support single-object fallback if model omits wrapper
                if "utterance_id" in parsed or "segments" in parsed:
                    raw_plans = [parsed]
                else:
                    return []

            valid_ids = {u.id: u for u in utterances}
            out: list[TurnTriagePlan] = []
            for item in raw_plans:
                if not isinstance(item, dict):
                    continue
                uid = str(item.get("utterance_id", "")).strip()
                if uid not in valid_ids:
                    if len(utterances) == 1 and not uid:
                        uid = utterances[0].id
                    else:
                        continue
                orig = valid_ids[uid]
                segs: list[TurnSegment] = []
                for s in item.get("segments") or []:
                    if not isinstance(s, dict):
                        continue
                    spk = str(s.get("speaker", "")).strip().lower()
                    txt = str(s.get("text", "")).strip()
                    if spk in ("interviewer", "candidate") and txt:
                        segs.append(TurnSegment(speaker=spk, text=txt))  # type: ignore[arg-type]
                if not segs:
                    segs = [TurnSegment(speaker=orig.speaker, text=orig.text)]
                action = str(item.get("card_action", "keep_in_current")).strip()
                if action not in ("keep_in_current", "open_new_card", "ignore_backchannel"):
                    action = "keep_in_current"
                conf = str(item.get("confidence", "high")).strip().lower()
                if conf not in ("high", "medium", "low"):
                    conf = "medium"
                out.append(
                    TurnTriagePlan(
                        utterance_id=uid,
                        segments=segs,
                        card_action=action,  # type: ignore[arg-type]
                        suggested_title=str(item.get("suggested_title", "")).strip(),
                        confidence=conf,  # type: ignore[arg-type]
                        rationale=str(item.get("rationale", "")).strip(),
                    )
                )
            return out
        except Exception as exc:
            logger.warning("Turn triage LLM call failed: %s", exc)
            return []

    async def classify_topic_shift(
        self,
        session: InterviewSession,
        qa: QAGroup,
        latest_interviewer_text: str,
    ) -> tuple[bool, str]:
        utts = session.utterances_for_card(qa)
        if not self.config.is_configured or len(utts) < 2:
            return (False, qa.title)

        prior_snippet = "\n".join(f"[{u.speaker.upper()}]: {u.text[:220]}" for u in utts[:-1][-4:])
        prompt = (
            "Does the interviewer's latest utterance pivot to a BRAND NEW unrelated interview question, "
            "or continue/probe the existing Q&A thread?\n"
            'Reply ONLY with JSON: {"is_new_topic": true|false, "title": "<5-8 word topic title>"}\n\n'
            f"Existing Q&A ({qa.title}):\n{prior_snippet}\n\n"
            f"Latest Interviewer Utterance:\n{latest_interviewer_text}"
        )
        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system="Classify whether an interview turn pivots to a new unrelated question. Return ONLY valid JSON.",
                user=prompt,
                temperature=0.0,
                max_tokens=90,
                timeout=20.0,
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            return (bool(parsed.get("is_new_topic", False)), str(parsed.get("title", "")).strip() or qa.title)
        except Exception:
            return (False, qa.title)

    async def check_speaker_role_orientation(self, session: InterviewSession) -> dict[str, Any]:
        """
        One-shot fast semantic check on early single-mic turns to detect whether the two
        acoustic speaker clusters (`INTERVIEWER` vs `CANDIDATE`) started globally inverted
        (for example, when the candidate spoke first at t=00:00 during greeting).
        Fails closed (`inverted=False`) on ambiguity or error.
        """
        default = {"inverted": False, "confidence": "low", "rationale": ""}
        if not self.config.is_configured:
            return default
        finals = [u for u in session.utterances if u.final and u.text.strip()]
        if len(finals) < 6:
            return default
        sample = finals[:16]
        speakers_seen = {u.speaker for u in sample}
        if len(speakers_seen) < 2:
            return default

        lines = [
            f"- [{fmt_ms(u.start_ms)} {u.speaker.upper()}]: {u.text[:240]}"
            for u in sample
        ]
        prompt = (
            f"Target Role: {session.config.role_title} ({session.config.interview_type})\n"
            "Below are the opening turns of a live 1-on-1 mock interview transcribed from a single mic.\n"
            "An acoustic diarizer separated the two voices accurately into two clusters labeled INTERVIEWER and CANDIDATE, "
            "but it arbitrarily assumed the first voice heard was INTERVIEWER.\n"
            "Decide whether the two role labels are CORRECTLY oriented or GLOBALLY INVERTED.\n"
            "- Clues for the real INTERVIEWER: introduces the problem/prompt, explains format or time limits, asks probing questions, grants scope clarifications, says 'I was at <Company>' or 'let me start the timer'.\n"
            "- Clues for the real CANDIDATE: says they have an upcoming interview, clarifies requirements, proposes designs/answers, asks the interviewer for permission/confirmation.\n"
            "Return ONLY valid JSON:\n"
            '{"inverted": true|false, "confidence": "high"|"medium"|"low", "rationale": "<1 short sentence>"}\n'
            "Use confidence='high' ONLY when the conversational roles are unmistakable.\n\n"
            "Opening transcript:\n" + "\n".join(lines)
        )
        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system="Determine whether INTERVIEWER and CANDIDATE voice labels are globally inverted. Return ONLY valid JSON.",
                user=prompt,
                temperature=0.0,
                max_tokens=200,
                timeout=25.0,
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            conf = str(parsed.get("confidence", "low")).strip().lower()
            if conf not in ("high", "medium", "low"):
                conf = "low"
            return {
                "inverted": bool(parsed.get("inverted", False)),
                "confidence": conf,
                "rationale": str(parsed.get("rationale", "")).strip(),
            }
        except Exception as exc:
            logger.warning("Speaker role-orientation check skipped: %s", exc)
            return default

    async def regroup_cards_full_context(self, session: InterviewSession) -> list[dict[str, Any]]:
        """
        End-of-session full-context pass that reviews all non-warmup utterances and returns clean
        card start boundaries `[{"start_utterance_id": "utt_...", "title": "..."}]`.
        """
        if not self.config.is_configured:
            return []
        warm_set = set(session.warmup_utterance_ids)
        utts = [u for u in session.utterances if u.id not in warm_set]
        if len(utts) < 3:
            return []

        mode_instruction = (
            "Group into major System Design / Coding PHASES (e.g. Requirements & Scope, High-Level Architecture, Deep Dive / Implementation, Bottlenecks & Verification)."
            if session.config.interview_type in ("system_design", "coding")
            else "Group into primary Behavioral / Technical QUESTION threads (keep follow-up probes and clarifications inside their parent question)."
        )

        lines = [f"- id={u.id} [{fmt_ms(u.start_ms)} {u.speaker.upper()}]: {u.text[:260]}" for u in utts]
        prompt = (
            f"{mode_instruction}\n"
            "Return ONLY valid JSON: "
            '{"cards": [{"start_utterance_id": "<exact utt id starting each card>", "title": "<5-8 word title>"}]}\n'
            f"First card MUST start at `{utts[0].id}`.\n\nUtterances:\n" + "\n".join(lines)
        )
        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system="Segment an interview transcript into clean Q&A or phase card boundaries. Return ONLY valid JSON.",
                user=prompt,
                temperature=0.1,
                max_tokens=1000,
                timeout=45.0,
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            cards = parsed.get("cards", [])
            return cards if isinstance(cards, list) else []
        except Exception as exc:
            logger.warning("Full-context regroup skipped: %s", exc)
            return []

    async def generate_calibration_report(
        self,
        session: InterviewSession,
        *,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        metrics = session.compute_metrics()
        if not self.config.is_configured:
            return self._offline_calibration_summary(session)

        card_blocks: list[str] = []
        for qa, stat in zip(session.qa_groups, metrics.per_card):
            turns = "\n".join(
                f"  - [{fmt_ms(u.start_ms)} | {u.id} | {u.speaker.upper()}]: {u.text}"
                for u in session.utterances_for_card(qa)
            )
            note_line = f"\n  - [INTERVIEWER NOTE]: {qa.notes}" if qa.notes.strip() else ""
            card_blocks.append(
                f"### {stat.badge}: {qa.title} "
                f"(Code-computed span={stat.span_seconds}s, interviewer_speak={stat.interviewer_speak_seconds}s/{stat.interviewer_words}w, "
                f"candidate_speak={stat.candidate_speak_seconds}s/{stat.candidate_words}w)\n{turns}{note_line}"
            )

        self_rating_block = "Interviewer did not submit a pre-rating."
        if session.self_rating is not None:
            sr = session.self_rating
            rating_lines = [f"- Hire Signal: {sr.hire_signal}"]
            if sr.perceived_level and sr.perceived_level.strip():
                rating_lines.append(f"- Perceived Level: {sr.perceived_level.strip()}")
            if sr.gut_rationale and sr.gut_rationale.strip():
                rating_lines.append(f"- Initial Rationale: {sr.gut_rationale.strip()}")
            if sr.biggest_concern and sr.biggest_concern.strip():
                rating_lines.append(f"- Biggest Concern: {sr.biggest_concern.strip()}")
            self_rating_block = "\n".join(rating_lines)

        rubric_block = _calibration_rubric_for(session.config.interview_type)
        system_prompt = (
            "You are a pragmatic Staff/Principal engineering peer coach and FAANG hiring-committee calibrator evaluating BOTH the interviewer's judgment and the candidate after a live mock interview.\n\n"
            f"INTERVIEW-TYPE EVALUATION RUBRIC ({session.config.interview_type}):\n"
            f"{rubric_block}\n\n"
            "GROUNDING & REALISM RULES:\n"
            "1. EVERY claim about what the candidate or interviewer said MUST cite exact timestamps and utterance IDs like `[04:12, utt_a1b2c3d4]`.\n"
            "2. Use ONLY the code-computed timing and word-ratio numbers provided—do not invent arithmetic.\n"
            "3. Evaluate like a calibrated human hiring committee, NOT an exhaustive textbook checklist or resume scanner:\n"
            "   - **Respect the 35–45 minute clock & interviewer steering:** A real live interview can only deep-dive 2–3 areas. Separate **explored weaknesses** (mistakes made where the candidate actually spent time or got probed) from **unprobed / out-of-time areas** (topics the interviewer affirmed and moved past, or never steered into). Do NOT downgrade a candidate's hire verdict merely because theoretical bottlenecks went unprobed—flag unprobed areas as missed interviewer probes in Section 3 instead.\n"
            "   - **Calibrate strictly to the Target Role / Level:** Judge E4/Mid-Level, E5/Senior, and E6/Staff against realistic passing bars for THAT level (do not fail an E4 candidate solely for Senior-depth distributed-systems gaps when core fundamentals and coachability are solid; do not withhold Strong Hire E5 / Hire E6 from a candidate who delivered crisp structure, clear tradeoffs, and clean recovery on every explored deep dive just because other areas were left unasked).\n"
            "   - **Stay technically sharp, never sycophantic:** Still call out genuine technical errors, over-engineering, missed hints, and even interviewer hint inaccuracies candidly—just weigh them proportionally to the target level and real-world hiring bars.\n"
            "   - NEVER police `'I'` vs. `'we'` pronouns or treat pronoun ratios as evidence. Normal people say `'we'` in collaborative design exercises and team discussions.\n"
            "   - NEVER penalize statements merely for lacking exact numbers or memorized statistics when qualitative reasoning, order-of-magnitude intuition, or architectural tradeoffs are what actually matter.\n"
            "   - Recognize that live spoken interviews include thinking out loud, informal phrasing, minor ASR typos, and self-correction—judge the substance of the candidate's engineering thinking.\n"
            "   - If the interviewer left optional pre-rating fields (level/rationale/concern) blank, do NOT scold them for leaving fields blank—just calibrate against the hire signal they submitted.\n"
            "4. Structure your Markdown output into these exact 4 sections:\n"
            "### 1. Calibration Check: Your Blind Call vs. Transcript Evidence\n"
            "- Compare the interviewer's pre-submitted call directly against the transcript evidence (or provide an independent level-calibrated hire signal if none was submitted). Clearly separate what the candidate demonstrated on explored areas from what went unprobed, and state both the hire signal at the target level and any clear leveling nuance.\n\n"
            "### 2. Candidate Signal & Evidence Audit (By Card)\n"
            "- For each card, give a concise **Strong / Mixed / Weak** read grounded in `[MM:SS, utt_id]` citations using the interview-type rubric above (focus on the real strengths, explored gaps, and decision quality—no filler sub-bullets).\n\n"
            "### 3. Interviewer Probing & Pacing Retro\n"
            "- Critique time allocation across cards and talk-to-listen ratio using the provided code metrics.\n"
            "- **Missed Probing Moments:** Cite 2-3 exact `[MM:SS, utt_id]` moments where a sharper follow-up would have unlocked stronger signal (or gently flag any technically imprecise interviewer hints), and write a natural follow-up question to ask next time.\n\n"
            "### 4. Constructive Takeaways for Candidate\n"
            "- 3 concrete, encouraging, high-leverage takeaways the interviewer can share directly with the mock candidate."
        )

        user_prompt = (
            f"Target Role: {session.config.role_title} | Focus: {session.config.interview_type}\n"
            f"Rubric Notes: {session.config.rubric_notes or 'Standard pragmatic engineering bar'}\n\n"
            f"Code-Computed Session Metrics (ground truth):\n"
            f"- Active Span: {metrics.duration_seconds}s\n"
            f"- Speaking Time Ratio: Interviewer {metrics.interviewer_time_pct}% ({metrics.interviewer_speak_seconds}s) vs Candidate {metrics.candidate_time_pct}% ({metrics.candidate_speak_seconds}s)\n"
            f"- Word Ratio: Interviewer {metrics.interviewer_talk_pct}% ({metrics.interviewer_words}w) vs Candidate {metrics.candidate_talk_pct}% ({metrics.candidate_words}w)\n\n"
            f"Interviewer's Blind Pre-Calibration Rating:\n{self_rating_block}\n\n"
            f"Transcript:\n" + ("\n\n".join(card_blocks) or "(Empty session)")
        )

        try:
            is_reasoning = _is_reasoning_model(self.config.calibration_model)
            content = await self._complete_text(
                model=self.config.calibration_model,
                system=system_prompt,
                user=user_prompt,
                temperature=0.25,
                max_tokens=8000 if is_reasoning else 3200,
                timeout=240.0,
                expect_json=False,
                disable_fast_thinking=False,
                on_delta=on_delta,
            )
            return content or self._offline_calibration_summary(session)
        except Exception as exc:
            logger.warning("Calibration LLM call failed: %s", exc)
            return (
                f"> **Note:** LLM call failed (`{exc}`), showing code-computed calibration audit.\n\n"
                + self._offline_calibration_summary(session)
            )

    @staticmethod
    def _heuristic_fallback_probes(
        session: InterviewSession,
        qa: QAGroup,
        utt_count: int,
        rev_sum: int = 0,
    ) -> DiveDeeperResponse:
        itype = session.config.interview_type
        if itype == "system_design":
            probes = [
                ProbeSuggestion(
                    angle="Core Bottleneck",
                    question="Walk me through the hardest path here under peak load—where does this design feel pressure first?",
                    why="Shifts focus from boilerplate setup to the real scaling bottleneck.",
                ),
                ProbeSuggestion(
                    angle="Key Tradeoff",
                    question="What is the strongest alternative architecture you ruled out here, and what tipped your decision?",
                    why="Tests deliberate engineering judgment rather than default choices.",
                ),
                ProbeSuggestion(
                    angle="Failure Mode",
                    question="What happens if a server drops mid-request or a client loses connectivity abruptly here?",
                    why="Probes resilience, delivery guarantees, and edge-case recovery.",
                ),
            ]
        elif itype == "coding":
            probes = [
                ProbeSuggestion(
                    angle="Edge Cases",
                    question="Which tricky inputs or boundary conditions could break this approach right now?",
                    why="Checks correctness rigor before moving on.",
                ),
                ProbeSuggestion(
                    angle="Complexity Tradeoff",
                    question="If input size grew much larger, what's the dominant time or memory bottleneck here?",
                    why="Tests asymptotic intuition and optimization tradeoffs.",
                ),
                ProbeSuggestion(
                    angle="Testing Strategy",
                    question="How would you structure tests or invariants to be confident this stays correct under changes?",
                    why="Surfaces practical engineering discipline.",
                ),
            ]
        else:
            probes = [
                ProbeSuggestion(
                    angle="Decision Tradeoff",
                    question="What was the hardest decision or tradeoff on this project, and what alternatives did you rule out?",
                    why="Tests judgment and reasoning behind the chosen path.",
                ),
                ProbeSuggestion(
                    angle="Your Role",
                    question="Of the pieces you described, which technical or cross-team part did you personally drive end-to-end?",
                    why="Clarifies concrete scope naturally without nitpicking phrasing.",
                ),
                ProbeSuggestion(
                    angle="Hindsight & Risk",
                    question="What surprised you or broke along the way, and what would you do differently next time?",
                    why="Surfaces real-world reflection and senior ownership.",
                ),
            ]

        return DiveDeeperResponse(
            qa_id=qa.id,
            quick_read="Configure an LLM in Settings (or local `agy` CLI) for transcript-tailored probes; showing offline suggestions.",
            probes=probes[:3],
            model_used="local-heuristic",
            based_on_utt_count=utt_count,
            based_on_rev_sum=rev_sum,
        )

    @staticmethod
    def _offline_calibration_summary(session: InterviewSession) -> str:
        m = session.compute_metrics()
        lines: list[str] = ["### Code-Computed Session & Pacing Audit"]
        if session.self_rating is not None:
            sr = session.self_rating
            level_note = f" ({sr.perceived_level})" if sr.perceived_level else ""
            rationale_note = f" — *{sr.gut_rationale}*" if sr.gut_rationale else ""
            lines.append(f"- **Your Blind Pre-Rating:** `{sr.hire_signal}`{level_note}{rationale_note}")
        lines.extend(
            [
                f"- **Speaking Time Ratio:** Interviewer **{m.interviewer_time_pct}%** ({m.interviewer_speak_seconds}s) / Candidate **{m.candidate_time_pct}%** ({m.candidate_speak_seconds}s)",
                f"- **Talk-to-Listen Word Ratio:** Interviewer **{m.interviewer_talk_pct}%** ({m.interviewer_words}w) / Candidate **{m.candidate_talk_pct}%** ({m.candidate_words}w)",
                "",
                "### Per-Card Timing & Pacing Breakdown",
            ]
        )
        for qa, stat in zip(session.qa_groups, m.per_card):
            utts = session.utterances_for_card(qa)
            first_cite = f"[{fmt_ms(utts[0].start_ms)}, {utts[0].id}]" if utts else "[—]"
            lines.append(
                f"- **{stat.badge} ({qa.title})** {first_cite}: Span **{stat.span_seconds}s** | Interviewer **{stat.interviewer_words}w** ({stat.interviewer_speak_seconds}s) | Candidate **{stat.candidate_words}w** ({stat.candidate_speak_seconds}s)"
            )
        return "\n".join(lines)


def _probe_guidance_for(interview_type: str) -> str:
    if interview_type == "system_design":
        return (
            "This is a SYSTEM DESIGN interview. Return 3 high-leverage follow-up probes covering:\n"
            "(1) the core architectural bottleneck or data access/fan-out pattern in what they just proposed,\n"
            "(2) a key design tradeoff or alternative choice (storage, schema, protocol, or abstraction),\n"
            "(3) a concrete failure mode, consistency edge case, or operational constraint.\n"
            "Do NOT ask about past resume experience or force back-of-the-envelope arithmetic unless a scale mismatch breaks the design."
        )
    if interview_type == "coding":
        return (
            "This is a CODING interview. Return 3 high-leverage follow-up probes covering:\n"
            "(1) subtle edge cases, invariants, or correctness bugs,\n"
            "(2) time/space complexity tradeoffs or scaling bottlenecks,\n"
            "(3) code clarity, decomposition, or verification/testing strategy."
        )
    if interview_type == "behavioral":
        return (
            "This is a BEHAVIORAL interview. Return 3 high-leverage follow-up probes covering:\n"
            "(1) the specific technical/org decision or piece of the work they drove (naturally, never lecturing on 'I' vs 'we'),\n"
            "(2) constraints, alternatives considered, or how they navigated disagreement/ambiguity,\n"
            "(3) concrete outcomes (qualitative or quantitative) and what went wrong or what they'd change in hindsight."
        )
    return (
        "Return 3 high-leverage follow-up probes covering:\n"
        "(1) deeper technical mechanism ('how does that actually work?'),\n"
        "(2) key tradeoffs or alternatives ruled out,\n"
        "(3) realistic edge cases, constraints, or failure modes."
    )


def _calibration_rubric_for(interview_type: str) -> str:
    if interview_type == "system_design":
        return (
            "- This is a collaborative **System Design** exercise, NOT a resume review. Never audit past-job credentials or `'I'` vs `'we'` ownership.\n"
            "- Judge signal on:\n"
            "  1. **Scoping & prioritization**: clarifying core requirements and moving smoothly into the high-level design and explored deep dives.\n"
            "  2. **Architecture & data/access patterns**: coherent end-to-end flows, clean data modeling/abstractions, sensible storage & transport choices.\n"
            "  3. **Scale intuition & bottlenecks (on explored areas)**: recognizing where the design breaks when probed or when scale is central (e.g., large-channel fan-out, hot keys, delivery guarantees, disconnects). Value clear qualitative reasoning about orders of magnitude and bottlenecks—do NOT ding them for skipping whiteboard arithmetic or for bottlenecks the interviewer never steered into during a 35–45 minute clock.\n"
            "  4. **Tradeoffs & collaboration**: articulating problem → options → tradeoffs → decision clearly, and defending or revising choices cleanly when probed.\n"
            "- **Level-Calibrated Passing Bars (35–45 min live interview):**\n"
            "  - **Mid-Level (E4 / L4):** Passing (`hire`) = clean requirements & scoping, coherent end-to-end HLD, sensible baseline DB/transport choices, and strong coachability. Technical mistakes at Senior depth (e.g., hot-key sharding at 100M+ scale, stateful socket ownership internals, complex race recovery) or mild over-engineering are *growth areas to flag*, not automatic No Hires if fundamentals and coachability are solid.\n"
            "  - **Senior (E5 / L5):** Passing (`hire`) = drives structure independently, clearly weighs tradeoffs across options, and lands solid correctness/scaling answers on the 2–3 deep dives actually explored. `strong_hire` at E5 = crisp pacing, unprompted problem→options→decision loops, clean correctness, and rapid recovery on extensions—does NOT require exhaustively covering unasked theoretical failure modes.\n"
            "  - **Staff (E6 / L6):** Passing (`hire`) = crisp end-to-end mastery, proactive bottleneck framing, production-grade tradeoff depth, and clean adaptation on hard extensions. If the interviewer paced the 35–40 min loop like a Senior interview and affirmed sections without pushing into Staff-only depth, distinguish demonstrated `strong_hire` (E5) / `hire` (E6) signal from unprobed Staff topics instead of defaulting to Below Bar."
        )
    if interview_type == "coding":
        return (
            "- This is a **Coding** interview, NOT a resume review. Never audit `'I'` vs `'we'` ownership or past-job metrics.\n"
            "- Judge signal on:\n"
            "  1. **Problem decomposition & clarity**: understanding requirements and structuring a clean approach before diving in.\n"
            "  2. **Algorithmic & data-structure judgment**: choosing appropriate abstractions and understanding complexity tradeoffs.\n"
            "  3. **Correctness & edge cases**: catching boundary conditions, tracing/testing thoughtfully, and debugging methodically.\n"
            "  4. **Communication**: explaining reasoning clearly and responding well to hints or probes."
        )
    if interview_type == "behavioral":
        return (
            "- This is a **Behavioral** interview focused on real past work, judgment, and collaboration.\n"
            "- Judge signal on:\n"
            "  1. **Clarity of role & impact**: can we understand what the candidate actually drove, built, or decided? (Normal people say `'we'` about team efforts—only flag ownership as unclear if their personal contribution stays genuinely ambiguous, never count pronouns.)\n"
            "  2. **Decision quality & tradeoffs**: navigating ambiguity, weighing alternatives, handling pushback or cross-team alignment.\n"
            "  3. **Outcomes & reflection**: concrete results (qualitative impact, shipped capabilities, team/customer outcomes, or rough scale are all valid—do NOT penalize a candidate just because they don't recall exact percentages or metrics from years ago) and honest lessons learned."
        )
    return (
        "- Evaluate technical depth, clarity of reasoning, tradeoff awareness, and communication matched to the actual topics discussed.\n"
        "- Judge the substance of engineering thinking and decision quality—do NOT police pronouns or penalize missing exact numbers where qualitative reasoning fits."
    )


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        return json.loads(cleaned)
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(cleaned, start)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    end = cleaned.rfind("}")
    if end > start:
        return json.loads(cleaned[start : end + 1])
    return json.loads(cleaned)


def _friendly_connection_hint(cfg: LLMRuntimeConfig, err_msg: str) -> str:
    low = err_msg.lower()
    if cfg.cli_backend:
        return f"Make sure `{cfg.cli_backend}` is signed in on your Mac (try running it once in Terminal)."
    if "401" in low or "unauthorized" in low or "invalid api key" in low or "authentication" in low:
        return "Double-check that your API key is copied completely and has not expired."
    if "404" in low or "model" in low and "not found" in low:
        return f"Model `{cfg.fast_model}` was not found by this provider — check the model name in Advanced settings."
    if "429" in low or "quota" in low or "rate limit" in low:
        return "Provider returned a rate limit or quota error — verify billing/free-tier quota."
    if cfg.provider in LOCAL_HTTP_PROVIDERS or "connection" in low or "refused" in low:
        if cfg.provider == "ollama":
            return "Ollama does not appear to be running. Open Ollama (or run `ollama serve`) and pull a model first."
        if cfg.provider == "lmstudio":
            return "LM Studio local server is not reachable on port 1234. Start the Local Server in LM Studio first."
        return "Could not reach the endpoint URL. Check your internet connection or Base URL."
    return "Check your credentials, model name, and network connection."


def get_system_memory_gb() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return max(1, round((pages * page_size) / (1024**3)))
    except (ValueError, OSError, AttributeError):
        pass
    return None


async def _probe_ollama(timeout: float = 0.6) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    str(m.get("name", "")).strip()
                    for m in (data.get("models") or [])
                    if isinstance(m, dict) and m.get("name")
                ]
                return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}


async def _probe_lmstudio(timeout: float = 0.6) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get("http://127.0.0.1:1234/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    str(m.get("id", "")).strip()
                    for m in (data.get("data") or [])
                    if isinstance(m, dict) and m.get("id")
                ]
                return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}


async def discover_available_providers() -> dict[str, Any]:
    """
    Fast (<600ms) local discovery of:
      - Installed subscription CLIs (`claude`, `codex`, `gemini`, `agy`, `pi`)
      - Running local OpenAI-compatible runtimes (Ollama, LM Studio) and their installed models
      - Host unified memory in GB (to guide local-LLM expectations)
    """
    detected_clis: list[dict[str, Any]] = []
    for preset_id, preset in LLM_PRESETS.items():
        if preset.get("category") != "cli":
            continue
        cli_bin = str(preset.get("cli_bin") or "")
        if cli_bin and shutil.which(cli_bin):
            detected_clis.append(
                {
                    "provider": preset_id,
                    "cli_bin": cli_bin,
                    "label": preset["label"],
                    "badge": preset.get("badge", "Detected"),
                    "hint": preset.get("hint", ""),
                    "base_url": preset["base_url"],
                    "fast_model": preset["fast_model"],
                    "calibration_model": preset["calibration_model"],
                }
            )

    ollama_info, lmstudio_info = await asyncio.gather(_probe_ollama(), _probe_lmstudio())

    env_keys_present = [
        prov
        for prov, env_name in PROVIDER_ENV_KEYS.items()
        if bool(os.getenv(env_name, "").strip())
    ]

    return {
        "detected_clis": detected_clis,
        "local_servers": {
            "ollama": ollama_info,
            "lmstudio": lmstudio_info,
        },
        "env_keys_present": env_keys_present,
        "system_memory_gb": get_system_memory_gb(),
    }

