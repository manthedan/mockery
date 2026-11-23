from __future__ import annotations

import asyncio
import codecs
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values
from openai import AsyncOpenAI

from .levels import normalize_level
from .models import (
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_PRACTICE_ROLE_TITLE,
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
from .practice_conversation import (
    build_practice_turn_messages,
    messages_prefix_digest,
    render_flat_messages,
)
from .prompts import (
    calibration_rubric_for,
    practice_debrief_rubric_args,
    practice_focus_brief,
    practice_hint_behavioral_nudge,
    practice_reply_rubric_args,
    practice_signal_schema,
    practice_tone_instruction,
    probe_guidance_for,
    render_prompt,
    triage_diarize_rules,
)
from .rubrics import CARL, SIGNAL_LEVELS, SIGNAL_SLOTS, Rubric

if TYPE_CHECKING:
    from .practice import PracticeSession

logger = logging.getLogger(__name__)


PROVIDER_ENV_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "omlx": "OMLX_API_KEY",
}

CLI_BACKEND_MAP: dict[str, str] = {
    "claude_cli": "claude",
    "codex_cli": "codex",
    "gemini_cli": "gemini",
    "agy": "agy",
    "pi": "pi",
}

LOCAL_HTTP_PROVIDERS = {"omlx", "ollama", "lmstudio"}


class LLMRuntimeConfig:
    """Manages OpenAI-compatible, native Anthropic, local oMLX/Ollama/LM Studio, & local CLI LLM + ASR config with `.env.local` persistence."""

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
    def is_local_http(self) -> bool:
        if self.cli_backend is not None or self.provider == "none":
            return False
        if self.provider in LOCAL_HTTP_PROVIDERS:
            return True
        try:
            host = (urlparse(self.base_url).hostname or "").strip().lower()
        except ValueError:
            host = ""
        if not host:
            return False
        if host == "localhost" or host.endswith(".localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def is_configured(self) -> bool:
        if self.provider == "none":
            return False
        cli = self.cli_backend
        if cli is not None:
            return shutil.which(cli) is not None
        if self.is_local_http:
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


_OPUS_4_MINOR_RE = re.compile(r"opus[-_.]4[-_.](\d+)")
_CLAUDE_5_PLUS_RE = re.compile(r"claude[-_.](?:opus|sonnet|haiku|fable)[-_.](\d+)")


def _anthropic_supports_temperature(model_name: str) -> bool:
    """Newer Claude models (Opus 4.7+, Mythos, and 5.x series) reject explicit `temperature` with HTTP 400."""
    m = model_name.lower().strip()
    if _is_reasoning_model(m):
        return False
    if "mythos" in m:
        return False
    opus4 = _OPUS_4_MINOR_RE.search(m)
    if opus4 and int(opus4.group(1)) >= 7:
        return False
    c5 = _CLAUDE_5_PLUS_RE.search(m)
    if c5 and int(c5.group(1)) >= 5:
        return False
    return True


def _head_tail_excerpt(text: str, max_chars: int) -> str:
    """Keep both opening setup (35%) and closing outcome/learning (65%) strictly within `max_chars`."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n[... middle omitted ...]\n"
    if max_chars <= len(marker) + 4:
        return text[:max_chars]
    usable = max_chars - len(marker)
    head_len = max(1, int(usable * 0.35))
    tail_len = max(1, usable - head_len)
    return text[:head_len].rstrip() + marker + text[-tail_len:].lstrip()


def _looks_like_truncated_json(raw: str, finish_reason: str | None = None) -> bool:
    """Detect output truncation via provider finish_reason or unbalanced JSON braces on cleaned output."""
    if finish_reason in ("length", "max_tokens"):
        return True
    cleaned = THINK_BLOCK_RE.sub("", raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    if start == -1:
        return False
    tail = cleaned[start:]
    if not tail.endswith("}"):
        return True
    depth = 0
    in_str = False
    esc = False
    for ch in tail:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return in_str or depth > 0


def _extract_usage_counts(usage: Any, *, provider_format: str = "generic") -> dict[str, int | None]:
    """
    Normalize provider-reported token counters across OpenAI, Anthropic, and CLI JSON envelopes.
    `input_tokens` always represents total prompt tokens (including cache reads/writes on Anthropic)
    so counters are directly comparable across providers.
    """
    out: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }
    if usage is None:
        return out
    if isinstance(usage, dict):
        get = usage.get
    else:
        def get(key: str, default: Any = None) -> Any:
            return getattr(usage, key, default)

    raw_in = get("input_tokens")
    if raw_in is None:
        raw_in = get("prompt_tokens")
    out_tok = get("output_tokens")
    if out_tok is None:
        out_tok = get("completion_tokens")
    cache_read = get("cache_read_input_tokens")
    if cache_read is None:
        cache_read = get("cache_read_tokens")
    if cache_read is None:
        cache_read = get("cached_tokens")
    if cache_read is None:
        details = get("prompt_tokens_details")
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens")
        elif details is not None:
            cache_read = getattr(details, "cached_tokens", None)
    cache_write = get("cache_creation_input_tokens")
    if cache_write is None:
        cache_write = get("cache_write_tokens")

    def _valid_int(val: Any) -> int | None:
        return val if isinstance(val, int) and not isinstance(val, bool) and val >= 0 else None

    in_i = _valid_int(raw_in)
    out_i = _valid_int(out_tok)
    cr_i = _valid_int(cache_read)
    cw_i = _valid_int(cache_write)

    # Anthropic's `input_tokens` excludes cache_read_input_tokens and cache_creation_input_tokens
    if provider_format == "anthropic" and in_i is not None:
        in_i = in_i + (cr_i or 0) + (cw_i or 0)

    out["input_tokens"] = in_i
    out["output_tokens"] = out_i
    out["cache_read_tokens"] = cr_i
    out["cache_write_tokens"] = cw_i
    return out


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
    Compact summary index of active interview state.
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


AGY_DENY_ALL_SETTINGS = json.dumps(
    {
        "permissions": {
            "deny": [
                "command(*)",
                "write_file(*)",
                "read_file(*)",
                "read_url(*)",
                "execute_url(*)",
                "mcp(*)",
            ]
        }
    },
    separators=(",", ":"),
)

_GEMINI_DENY_ALL_POLICY_TOML = (
    '[[rule]]\n'
    'name = "Deny all tools in Mockery"\n'
    'toolName = "*"\n'
    'decision = "deny"\n'
    'priority = 999\n'
    'denyMessage = "Tools are disabled in Mockery."\n'
)

_WARM_SPARE_SYSTEM_PROMPT = (
    "You are a stateless single-turn engine. Follow the instructions inside <INSTRUCTIONS> "
    "strictly and reply with ONLY the requested output format."
)
_WARM_SPARE_TTL_SEC = 600.0
_CLI_SESSION_TTL_SEC = 900.0
_MAX_CLI_SESSION_WORKERS = 3
_STDERR_TAIL_LIMIT = 4096
_HINT_HISTORY_MAX_CHARS = 4500


class _WarmSpareFallbackError(RuntimeError):
    """Raised when a pre-spawned warm CLI worker dies before producing output; triggers fallback/rehydration."""


@dataclass
class _CLISessionWorker:
    """
    Unified hardened CLI `stream-json` worker for `agy` and `claude`.
    - `max_turns=1`: single-use disposable spare (Live Copilot `D` / one-off calls).
    - `max_turns=None`: persistent session worker (Practice Studio multi-turn prompt caching).
    """

    backend: str  # "agy" | "claude"
    model_key: tuple[str, str | None]
    proc: asyncio.subprocess.Process
    created_at: float
    last_used: float
    max_turns: int | None = None
    system_prompt: str | None = None
    prefix_digest: str | None = None
    turn_count: int = 0
    stderr_buf: bytearray = field(default_factory=bytearray)
    stderr_task: asyncio.Task[None] | None = None
    in_use: bool = False


def _cli_sandbox_cwd() -> Path:
    """
    Return an empty private working directory for one-shot CLI LLM calls so agentic CLIs
    (`agy`, `claude`, `codex`, `gemini`, `pi`) do not scan repo files or inject AGENTS.md/CLAUDE.md.
    """
    sandbox = Path(tempfile.gettempdir()) / f"mockery_cli_sandbox_{os.getuid()}"
    sandbox.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(sandbox, 0o700)
    except OSError:
        pass
    return sandbox


_CODEX_ISOLATED_CONFIG_TOML = (
    "[mcp_servers]\n\n"
    "[features]\n"
    "shell_tool = false\n"
    "unified_exec = false\n"
    "apps = false\n"
    "browser_use = false\n"
    "plugins = false\n"
)


def _write_private_file_atomic(target: Path, content: str) -> None:
    """Atomically write a private 0600 file; fails closed on OS errors."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        try:
            if target.read_text(encoding="utf-8") == content:
                os.chmod(target, 0o600)
                return
        except OSError:
            pass
    tmp = target.parent / f".{target.name}.{secrets.token_hex(6)}.tmp"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
        os.replace(str(tmp), str(target))
    except BaseException:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise


def _gemini_deny_policy_path() -> Path:
    """Materialize a private 0600 deny-all Gemini CLI Policy Engine TOML (fails closed if unwritable)."""
    policy_path = _cli_sandbox_cwd() / "gemini_deny_all_tools.toml"
    _write_private_file_atomic(policy_path, _GEMINI_DENY_ALL_POLICY_TOML)
    return policy_path


def _sanitize_gemini_cli_prompt(text: str) -> str:
    """Neutralize leading `@path` attachment tokens so Gemini CLI's atCommandProcessor cannot read local files."""
    return re.sub(r"(?<!\S)@(?=\S)", "@ ", text)


def _codex_isolated_env() -> dict[str, str]:
    """
    Build an isolated CODEX_HOME containing a locked-down config.toml (zero inherited MCP servers/plugins)
    while preserving auth.json via read-only symlink so Codex subscription/OAuth keeps working.
    """
    iso_home = _cli_sandbox_cwd() / "codex_home"
    iso_home.mkdir(parents=True, exist_ok=True)
    os.chmod(iso_home, 0o700)
    _write_private_file_atomic(iso_home / "config.toml", _CODEX_ISOLATED_CONFIG_TOML)

    real_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    real_auth = real_home / "auth.json"
    iso_auth = iso_home / "auth.json"
    if real_auth.is_file():
        try:
            if iso_auth.is_symlink():
                if iso_auth.resolve() != real_auth.resolve():
                    iso_auth.unlink()
                    iso_auth.symlink_to(real_auth)
            elif not iso_auth.exists():
                iso_auth.symlink_to(real_auth)
        except OSError:
            pass

    env = dict(os.environ)
    env["CODEX_HOME"] = str(iso_home)
    return env


def _build_cached_user_blocks(user: str, cache_prefix_len: int) -> str | list[dict[str, Any]]:
    """Split a prompt into a cached stable prefix block + uncached mutable suffix block."""
    if 0 < cache_prefix_len < len(user):
        prefix = user[:cache_prefix_len]
        suffix = user[cache_prefix_len:]
        if prefix.strip() and suffix.strip():
            return [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": suffix},
            ]
    return user


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if proc.stdin is not None and not proc.stdin.is_closing():
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.5)
    except (TimeoutError, asyncio.TimeoutError):
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.5)
        except Exception:
            pass


async def _drain_stderr(stream: Any, buf: bytearray) -> None:
    """Continuously drain a worker's stderr into a bounded tail buffer so multi-turn processes never deadlock."""
    try:
        while True:
            try:
                chunk = await stream.read(1024)
            except TypeError:
                chunk = await stream.read()
                if chunk:
                    buf.extend(chunk)
                    if len(buf) > _STDERR_TAIL_LIMIT:
                        del buf[:-_STDERR_TAIL_LIMIT]
                return
            if not chunk:
                return
            buf.extend(chunk)
            if len(buf) > _STDERR_TAIL_LIMIT:
                del buf[:-_STDERR_TAIL_LIMIT]
    except asyncio.CancelledError:
        return
    except Exception:
        return


async def _close_worker(worker: _CLISessionWorker) -> None:
    await _terminate_proc(worker.proc)
    task = worker.stderr_task
    if task is not None:
        if not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
            except Exception:
                task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        worker.stderr_task = None


async def _close_workers_shielded(workers: list[_CLISessionWorker]) -> None:
    """Close every worker to completion even if the caller is cancelled mid-cleanup."""
    if not workers:
        return
    cancelled: asyncio.CancelledError | None = None
    for w in workers:
        while True:
            try:
                await asyncio.shield(_close_worker(w))
                break
            except asyncio.CancelledError as exc:
                cancelled = exc
                # Keep draining the close loop so no detached worker is ever stranded
                try:
                    await _terminate_proc(w.proc)
                except BaseException:
                    pass
                if w.stderr_task is not None and not w.stderr_task.done():
                    w.stderr_task.cancel()
                break
            except Exception:
                break
    if cancelled is not None:
        raise cancelled


class InterviewCopilotLLM:
    def __init__(self, config: LLMRuntimeConfig) -> None:
        self.config = config
        self._cached_client: AsyncOpenAI | None = None
        self._cached_client_key: tuple[str, str, int] | None = None
        self._anthropic_http: httpx.AsyncClient | None = None
        self._anthropic_http_loop: asyncio.AbstractEventLoop | None = None
        self._warm_spare: _CLISessionWorker | None = None
        self._warm_spare_reaper: asyncio.Task[Any] | None = None
        self._warm_spare_prewarm_tasks: set[asyncio.Task[Any]] = set()
        self._session_workers: OrderedDict[str, _CLISessionWorker] = OrderedDict()
        self._warm_spare_lock = asyncio.Lock()
        self._warm_spare_tasks: set[asyncio.Task[Any]] = set()
        self._session_prewarm_tasks: dict[str, asyncio.Task[Any]] = {}
        self._closing_tasks: set[asyncio.Task[Any]] = set()
        self._worker_generation: int = 0
        self._warm_spare_generation: int = 0
        self.recent_usage_events: deque[dict[str, Any]] = deque(maxlen=50)

    def _client(self) -> AsyncOpenAI:
        key = self.config.api_key.strip() or "ollama-local"
        base_url = self.config.base_url
        default_retries = 0 if self.config.is_local_http else 1
        cache_key = (key, base_url, default_retries)
        if self._cached_client is None or self._cached_client_key != cache_key:
            self._cached_client = AsyncOpenAI(
                api_key=key,
                base_url=base_url,
                timeout=60.0,
                max_retries=default_retries,
            )
            self._cached_client_key = cache_key
        return self._cached_client

    def _anthropic_client(self) -> httpx.AsyncClient:
        try:
            cur_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            cur_loop = None
        stale_loop = (
            self._anthropic_http_loop is not None
            and (self._anthropic_http_loop is not cur_loop or self._anthropic_http_loop.is_closed())
        )
        if self._anthropic_http is None or getattr(self._anthropic_http, "is_closed", False) or stale_loop:
            self._anthropic_http = httpx.AsyncClient(timeout=60.0)
            self._anthropic_http_loop = cur_loop
        return self._anthropic_http

    def _record_usage(
        self,
        *,
        operation: str,
        transport: str,
        model: str,
        mode: str,
        elapsed_ms: int,
        input_chars: int,
        output_chars: int,
        usage: dict[str, int | None] | None,
        ok: bool,
    ) -> None:
        u = usage or {}
        event: dict[str, Any] = {
            "op": operation,
            "transport": transport,
            "model": model,
            "mode": mode,
            "elapsed_ms": elapsed_ms,
            "input_chars": input_chars,
            "output_chars": output_chars,
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read_tokens": u.get("cache_read_tokens"),
            "cache_write_tokens": u.get("cache_write_tokens"),
            "ok": ok,
        }
        self.recent_usage_events.append(event)
        logger.info("llm_call %s", json.dumps(event, separators=(",", ":")))

    async def aclose(self) -> None:
        await self.close_all_workers()
        if self._cached_client is not None:
            client = self._cached_client
            self._cached_client = None
            self._cached_client_key = None
            try:
                await client.close()
            except Exception:
                pass
        if self._anthropic_http is not None:
            http_client = self._anthropic_http
            http_loop = self._anthropic_http_loop
            self._anthropic_http = None
            self._anthropic_http_loop = None
            try:
                cur_loop = asyncio.get_running_loop()
            except RuntimeError:
                cur_loop = None
            if http_loop is None or (http_loop is cur_loop and not http_loop.is_closed()):
                try:
                    await http_client.aclose()
                except Exception:
                    pass

    @staticmethod
    def _cli_model_key(backend: str, model: str) -> tuple[str, str | None]:
        cleaned = re.sub(rf"^{re.escape(backend)}/", "", model.strip())
        return _parse_agy_model_spec(cleaned)

    @staticmethod
    def _build_agy_warm_args(model_name: str, effort: str | None) -> list[str]:
        args = [
            "agy",
            "--model",
            model_name,
            "--disable-slash-commands",
            "--sandbox",
            "--settings",
            AGY_DENY_ALL_SETTINGS,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
        ]
        if effort:
            args.extend(["--effort", effort])
        return args

    @staticmethod
    def _build_claude_warm_args(
        model_name: str,
        effort: str | None,
        *,
        system_prompt: str | None = None,
    ) -> list[str]:
        args = [
            "claude",
            "-p",
            "--no-session-persistence",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--system-prompt",
            system_prompt if system_prompt is not None else _WARM_SPARE_SYSTEM_PROMPT,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        if model_name:
            args.extend(["--model", model_name])
        if effort:
            args.extend(["--effort", effort])
        return args

    async def _spawn_cli_worker(
        self,
        backend: str,
        model_key: tuple[str, str | None],
        *,
        max_turns: int | None,
        system_prompt: str | None = None,
    ) -> _CLISessionWorker:
        model_name, effort = model_key
        args = (
            self._build_agy_warm_args(model_name, effort)
            if backend == "agy"
            else self._build_claude_warm_args(model_name, effort, system_prompt=system_prompt)
        )
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(_cli_sandbox_cwd()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        now = time.monotonic()
        worker = _CLISessionWorker(
            backend=backend,
            model_key=model_key,
            proc=proc,
            created_at=now,
            last_used=now,
            max_turns=max_turns,
            system_prompt=system_prompt if backend == "claude" else None,
        )
        if proc.stderr is not None:
            worker.stderr_task = asyncio.create_task(_drain_stderr(proc.stderr, worker.stderr_buf))
        return worker

    def _spawn_tracked_close(self, workers: list[_CLISessionWorker]) -> None:
        if not workers:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(_close_workers_shielded(list(workers)))
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    def prewarm_fast_model(self, *, expected_spare_gen: int | None = None) -> None:
        """
        Best-effort pre-spawn of a single disposable warm CLI spare (`agy` or `claude`)
        so the next Live Copilot `D` press skips 2–7s of CLI boot.
        """
        backend = self.config.cli_backend
        if backend not in ("agy", "claude") or not self.config.is_configured:
            return
        model = (self.config.fast_model or "").strip()
        if not model:
            return
        if expected_spare_gen is not None and expected_spare_gen != self._warm_spare_generation:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        spare_gen = self._warm_spare_generation
        task = loop.create_task(self._ensure_warm_spare(backend, model, spare_gen=spare_gen))
        self._warm_spare_tasks.add(task)
        self._warm_spare_prewarm_tasks.add(task)

        def _done(t: asyncio.Task[Any]) -> None:
            self._warm_spare_tasks.discard(t)
            self._warm_spare_prewarm_tasks.discard(t)

        task.add_done_callback(_done)

    @staticmethod
    def _cli_format_rule(expect_json: bool) -> str:
        if expect_json:
            return "Do NOT call any tools or read any files. Use ONLY the input below and reply with ONLY raw valid JSON (no markdown fences)."
        return "Do NOT call any tools or read any files. Use ONLY the input below and reply with ONLY the requested response."

    def prewarm_session_worker(
        self,
        session_key: str,
        *,
        system: str | None = None,
        expect_json: bool = True,
    ) -> None:
        """
        Best-effort pre-spawn of a persistent locked-down CLI worker (`agy` or `claude`)
        bound to an active Practice session. Sends nothing until Turn 1 (zero wasted tokens).
        """
        backend = self.config.cli_backend
        if backend not in ("agy", "claude") or not self.config.is_configured:
            return
        model = (self.config.fast_model or "").strip()
        if not model:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        existing_task = self._session_prewarm_tasks.get(session_key)
        if existing_task is not None and not existing_task.done():
            return
        claude_sys = (
            f"{(system or '').strip()}\n\n{self._cli_format_rule(expect_json)}".strip()
            if backend == "claude" and system is not None
            else None
        )
        task = loop.create_task(self._ensure_session_worker(session_key, backend, model, system_prompt=claude_sys))
        self._session_prewarm_tasks[session_key] = task
        self._warm_spare_tasks.add(task)

        def _cleanup(t: asyncio.Task[Any]) -> None:
            self._warm_spare_tasks.discard(t)
            if self._session_prewarm_tasks.get(session_key) is t:
                self._session_prewarm_tasks.pop(session_key, None)

        task.add_done_callback(_cleanup)

    async def _ensure_warm_spare(self, backend: str, model: str, *, spare_gen: int | None = None) -> None:
        model_key = self._cli_model_key(backend, model)
        now = time.monotonic()
        stale: _CLISessionWorker | None = None
        async with self._warm_spare_lock:
            gen = self._worker_generation
            cur_spare_gen = self._warm_spare_generation if spare_gen is None else spare_gen
            if cur_spare_gen != self._warm_spare_generation:
                return
            cur = self._warm_spare
            if cur is not None:
                alive = cur.proc.returncode is None and (now - cur.created_at) < _WARM_SPARE_TTL_SEC
                if alive and cur.backend == backend and cur.model_key == model_key:
                    return
                stale = cur
                self._warm_spare = None
        if stale is not None:
            await _close_workers_shielded([stale])

        try:
            worker = await self._spawn_cli_worker(backend, model_key, max_turns=1)
        except Exception as exc:
            logger.debug("Warm CLI spare pre-spawn skipped for %s: %s", backend, exc)
            return

        discard: _CLISessionWorker | None = None
        try:
            async with self._warm_spare_lock:
                if (
                    self._worker_generation != gen
                    or self._warm_spare_generation != cur_spare_gen
                    or self._warm_spare is not None
                ):
                    discard = worker
                else:
                    self._warm_spare = worker
                    self._schedule_warm_spare_reaper_locked(worker)
        except BaseException:
            await _close_workers_shielded([worker])
            raise
        if discard is not None:
            await _close_workers_shielded([discard])

    def _schedule_warm_spare_reaper_locked(self, worker: _CLISessionWorker) -> None:
        old_reaper = self._warm_spare_reaper
        if old_reaper is not None and not old_reaper.done():
            old_reaper.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._warm_spare_reaper = None
            return

        async def _reap_after_ttl() -> None:
            try:
                await asyncio.sleep(_WARM_SPARE_TTL_SEC)
            except asyncio.CancelledError:
                return
            expired: _CLISessionWorker | None = None
            async with self._warm_spare_lock:
                if self._warm_spare is worker:
                    self._warm_spare = None
                    expired = worker
            if expired is not None:
                # Hand close to a tracked task so cancelling the reaper timer never strands cleanup.
                self._spawn_tracked_close([expired])

        self._warm_spare_reaper = loop.create_task(_reap_after_ttl())

    async def _ensure_session_worker(
        self,
        session_key: str,
        backend: str,
        model: str,
        *,
        system_prompt: str | None = None,
    ) -> None:
        model_key = self._cli_model_key(backend, model)
        now = time.monotonic()
        stale: list[_CLISessionWorker] = []
        async with self._warm_spare_lock:
            gen = self._worker_generation
            cur = self._session_workers.get(session_key)
            if cur is not None:
                alive = cur.proc.returncode is None and (now - cur.last_used) < _CLI_SESSION_TTL_SEC
                sys_ok = backend != "claude" or system_prompt is None or cur.system_prompt == system_prompt
                if alive and cur.backend == backend and cur.model_key == model_key and sys_ok:
                    self._session_workers.move_to_end(session_key)
                    return
                if not cur.in_use:
                    stale.append(self._session_workers.pop(session_key))
        if stale:
            await _close_workers_shielded(stale)

        try:
            worker = await self._spawn_cli_worker(
                backend,
                model_key,
                max_turns=None,
                system_prompt=system_prompt if backend == "claude" else None,
            )
        except Exception as exc:
            logger.debug("Practice CLI session pre-spawn skipped for %s: %s", backend, exc)
            return

        to_close: list[_CLISessionWorker] = []
        try:
            async with self._warm_spare_lock:
                if self._worker_generation != gen or session_key in self._session_workers:
                    to_close.append(worker)
                else:
                    self._session_workers[session_key] = worker
                    self._session_workers.move_to_end(session_key)
                    to_close.extend(self._evict_excess_session_workers_locked())
        except BaseException:
            await _close_workers_shielded([worker])
            raise
        if to_close:
            await _close_workers_shielded(to_close)

    def _evict_excess_session_workers_locked(self) -> list[_CLISessionWorker]:
        now = time.monotonic()
        evicted: list[_CLISessionWorker] = []
        for k in list(self._session_workers.keys()):
            w = self._session_workers[k]
            if w.in_use:
                continue
            if w.proc.returncode is not None or (now - w.last_used) >= _CLI_SESSION_TTL_SEC:
                evicted.append(self._session_workers.pop(k))
        while len(self._session_workers) > _MAX_CLI_SESSION_WORKERS:
            victim_key = next((k for k, w in self._session_workers.items() if not w.in_use), None)
            if victim_key is None:
                break
            evicted.append(self._session_workers.pop(victim_key))
        return evicted

    async def _pop_warm_spare(self, backend: str, model: str) -> _CLISessionWorker | None:
        model_key = self._cli_model_key(backend, model)
        now = time.monotonic()
        stale: _CLISessionWorker | None = None
        async with self._warm_spare_lock:
            cur = self._warm_spare
            if cur is None:
                return None
            alive = cur.proc.returncode is None and (now - cur.created_at) < _WARM_SPARE_TTL_SEC
            if not alive:
                stale = cur
                self._warm_spare = None
                if self._warm_spare_reaper is not None and not self._warm_spare_reaper.done():
                    self._warm_spare_reaper.cancel()
                self._warm_spare_reaper = None
                cur = None
            elif cur.backend == backend and cur.model_key == model_key:
                self._warm_spare = None
                if self._warm_spare_reaper is not None and not self._warm_spare_reaper.done():
                    self._warm_spare_reaper.cancel()
                self._warm_spare_reaper = None
                return cur
            else:
                return None
        if stale is not None:
            await _close_workers_shielded([stale])
        return None

    async def release_warm_spare(self) -> None:
        """Close only the disposable warm spare & cancel pending spare prewarms (leaving Practice workers intact)."""
        to_close: list[_CLISessionWorker] = []
        async with self._warm_spare_lock:
            self._warm_spare_generation += 1
            if self._warm_spare is not None:
                to_close.append(self._warm_spare)
                self._warm_spare = None
            tasks = list(self._warm_spare_prewarm_tasks)
            self._warm_spare_prewarm_tasks.clear()
            if self._warm_spare_reaper is not None:
                tasks.append(self._warm_spare_reaper)
                self._warm_spare_reaper = None
        try:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            async with self._warm_spare_lock:
                if self._warm_spare is not None:
                    to_close.append(self._warm_spare)
                    self._warm_spare = None
        finally:
            await _close_workers_shielded(to_close)

    async def close_session_worker(self, session_key: str) -> None:
        task = self._session_prewarm_tasks.pop(session_key, None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        stale: _CLISessionWorker | None = None
        async with self._warm_spare_lock:
            stale = self._session_workers.pop(session_key, None)
        if stale is not None:
            await _close_workers_shielded([stale])

    async def close_all_workers(self) -> None:
        to_close: list[_CLISessionWorker] = []
        async with self._warm_spare_lock:
            self._worker_generation += 1
            self._warm_spare_generation += 1
            tasks = list(self._warm_spare_tasks)
            if self._warm_spare_reaper is not None:
                tasks.append(self._warm_spare_reaper)
                self._warm_spare_reaper = None
            self._warm_spare_tasks.clear()
            self._warm_spare_prewarm_tasks.clear()
            self._session_prewarm_tasks.clear()
            if self._warm_spare is not None:
                to_close.append(self._warm_spare)
                self._warm_spare = None
            while self._session_workers:
                _, w = self._session_workers.popitem(last=False)
                to_close.append(w)

        try:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            async with self._warm_spare_lock:
                if self._warm_spare is not None:
                    to_close.append(self._warm_spare)
                    self._warm_spare = None
                while self._session_workers:
                    _, w = self._session_workers.popitem(last=False)
                    to_close.append(w)
        finally:
            await _close_workers_shielded(to_close)
            closing = list(self._closing_tasks)
            if closing:
                await asyncio.gather(*closing, return_exceptions=True)

    async def invalidate_warm_spare(self) -> None:
        await self.close_all_workers()

    async def _exchange_cli_worker_turn(
        self,
        worker: _CLISessionWorker,
        *,
        prompt_text: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None,
        usage_out: dict[str, int | None] | None = None,
    ) -> str:
        """
        Send one user message to a locked-down `agy` or `claude` `stream-json` worker and read
        back its response. Terminates the worker automatically when `worker.max_turns == 1`;
        keeps it alive for subsequent cached turns when `worker.max_turns is None`.
        """
        proc = worker.proc
        assert proc.stdin is not None and proc.stdout is not None
        envelope = (
            {"event": "user", "message": {"role": "user", "content": prompt_text}}
            if worker.backend == "agy"
            else {"type": "user", "message": {"role": "user", "content": prompt_text}}
        )
        wire_line = (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")
        deadline = time.monotonic() + timeout
        streamed_chunks: list[str] = []
        final_text: str | None = None
        use_deltas = on_delta is not None and not expect_json
        single_use = worker.max_turns == 1

        try:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                proc.stdin.write(wire_line)
                await asyncio.wait_for(proc.stdin.drain(), timeout=remaining)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                raise _WarmSpareFallbackError(f"{worker.backend} CLI worker stdin failed: {exc}") from exc
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
                if worker.backend == "agy":
                    ev_type = ev.get("event")
                    if ev_type == "step_update":
                        su = ev.get("step_update") or {}
                        if su.get("step_type") == "agent_response":
                            delta = su.get("text_delta")
                            if isinstance(delta, str) and delta:
                                streamed_chunks.append(delta)
                                if use_deltas and on_delta is not None:
                                    await on_delta(delta)
                    elif ev_type == "result":
                        res_obj = ev.get("result")
                        if isinstance(res_obj, dict):
                            status = res_obj.get("status")
                            if status and status != "SUCCESS":
                                raise RuntimeError(f"agy status={status}: {res_obj.get('error', '')}")
                            usage = res_obj.get("usage") or res_obj.get("stats")
                            if isinstance(usage, dict):
                                logger.debug("agy session turn usage: %s", usage)
                                if usage_out is not None:
                                    usage_out.update(_extract_usage_counts(usage))
                            final_text = str(res_obj.get("response", "")).strip()
                        break
                else:
                    ev_type = ev.get("type")
                    if ev_type == "stream_event":
                        inner = ev.get("event") or {}
                        if inner.get("type") == "content_block_delta":
                            delta_obj = inner.get("delta") or {}
                            txt = delta_obj.get("text")
                            if isinstance(txt, str) and txt:
                                streamed_chunks.append(txt)
                                if use_deltas and on_delta is not None:
                                    await on_delta(txt)
                    elif ev_type == "result":
                        if ev.get("is_error"):
                            raise RuntimeError(f"claude CLI error: {str(ev.get('result', ''))[:300]}")
                        usage = ev.get("usage")
                        if isinstance(usage, dict):
                            logger.debug("claude session turn usage: %s", usage)
                            if usage_out is not None:
                                usage_out.update(_extract_usage_counts(usage))
                        final_text = str(ev.get("result", "")).strip()
                        break
        except asyncio.TimeoutError:
            await _close_workers_shielded([worker])
            raise TimeoutError(f"{worker.backend} CLI worker timed out after {timeout:.0f}s")
        except BaseException:
            await _close_workers_shielded([worker])
            raise
        finally:
            if single_use:
                await _close_workers_shielded([worker])

        if final_text is not None and final_text:
            return final_text
        joined = "".join(streamed_chunks).strip()
        if joined:
            return joined
        stderr = worker.stderr_buf.decode("utf-8", errors="replace").strip()
        await _close_workers_shielded([worker])
        raise _WarmSpareFallbackError(f"{worker.backend} CLI worker returned empty response: {stderr[:300]}")

    async def _call_persistent_cli_session(
        self,
        *,
        backend: str,
        session_key: str,
        model: str,
        messages: list[dict[str, str]],
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        meta_out: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute one turn on a session-scoped persistent `agy` or `claude` worker.
        Sends ONLY the latest user message when the worker's `prefix_digest` matches `messages[:-1]`,
        achieving full provider-side prompt caching (~81%+ cache read on `agy`) and zero CLI boot.
        On cold start, crash, eviction, or digest mismatch, spawns a fresh worker and sends ONE
        flattened bootstrap prompt (`render_flat_messages`), then resumes incremental turns.
        """
        model_key = self._cli_model_key(backend, model)
        sys_str = "\n\n".join(
            str(m.get("content", "")).strip() for m in messages if m.get("role") == "system" and m.get("content")
        )
        format_rule = self._cli_format_rule(expect_json)
        claude_sys = f"{sys_str}\n\n{format_rule}".strip() if backend == "claude" else None
        prior_digest = messages_prefix_digest(messages[:-1])

        # Wait briefly if a prewarm task for this exact session_key is already in flight
        prewarm = self._session_prewarm_tasks.get(session_key)
        if prewarm is not None and not prewarm.done():
            try:
                await asyncio.wait_for(asyncio.shield(prewarm), timeout=5.0)
            except Exception:
                pass

        for attempt in range(2):
            now = time.monotonic()
            worker: _CLISessionWorker | None = None
            stale: list[_CLISessionWorker] = []
            send_delta_only = False
            registered = False

            async with self._warm_spare_lock:
                gen = self._worker_generation
                cur = self._session_workers.get(session_key)
                if cur is not None:
                    alive = cur.proc.returncode is None and (now - cur.last_used) < _CLI_SESSION_TTL_SEC
                    model_ok = cur.backend == backend and cur.model_key == model_key
                    sys_ok = backend != "claude" or cur.system_prompt == claude_sys
                    if alive and model_ok and sys_ok and not cur.in_use:
                        if cur.turn_count == 0:
                            worker = cur
                            worker.in_use = True
                            registered = True
                            self._session_workers.move_to_end(session_key)
                        elif cur.prefix_digest == prior_digest:
                            worker = cur
                            worker.in_use = True
                            registered = True
                            send_delta_only = True
                            self._session_workers.move_to_end(session_key)
                        else:
                            stale.append(self._session_workers.pop(session_key))
                    elif not cur.in_use:
                        stale.append(self._session_workers.pop(session_key))

            if stale:
                await _close_workers_shielded(stale)

            if worker is None:
                worker = await self._spawn_cli_worker(
                    backend,
                    model_key,
                    max_turns=None,
                    system_prompt=claude_sys,
                )
                worker.in_use = True
                evicted: list[_CLISessionWorker] = []
                try:
                    async with self._warm_spare_lock:
                        if self._worker_generation == gen:
                            prev = self._session_workers.get(session_key)
                            if prev is None:
                                self._session_workers[session_key] = worker
                                self._session_workers.move_to_end(session_key)
                                registered = True
                            elif not prev.in_use:
                                evicted.append(self._session_workers.pop(session_key))
                                self._session_workers[session_key] = worker
                                self._session_workers.move_to_end(session_key)
                                registered = True
                            evicted.extend(self._evict_excess_session_workers_locked())
                except BaseException:
                    await _close_workers_shielded([worker])
                    raise
                if evicted:
                    await _close_workers_shielded(evicted)

            if send_delta_only:
                prompt_text = str(messages[-1].get("content", ""))
            else:
                prompt_text = render_flat_messages(
                    messages,
                    include_system=(backend == "agy"),
                    format_rule=format_rule if backend == "agy" else "",
                )

            try:
                usage_counts: dict[str, int | None] = {}
                result = await self._exchange_cli_worker_turn(
                    worker,
                    prompt_text=prompt_text,
                    timeout=timeout,
                    expect_json=expect_json,
                    on_delta=on_delta,
                    usage_out=usage_counts,
                )
                if meta_out is not None:
                    meta_out["mode"] = "delta" if send_delta_only else "bootstrap"
                    meta_out["usage"] = usage_counts
            except _WarmSpareFallbackError as exc:
                invalidated = False
                async with self._warm_spare_lock:
                    if self._session_workers.get(session_key) is worker:
                        self._session_workers.pop(session_key, None)
                    invalidated = self._worker_generation != gen
                worker.in_use = False
                if attempt == 0 and not invalidated:
                    logger.debug(
                        "%s persistent session worker died before output (%s); rehydrating once",
                        backend,
                        exc,
                    )
                    continue
                raise RuntimeError(str(exc)) from exc
            except BaseException:
                async with self._warm_spare_lock:
                    if self._session_workers.get(session_key) is worker:
                        self._session_workers.pop(session_key, None)
                worker.in_use = False
                await _close_workers_shielded([worker])
                raise

            to_close_after: list[_CLISessionWorker] = []
            async with self._warm_spare_lock:
                worker.turn_count += 1
                worker.last_used = time.monotonic()
                worker.prefix_digest = messages_prefix_digest(
                    messages + [{"role": "assistant", "content": result}]
                )
                worker.in_use = False
                if not registered or self._worker_generation != gen or self._session_workers.get(session_key) is not worker:
                    if self._session_workers.get(session_key) is worker:
                        self._session_workers.pop(session_key, None)
                    to_close_after.append(worker)
                else:
                    self._session_workers.move_to_end(session_key)
                to_close_after.extend(self._evict_excess_session_workers_locked())
            if to_close_after:
                await _close_workers_shielded(to_close_after)
            return result

        raise RuntimeError(f"{backend} persistent session failed after recovery attempt")

    async def _call_agy_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        use_warm_spare: bool = False,
        meta_out: dict[str, Any] | None = None,
    ) -> str:
        model_name, effort = _parse_agy_model_spec(model)
        format_rule = self._cli_format_rule(expect_json)
        combined = f"<INSTRUCTIONS>\n{system}\n</INSTRUCTIONS>\n\n{format_rule}\n\n<INPUT>\n{user}\n</INPUT>"
        is_fast_model = self._cli_model_key("agy", model) == self._cli_model_key("agy", self.config.fast_model)
        spare_gen = self._warm_spare_generation

        spare = await self._pop_warm_spare("agy", model) if use_warm_spare else None
        if spare is not None:
            try:
                usage_counts: dict[str, int | None] = {}
                result = await self._exchange_cli_worker_turn(
                    spare,
                    prompt_text=combined,
                    timeout=timeout,
                    expect_json=expect_json,
                    on_delta=on_delta,
                    usage_out=usage_counts,
                )
                if meta_out is not None:
                    meta_out["mode"] = "warm"
                    meta_out["usage"] = usage_counts
                if is_fast_model:
                    self.prewarm_fast_model(expected_spare_gen=spare_gen)
                return result
            except _WarmSpareFallbackError as exc:
                logger.debug("agy warm spare fallback to one-shot: %s", exc)

        if meta_out is not None:
            meta_out["mode"] = "cold"

        use_stream = on_delta is not None and not expect_json
        args = [
            "agy",
            "--model",
            model_name,
            "--disable-slash-commands",
            "--sandbox",
            "--settings",
            AGY_DENY_ALL_SETTINGS,
            "--output-format",
            "stream-json" if use_stream else "json",
        ]
        if effort:
            args.extend(["--effort", effort])
        args.append(f"-p={combined}")

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(_cli_sandbox_cwd()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if not use_stream:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await _terminate_proc(proc)
                raise TimeoutError(f"agy CLI timed out after {timeout:.0f}s")
            except BaseException:
                await _terminate_proc(proc)
                raise

            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"agy CLI exited {proc.returncode}: {(stderr or stdout)[:300]}")

            envelope = _extract_json_object(stdout)
            status = envelope.get("status")
            if status and status != "SUCCESS":
                raise RuntimeError(f"agy status={status}: {envelope.get('error', '')}")
            if meta_out is not None:
                raw_usage = envelope.get("usage") or envelope.get("stats")
                if isinstance(raw_usage, dict):
                    meta_out["usage"] = _extract_usage_counts(raw_usage)
            if use_warm_spare and is_fast_model:
                self.prewarm_fast_model(expected_spare_gen=spare_gen)
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
            await _terminate_proc(proc)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise TimeoutError(f"agy CLI timed out after {timeout:.0f}s")
        except BaseException:
            await _terminate_proc(proc)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise

        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        if final_result is not None:
            status = final_result.get("status")
            if status and status != "SUCCESS":
                raise RuntimeError(f"agy status={status}: {final_result.get('error', '')}")
            if meta_out is not None:
                raw_usage = final_result.get("usage") or final_result.get("stats")
                if isinstance(raw_usage, dict):
                    meta_out["usage"] = _extract_usage_counts(raw_usage)
            resp_text = str(final_result.get("response", "")).strip()
            if resp_text:
                if use_warm_spare and is_fast_model:
                    self.prewarm_fast_model(expected_spare_gen=spare_gen)
                return resp_text
        if proc.returncode != 0:
            raise RuntimeError(f"agy CLI exited {proc.returncode}: {stderr[:300]}")
        if use_warm_spare and is_fast_model:
            self.prewarm_fast_model(expected_spare_gen=spare_gen)
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
            "--no-skills",
            "--no-prompt-templates",
            "--model",
            clean_model,
            "--system-prompt",
            system,
            "--",
            user,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(_cli_sandbox_cwd()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await self._communicate_or_stream_cli(
            proc,
            name="pi CLI",
            stdin_text=None,
            timeout=timeout,
            on_delta=on_delta,
        )

    async def _call_claude_cli(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: float,
        expect_json: bool,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        use_warm_spare: bool = False,
        meta_out: dict[str, Any] | None = None,
    ) -> str:
        model_name, effort = _parse_agy_model_spec(re.sub(r"^claude/", "", model.strip()))
        format_rule = (
            "Reply with ONLY raw valid JSON (no markdown fences)."
            if expect_json
            else "Reply with ONLY the requested Markdown response."
        )
        sys_prompt = f"{system.strip()}\n\n{format_rule}".strip()
        is_fast_model = self._cli_model_key("claude", model) == self._cli_model_key("claude", self.config.fast_model)
        spare_gen = self._warm_spare_generation

        spare = await self._pop_warm_spare("claude", model) if use_warm_spare else None
        if spare is not None:
            combined = f"<INSTRUCTIONS>\n{sys_prompt}\n</INSTRUCTIONS>\n\n<INPUT>\n{user}\n</INPUT>"
            try:
                usage_counts: dict[str, int | None] = {}
                result = await self._exchange_cli_worker_turn(
                    spare,
                    prompt_text=combined,
                    timeout=timeout,
                    expect_json=expect_json,
                    on_delta=on_delta,
                    usage_out=usage_counts,
                )
                if meta_out is not None:
                    meta_out["mode"] = "warm"
                    meta_out["usage"] = usage_counts
                if is_fast_model:
                    self.prewarm_fast_model(expected_spare_gen=spare_gen)
                return result
            except _WarmSpareFallbackError as exc:
                logger.debug("claude warm spare fallback to one-shot: %s", exc)

        if meta_out is not None:
            meta_out["mode"] = "cold"

        args = [
            "claude",
            "-p",
            "--no-session-persistence",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
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
            cwd=str(_cli_sandbox_cwd()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        result = await self._communicate_or_stream_cli(
            proc,
            name="claude CLI",
            stdin_text=user,
            timeout=timeout,
            on_delta=on_delta,
        )
        if use_warm_spare and is_fast_model:
            self.prewarm_fast_model(expected_spare_gen=spare_gen)
        return result

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
                "--disable",
                "shell_tool",
                "--disable",
                "unified_exec",
                "--disable",
                "apps",
                "--disable",
                "browser_use",
                "-c",
                "mcp_servers={}",
                "-o",
                out_path,
            ]
            if clean_model:
                args.extend(["-m", clean_model])
            args.append("-")

            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(_cli_sandbox_cwd()),
                env=_codex_isolated_env(),
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
        combined = _sanitize_gemini_cli_prompt(
            f"<INSTRUCTIONS>\n{system}\n</INSTRUCTIONS>\n\n{format_rule}\n\n<INPUT>\n{user}\n</INPUT>"
        )
        deny_policy = str(_gemini_deny_policy_path())
        args = [
            "gemini",
            "--approval-mode",
            "plan",
            "-e",
            "none",
            "--allowed-mcp-server-names",
            "",
            "--policy",
            deny_policy,
            "--admin-policy",
            deny_policy,
            "-o",
            "text",
        ]
        if clean_model:
            args.extend(["-m", clean_model])
        args.extend(["-p", ""])

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(_cli_sandbox_cwd()),
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
        stdin_text: str | None,
        timeout: float,
        on_delta: Callable[[str], Awaitable[None]] | None,
    ) -> str:
        stdin_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
        if on_delta is None:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=timeout)
            except TimeoutError:
                await _terminate_proc(proc)
                raise TimeoutError(f"{name} timed out after {timeout:.0f}s")
            except BaseException:
                await _terminate_proc(proc)
                raise
            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"{name} exited {proc.returncode}: {(stderr or stdout)[:300]}")
            return stdout

        assert proc.stdout is not None and proc.stderr is not None
        deadline = time.monotonic() + timeout
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        chunks: list[str] = []
        stderr_task = asyncio.create_task(proc.stderr.read())
        try:
            if stdin_bytes is not None:
                assert proc.stdin is not None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                proc.stdin.write(stdin_bytes)
                await asyncio.wait_for(proc.stdin.drain(), timeout=remaining)
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
            await _terminate_proc(proc)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise TimeoutError(f"{name} timed out after {timeout:.0f}s")
        except BaseException:
            await _terminate_proc(proc)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise

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
        user: str = "",
        messages: list[dict[str, str]] | None = None,
        temperature: float,
        max_tokens: int,
        timeout: float,
        cache_prefix_len: int = 0,
        cache_key: str | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        meta_out: dict[str, Any] | None = None,
    ) -> str:
        base = self.config.base_url.rstrip("/")
        url = f"{base}/messages" if not base.endswith("/messages") else base
        headers = {
            "x-api-key": self.config.api_key.strip(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        api_messages: list[dict[str, Any]] = []
        if messages is not None:
            non_sys = [m for m in messages if m.get("role") in ("user", "assistant")]
            if len(non_sys) == 1 and non_sys[0].get("role") == "user":
                raw_u = str(non_sys[0].get("content", ""))
                if cache_prefix_len > 0:
                    u_blocks = _build_cached_user_blocks(raw_u, cache_prefix_len)
                elif cache_key and raw_u.strip():
                    u_blocks = [{"type": "text", "text": raw_u, "cache_control": {"type": "ephemeral"}}]
                else:
                    u_blocks = raw_u
                api_messages.append({"role": "user", "content": u_blocks})
            else:
                for idx, m in enumerate(non_sys):
                    txt = str(m.get("content", ""))
                    if idx == len(non_sys) - 1 and txt.strip():
                        api_messages.append(
                            {
                                "role": m["role"],
                                "content": [
                                    {"type": "text", "text": txt, "cache_control": {"type": "ephemeral"}}
                                ],
                            }
                        )
                    else:
                        api_messages.append({"role": m["role"], "content": txt})
        else:
            api_messages.append(
                {"role": "user", "content": _build_cached_user_blocks(user, cache_prefix_len)}
            )

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if _anthropic_supports_temperature(model):
            payload["temperature"] = temperature
        if system.strip():
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        http = self._anthropic_client()
        post_kwargs: dict[str, Any] = {"headers": headers, "json": payload}
        if isinstance(http, httpx.AsyncClient):
            post_kwargs["timeout"] = timeout

        if on_delta is not None:
            payload["stream"] = True
            parts: list[str] = []
            stream_usage_raw: dict[str, Any] = {}
            stop_reason: str | None = None
            async with http.stream("POST", url, **post_kwargs) as resp:
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
                    ev_type = ev.get("type")
                    if ev_type == "message_start":
                        msg_usage = (ev.get("message") or {}).get("usage")
                        if isinstance(msg_usage, dict):
                            stream_usage_raw.update(msg_usage)
                    elif ev_type == "message_delta":
                        delta_obj = ev.get("delta") or {}
                        if isinstance(delta_obj.get("stop_reason"), str):
                            stop_reason = delta_obj["stop_reason"]
                        delta_usage = ev.get("usage")
                        if isinstance(delta_usage, dict):
                            stream_usage_raw.update(delta_usage)
                    elif ev_type == "content_block_delta":
                        delta_obj = ev.get("delta") or {}
                        txt = delta_obj.get("text")
                        if isinstance(txt, str) and txt:
                            parts.append(txt)
                            await on_delta(txt)
            if meta_out is not None:
                if stream_usage_raw:
                    meta_out["usage"] = _extract_usage_counts(stream_usage_raw, provider_format="anthropic")
                if stop_reason:
                    meta_out["finish_reason"] = stop_reason
            return "".join(parts).strip()

        resp = await http.post(url, **post_kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if meta_out is not None:
            if isinstance(data.get("usage"), dict):
                meta_out["usage"] = _extract_usage_counts(data["usage"], provider_format="anthropic")
            if isinstance(data.get("stop_reason"), str):
                meta_out["finish_reason"] = data["stop_reason"]
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

    async def _complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: float = 30.0,
        expect_json: bool = False,
        disable_fast_thinking: bool = True,
        cache_prefix_len: int = 0,
        cache_key: str | None = None,
        session_key: str | None = None,
        use_warm_spare: bool = False,
        operation: str = "completion",
        max_retries: int | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Unified multi-turn completion entry point across all providers:
        - `agy` / `claude` with `session_key`: persistent locked-down `stream-json` worker (true prefix caching + 0s boot).
        - `agy` / `claude` without `session_key`: reserved disposable warm spare (`use_warm_spare=True`) or hardened one-shot.
        - `codex` / `gemini` / `pi`: flattened canonical history via `render_flat_messages`.
        - HTTP (Anthropic, OpenRouter, OpenAI, Gemini, DeepSeek, oMLX, Ollama): native multi-turn `messages`
          with sliding `cache_control` breakpoints or `prompt_cache_key`.
        """
        t0 = time.perf_counter()
        input_chars = sum(len(str(m.get("content", ""))) for m in messages)
        sys_str = "\n\n".join(
            str(m.get("content", "")).strip() for m in messages if m.get("role") == "system" and m.get("content")
        )
        non_sys = [m for m in messages if m.get("role") in ("user", "assistant")]
        cli = self.config.cli_backend
        transport = f"cli:{cli}" if cli else ("anthropic" if self.config.is_anthropic_native else self.config.provider)
        meta: dict[str, Any] = {"mode": "http" if not cli else "cold", "usage": None}
        out = ""
        ok = False

        try:
            if cli in ("agy", "claude") and session_key:
                out = await self._call_persistent_cli_session(
                    backend=cli,
                    session_key=session_key,
                    model=model,
                    messages=messages,
                    timeout=timeout,
                    expect_json=expect_json,
                    on_delta=on_delta,
                    meta_out=meta,
                )
                ok = True
                return out

            if cli is not None:
                flat_user = (
                    str(non_sys[0].get("content", ""))
                    if len(non_sys) == 1 and non_sys[0].get("role") == "user"
                    else render_flat_messages(messages, include_system=False)
                )
                if cli == "pi":
                    out = await self._call_pi_cli(
                        model=model,
                        system=sys_str,
                        user=flat_user,
                        timeout=timeout,
                        on_delta=on_delta,
                    )
                    ok = True
                    return out
                if cli == "agy":
                    out = await self._call_agy_cli(
                        model=model,
                        system=sys_str,
                        user=flat_user,
                        timeout=timeout,
                        expect_json=expect_json,
                        on_delta=on_delta,
                        use_warm_spare=use_warm_spare,
                        meta_out=meta,
                    )
                    ok = True
                    return out
                if cli == "claude":
                    out = await self._call_claude_cli(
                        model=model,
                        system=sys_str,
                        user=flat_user,
                        timeout=timeout,
                        expect_json=expect_json,
                        on_delta=on_delta,
                        use_warm_spare=use_warm_spare,
                        meta_out=meta,
                    )
                    ok = True
                    return out
                if cli == "codex":
                    out = await self._call_codex_cli(
                        model=model,
                        system=sys_str,
                        user=flat_user,
                        timeout=timeout,
                        expect_json=expect_json,
                        on_delta=on_delta,
                    )
                    ok = True
                    return out
                if cli == "gemini":
                    out = await self._call_gemini_cli(
                        model=model,
                        system=sys_str,
                        user=flat_user,
                        timeout=timeout,
                        expect_json=expect_json,
                        on_delta=on_delta,
                    )
                    ok = True
                    return out

            if self.config.is_anthropic_native:
                out = await self._call_anthropic_api(
                    model=model,
                    system=sys_str,
                    messages=non_sys,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    cache_prefix_len=cache_prefix_len,
                    cache_key=cache_key,
                    on_delta=on_delta,
                    meta_out=meta,
                )
                ok = True
                return out

            client = self._client()
            if max_retries is not None and hasattr(client, "with_options"):
                client = client.with_options(max_retries=max_retries)
            is_reasoning = _is_reasoning_model(model)
            is_openrouter_claude = self.config.provider == "openrouter" and (
                "anthropic/" in model.lower() or "claude" in model.lower()
            )
            wire_messages: list[dict[str, Any]] = []
            if sys_str:
                if is_openrouter_claude:
                    wire_messages.append(
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": sys_str, "cache_control": {"type": "ephemeral"}}
                            ],
                        }
                    )
                else:
                    wire_messages.append({"role": "system", "content": sys_str})

            if len(non_sys) == 1 and non_sys[0].get("role") == "user":
                raw_u = str(non_sys[0].get("content", ""))
                if is_openrouter_claude:
                    if cache_prefix_len > 0:
                        user_content: str | list[dict[str, Any]] = _build_cached_user_blocks(raw_u, cache_prefix_len)
                    elif cache_key and raw_u.strip():
                        user_content = [{"type": "text", "text": raw_u, "cache_control": {"type": "ephemeral"}}]
                    else:
                        user_content = raw_u
                else:
                    user_content = raw_u
                wire_messages.append({"role": "user", "content": user_content})
            else:
                for idx, m in enumerate(non_sys):
                    txt = str(m.get("content", ""))
                    if is_openrouter_claude and idx == len(non_sys) - 1 and txt.strip():
                        wire_messages.append(
                            {
                                "role": m["role"],
                                "content": [
                                    {"type": "text", "text": txt, "cache_control": {"type": "ephemeral"}}
                                ],
                            }
                        )
                    else:
                        wire_messages.append({"role": m["role"], "content": txt})

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": wire_messages,
                "max_tokens": max_tokens,
            }
            if not is_reasoning and (not is_openrouter_claude or _anthropic_supports_temperature(model)):
                kwargs["temperature"] = temperature
            extra_body: dict[str, Any] = {}
            if disable_fast_thinking:
                extra = _fast_extra_body(model)
                if extra:
                    extra_body.update(extra)
            is_openai_direct = (
                self.config.provider == "openai" or "api.openai.com" in self.config.base_url.lower()
            )
            if is_openai_direct and cache_key:
                extra_body["prompt_cache_key"] = cache_key[:64]
            if extra_body:
                kwargs["extra_body"] = extra_body

            if on_delta is not None:
                stream_kwargs = dict(kwargs)
                if is_openai_direct or self.config.provider == "openrouter":
                    stream_kwargs["stream_options"] = {"include_usage": True}
                stream = await client.chat.completions.create(**stream_kwargs, stream=True, timeout=timeout)
                parts: list[str] = []
                async for chunk in stream:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        meta["usage"] = _extract_usage_counts(chunk_usage)
                    if not chunk.choices:
                        continue
                    fr = getattr(chunk.choices[0], "finish_reason", None)
                    if isinstance(fr, str) and fr:
                        meta["finish_reason"] = fr
                    delta = chunk.choices[0].delta.content
                    if delta:
                        parts.append(delta)
                        await on_delta(delta)
                out = "".join(parts).strip()
                ok = True
                return out

            resp = await client.chat.completions.create(**kwargs, timeout=timeout)
            resp_usage = getattr(resp, "usage", None)
            if resp_usage is not None:
                meta["usage"] = _extract_usage_counts(resp_usage)
            fr = getattr(resp.choices[0], "finish_reason", None) if resp.choices else None
            if isinstance(fr, str) and fr:
                meta["finish_reason"] = fr
            out = (resp.choices[0].message.content or "").strip()
            ok = True
            return out
        finally:
            self._last_finish_reason: str | None = meta.get("finish_reason")
            self._record_usage(
                operation=operation,
                transport=transport,
                model=model,
                mode=str(meta.get("mode") or "unknown"),
                elapsed_ms=round((time.perf_counter() - t0) * 1000),
                input_chars=input_chars,
                output_chars=len(out),
                usage=meta.get("usage"),
                ok=ok,
            )

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
        cache_prefix_len: int = 0,
        cache_key: str | None = None,
        use_warm_spare: bool = False,
        operation: str = "completion",
        max_retries: int | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        msgs: list[dict[str, str]] = []
        if system.strip():
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return await self._complete_messages(
            msgs,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            expect_json=expect_json,
            disable_fast_thinking=disable_fast_thinking,
            cache_prefix_len=cache_prefix_len,
            cache_key=cache_key,
            use_warm_spare=use_warm_spare,
            operation=operation,
            max_retries=max_retries,
            on_delta=on_delta,
        )

    async def suggest_probes(self, session: InterviewSession, qa: QAGroup) -> DiveDeeperResponse:
        # Include live in-progress (`final=False`) utterances so Dive Deeper works mid-answer!
        utts = session.utterances_for_card(qa, include_drafts=True)
        rev_sum = sum(u.rev for u in utts)
        if not self.config.is_configured:
            return self._heuristic_fallback_probes(session, qa, len(utts), rev_sum)

        # Keep finalized turns in an append-only prefix so KV/prefix caches hit across turns on this card;
        # live mutating drafts and prior-suggestion avoid lists sit strictly in the suffix.
        final_lines = [
            f"[{fmt_ms(u.start_ms)} {u.speaker.upper()}]: {u.text}"
            for u in utts
            if u.final
        ]
        draft_lines = [
            f"[{fmt_ms(u.start_ms)} {u.speaker.upper()} (speaking live)]: {u.text}"
            for u in utts
            if not u.final
        ]

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
        system_prompt = render_prompt("suggest_probes", probe_guidance=probe_guidance)

        stable_prefix = (
            f"Target Role: {session.config.role_title} | Focus: {session.config.interview_type}\n"
            f"Rubric Context: {session.config.rubric_notes or 'Standard pragmatic engineering bar'}\n\n"
            f"Active Card ({qa.badge_prefix()}{qa.index}: {qa.title}):\n"
            + ("\n".join(final_lines) if final_lines else "(Waiting for first finalized turn)")
        )
        mutable_suffix = (
            ("\n" + "\n".join(draft_lines) if draft_lines else "")
            + avoid_block
        )
        user_prompt = stable_prefix + mutable_suffix

        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system=system_prompt,
                user=user_prompt,
                temperature=0.35,
                max_tokens=420,
                timeout=35.0,
                expect_json=True,
                cache_prefix_len=len(stable_prefix) if mutable_suffix else 0,
                cache_key=f"mockery:copilot:{session.id}:{qa.id}",
                use_warm_spare=True,
                operation="suggest_probes",
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

        diarize_rules = triage_diarize_rules(diarize)
        system_prompt = render_prompt("triage_window", diarize_rules=diarize_rules)

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
                operation="triage_window",
                max_retries=0,
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
                    spk_raw = str(s.get("speaker", "")).strip().lower()
                    txt = str(s.get("text", "")).strip()
                    if spk_raw == "interviewer" and txt:
                        segs.append(TurnSegment(speaker="interviewer", text=txt))
                    elif spk_raw == "candidate" and txt:
                        segs.append(TurnSegment(speaker="candidate", text=txt))
                if not segs:
                    segs = [TurnSegment(speaker=orig.speaker, text=orig.text)]
                action_raw = str(item.get("card_action", "keep_in_current")).strip()
                card_action: Literal["keep_in_current", "open_new_card", "ignore_backchannel"]
                if action_raw == "open_new_card":
                    card_action = "open_new_card"
                elif action_raw == "ignore_backchannel":
                    card_action = "ignore_backchannel"
                else:
                    card_action = "keep_in_current"
                conf_raw = str(item.get("confidence", "high")).strip().lower()
                confidence: Literal["high", "medium", "low"]
                if conf_raw == "high":
                    confidence = "high"
                elif conf_raw == "low":
                    confidence = "low"
                else:
                    confidence = "medium"
                out.append(
                    TurnTriagePlan(
                        utterance_id=uid,
                        segments=segs,
                        card_action=card_action,
                        suggested_title=str(item.get("suggested_title", "")).strip(),
                        confidence=confidence,
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
                system=render_prompt("classify_topic_shift"),
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
                system=render_prompt("check_speaker_role_orientation"),
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
                system=render_prompt("regroup_cards_full_context"),
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
        system_prompt = render_prompt(
            "calibration_report",
            interview_type=session.config.interview_type,
            rubric_block=rubric_block,
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

    # AI Practice Interviewer & Supportive CARL Coach

    async def generate_practice_plan(
        self,
        *,
        role_title: str,
        company: str,
        interview_type: str,
        target_level: str,
        question_source: str,
        question_count: int,
        bank_candidates: list[dict[str, Any]],
        redacted_resume: str = "",
        jd_text: str = "",
        selected_question_id: str | None = None,
        custom_question: str = "",
        rubric: Rubric = CARL,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Build the interview question plan and a warm opening greeting.
        Returns `(questions_list, opening_message)`.
        Always keeps `redacted_resume` privacy-safe and degrades cleanly to curated banks if unconfigured.
        A `custom_question` opens the session exactly as the candidate wrote it.
        """
        effective_count = 1 if interview_type in ("system_design", "coding") else max(1, min(6, question_count))

        def _to_str_list(val: Any) -> list[str]:
            if isinstance(val, str):
                s = val.strip()
                return [s] if s else []
            if isinstance(val, (list, tuple)):
                out: list[str] = []
                for item in val:
                    if isinstance(item, str) and item.strip():
                        out.append(item.strip())
                    elif isinstance(item, (int, float)) and not isinstance(item, bool):
                        out.append(str(item))
                return out
            return []

        def _bank_to_item(q: dict[str, Any]) -> dict[str, Any]:
            playbook = q.get("playbook") if isinstance(q.get("playbook"), dict) else {}
            core_reqs = _to_str_list(q.get("core_requirements")) or _to_str_list(playbook.get("core_requirements"))
            deep_dives = _to_str_list(q.get("deep_dives")) or _to_str_list(playbook.get("deep_dives"))
            follow_ups = _to_str_list(q.get("follow_ups")) or _to_str_list(q.get("hints"))
            return {
                "bank_question_id": q.get("id"),
                "source": "bank",
                "category": q.get("category") or interview_type,
                "title": q.get("title") or q.get("competency") or "Interview Question",
                "competency": q.get("competency") or q.get("title") or "",
                "prompt": q.get("prompt") or "",
                "carl_focus": q.get("carl_focus") or "",
                "follow_ups": follow_ups,
                "core_requirements": core_reqs,
                "deep_dives": deep_dives,
                "starter_code": dict(q.get("starter_code")) if isinstance(q.get("starter_code"), dict) else {},
                "tests": dict(q.get("tests")) if isinstance(q.get("tests"), dict) else {},
                "difficulty": str(q.get("difficulty") or ""),
                "level_bar": _level_bar(playbook.get("level_extensions")),
            }

        def _level_bar(extensions: Any) -> str:
            if not isinstance(extensions, dict):
                return ""
            wanted = normalize_level(target_level)
            for level, bar in extensions.items():
                if normalize_level(level) == wanted:
                    return str(bar or "").strip()
            return ""

        selected_bank = [_bank_to_item(q) for q in bank_candidates[:effective_count]]
        pinned_item: dict[str, Any] | None = None
        custom_text = custom_question.strip()
        if custom_text:
            first_line = custom_text.splitlines()[0].strip()
            pinned_item = {
                **_bank_to_item({}),
                "bank_question_id": None,
                "source": "custom",
                "category": interview_type,
                "title": first_line if len(first_line) <= 64 else first_line[:63].rstrip() + "…",
                "competency": "Your question",
                "prompt": custom_text,
            }
            selected_bank = [pinned_item, *selected_bank][:effective_count]
        elif selected_question_id:
            for item in selected_bank:
                if item.get("bank_question_id") == selected_question_id:
                    pinned_item = item
                    break
        remaining_bank = [item for item in selected_bank if item is not pinned_item]

        role_label = f"{role_title} at {company}" if company else role_title
        session_for = (
            f"at the {target_level} level"
            if role_title == DEFAULT_PRACTICE_ROLE_TITLE and not company
            else f"for {role_label} ({target_level})"
        )
        default_opening = (
            f"Welcome! Today we're running a supportive {interview_type.replace('_', ' ')} practice session "
            f"{session_for}. Take a breath, think out loud whenever helpful, and let's dive right in."
        )

        # If bank_only OR LLM not configured OR system_design/coding with a specific bank problem:
        # For system_design/coding, curated playbooks/test harnesses stay primary, with tailored framing if LLM is available.
        if not self.config.is_configured or (
            question_source == "bank_only" and selected_bank
        ):
            ordered = ([pinned_item] + remaining_bank) if pinned_item else selected_bank
            return ordered[:effective_count], default_opening

        # Determine how many tailored questions to ask LLM to generate
        if interview_type in ("system_design", "coding") and selected_bank:
            # Keep the rich curated playbook/test-harness item, and ask LLM only for a tailored framing + greeting
            # when a target JD is actually present (the framing prompt does not use resume text).
            base = pinned_item or selected_bank[0]
            if not jd_text.strip() or base.get("source") == "custom":
                return [base], default_opening
            sys_p = render_prompt("practice_plan_framing")
            usr_p = (
                f"Target Role: {role_label} ({target_level}) | Format: {interview_type}\n"
                f"Base Problem Title: {base['title']}\nBase Prompt: {base['prompt']}\n\n"
                f"Target Job Listing excerpt:\n{jd_text[:3000]}\n"
            )
            try:
                raw = await self._complete_text(
                    model=self.config.fast_model,
                    system=sys_p,
                    user=usr_p,
                    temperature=0.4,
                    max_tokens=360,
                    timeout=25.0,
                    expect_json=True,
                )
                parsed = _extract_json_object(raw)
                opening = str(parsed.get("opening") or "").strip() or default_opening
                if parsed.get("tailored_prompt"):
                    base["prompt"] = str(parsed["tailored_prompt"]).strip()
                return [base], opening
            except Exception as exc:
                logger.warning("Practice framing failed (%s); using base bank item", exc)
                return [base], default_opening

        # Behavioral / General: generate tailored questions from redacted resume + target JD
        slots_after_pin = effective_count - (1 if pinned_item else 0)
        if slots_after_pin <= 0 and pinned_item is not None:
            return [pinned_item], default_opening
        # Tailoring needs something to tailor to: with no resume and no job listing the curated bank is the plan.
        if selected_bank and not (redacted_resume.strip() or jd_text.strip()):
            ordered = ([pinned_item] + remaining_bank) if pinned_item else selected_bank
            return ordered[:effective_count], default_opening
        want_tailored = (
            slots_after_pin
            if question_source == "resume_jd_tailored"
            else max(1, slots_after_pin // 2)
        )
        bank_slice = (
            []
            if question_source == "resume_jd_tailored"
            else remaining_bank[: max(0, slots_after_pin - want_tailored)]
        )

        sys_p = render_prompt(
            "practice_plan_tailored",
            focus_brief=practice_focus_brief(rubric.grader, coached=rubric.coached),
        )
        usr_p = (
            f"Target Role: {role_label} ({target_level}) | Format: {interview_type}\n"
            f"Generate exactly {want_tailored} distinct, tailored interview question(s).\n\n"
            f"Redacted Candidate Resume:\n{(redacted_resume or 'Not provided')[:4500]}\n\n"
            f"Target Job Listing:\n{(jd_text or 'Not provided')[:3500]}\n"
        )
        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system=sys_p,
                user=usr_p,
                temperature=0.45,
                max_tokens=900,
                timeout=35.0,
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            opening = str(parsed.get("opening") or "").strip() or default_opening
            tailored_items: list[dict[str, Any]] = []
            raw_qs = parsed.get("questions") if isinstance(parsed.get("questions"), list) else []
            for item in raw_qs[:want_tailored]:
                if not isinstance(item, dict):
                    continue
                prompt_text = str(item.get("prompt") or "").strip()
                if not prompt_text:
                    continue
                tailored_items.append(
                    {
                        "bank_question_id": None,
                        "source": "tailored",
                        "category": interview_type,
                        "title": str(item.get("title") or "Tailored Question").strip(),
                        "competency": str(item.get("competency") or "Role Fit").strip(),
                        "prompt": prompt_text,
                        "carl_focus": str(item.get("carl_focus") or rubric.default_focus).strip(),
                        "follow_ups": [str(f).strip() for f in (item.get("follow_ups") or [])[:3] if str(f).strip()],
                        "core_requirements": [],
                        "deep_dives": [],
                        "starter_code": {},
                        "tests": {},
                        "difficulty": "",
                    }
                )
            combined: list[dict[str, Any]] = []
            if pinned_item is not None:
                combined.append(pinned_item)
            # Interleave tailored + remaining bank for mixed mode
            for i in range(max(len(tailored_items), len(bank_slice))):
                if i < len(tailored_items):
                    combined.append(tailored_items[i])
                if i < len(bank_slice):
                    combined.append(bank_slice[i])
            return (combined[:effective_count] or selected_bank), opening
        except Exception as exc:
            logger.warning("LLM generate_practice_plan failed (%s); falling back to curated banks", exc)
            ordered = ([pinned_item] + remaining_bank) if pinned_item else selected_bank
            return ordered[:effective_count], default_opening

    async def generate_practice_interviewer_reply(
        self,
        *,
        role_title: str,
        company: str,
        interview_type: str,
        target_level: str,
        coach_style: str,
        current_question: dict[str, Any],
        next_question: dict[str, Any] | None,
        recent_turns: list[dict[str, Any]],
        candidate_answer: str,
        follow_up_count: int,
        existing_carl: dict[str, str],
        workspace_context: dict[str, Any] | None = None,
        cache_key: str | None = None,
        session: PracticeSession | None = None,
        rubric: Rubric = CARL,
    ) -> dict[str, Any]:
        """
        Generate the AI interviewer's natural next turn + live CARL status + supportive coach nudge.
        When `session` is provided, executes through the unified append-only multi-turn conversation
        (`build_practice_turn_messages` -> `_complete_messages`), giving optimal prompt caching on
        both persistent `agy`/`claude` CLI workers and direct HTTP APIs.
        """
        from .practice import build_heuristic_nudge, estimate_carl_heuristic, said_too_little

        if session is not None:
            rubric = session.config.rubric
        heuristic_carl = estimate_carl_heuristic(candidate_answer, existing_carl, rubric)
        follow_ups = list(current_question.get("follow_ups") or [])
        max_followups = 3 if interview_type in ("system_design", "coding") else 2

        # In a conversation round a few words is not an answer, so it is neither thanked nor built on.
        no_answer = interview_type not in ("system_design", "coding") and said_too_little(candidate_answer)

        def _fallback() -> dict[str, Any]:
            if no_answer and follow_up_count < max_followups:
                reply = (
                    "No rush — take a moment and give me one real example. "
                    f"I'll ask again: {current_question.get('prompt', '')}"
                )
                advance = False
            elif not no_answer and follow_up_count < len(follow_ups) and follow_up_count < max_followups:
                reply = f"Thanks for walking through that. {follow_ups[follow_up_count]}"
                advance = False
            elif next_question is not None:
                lead = (
                    "Let's leave that one there and move on to the next question: "
                    if no_answer
                    else "That gives me a clear picture — thank you! Let's move on to our next question: "
                )
                reply = f"{lead}{next_question.get('prompt', '')}"
                advance = True
            else:
                lead = (
                    "Let's stop there for today. "
                    if no_answer
                    else "Thank you for sharing that — you covered a lot of thoughtful ground today! "
                )
                reply = f"{lead}Whenever you're ready, click End & Debrief to review your coaching report."
                advance = True
            return {
                "interviewer_reply": reply,
                "should_advance": advance,
                "carl_signals": heuristic_carl,
                "coach_nudge": build_heuristic_nudge(heuristic_carl, interview_type, rubric, candidate_answer),
            }

        if not self.config.is_configured:
            return _fallback()

        conv_commit: dict[str, Any] | None = None
        session_key = f"practice:{session.id}" if session is not None else None

        try:
            if session is not None:
                messages, pending_user, ws_snapshot, sys_prompt, reset_conv = build_practice_turn_messages(
                    session,
                    current_question=current_question,
                    next_question=next_question,
                    candidate_answer=candidate_answer,
                    follow_up_count=follow_up_count,
                    max_followups=max_followups,
                    workspace_context=workspace_context,
                )
                raw = await self._complete_messages(
                    messages,
                    model=self.config.fast_model,
                    temperature=0.4,
                    max_tokens=480,
                    timeout=30.0,
                    expect_json=True,
                    cache_key=cache_key or f"mockery:practice:{session.id}",
                    session_key=session_key,
                    operation="practice_turn",
                )
                conv_commit = {
                    "system_prompt": sys_prompt,
                    "user_content": pending_user,
                    "assistant_raw": raw,
                    "workspace_snapshot": ws_snapshot,
                    "reset_conversation": reset_conv,
                }
            else:
                history_lines = [
                    f"{'INTERVIEWER' if t.get('role') == 'interviewer' else 'CANDIDATE'}: {t.get('text', '')}"
                    for t in recent_turns[-24:]
                    if t.get("role") in ("interviewer", "candidate")
                ]
                ws_block = ""
                if workspace_context:
                    if workspace_context.get("diagram_summary"):
                        ws_block += f"\nLIVE SYSTEM DESIGN DIAGRAM STATE:\n{str(workspace_context['diagram_summary'])[:2500]}\n"
                    if workspace_context.get("code"):
                        ws_block += (
                            f"\nLIVE CODING WORKSPACE ({workspace_context.get('language', 'code')}):\n"
                            f"```\n{str(workspace_context['code'])[:3000]}\n```\n"
                        )
                    if workspace_context.get("test_output"):
                        ws_block += f"\nLATEST TEST RUN OUTPUT:\n{str(workspace_context['test_output'])[:1200]}\n"

                next_q_block = (
                    f"Next planned question (ONLY introduce if should_advance=true): {next_question.get('prompt')}"
                    if next_question
                    else f"This is the FINAL question of the interview. If should_advance=true, wrap up warmly and invite them to view their {rubric.debrief_name}."
                )

                tone_instruction = practice_tone_instruction(coach_style)
                system_prompt = render_prompt(
                    "practice_interviewer_reply",
                    interview_type_label=interview_type.replace("_", " "),
                    role_title=role_title,
                    target_level=target_level,
                    tone_instruction=tone_instruction,
                    **practice_reply_rubric_args(rubric.grader, rubric.signal_keys, coached=rubric.coached),
                )
                stable_prefix = (
                    f"Current Question: {current_question.get('prompt')}\n"
                    f"{rubric.focus_label}: {current_question.get('carl_focus') or ', '.join(current_question.get('core_requirements') or [])}\n"
                    f"Suggested follow-up angles: {follow_ups}\n"
                    f"{next_q_block}\n\n"
                    f"Recent conversation:\n" + ("\n".join(history_lines) if history_lines else "(Opening turn)")
                )
                mutable_suffix = (
                    f"\n\nFollow-ups already asked on this question: {follow_up_count} "
                    f"(advance once you've asked {max_followups} follow-ups OR they've already given strong complete signal)."
                    f"{ws_block}\n\n"
                    f"Candidate's latest answer:\n{candidate_answer}"
                )
                user_prompt = stable_prefix + mutable_suffix

                raw = await self._complete_text(
                    model=self.config.fast_model,
                    system=system_prompt,
                    user=user_prompt,
                    temperature=0.4,
                    max_tokens=480,
                    timeout=30.0,
                    expect_json=True,
                    cache_prefix_len=len(stable_prefix),
                    cache_key=cache_key,
                )

            parsed = _extract_json_object(raw)
            reply = str(parsed.get("interviewer_reply") or "").strip()
            if not reply:
                if session_key:
                    await self.close_session_worker(session_key)
                return {**_fallback(), "fallback_reason": built_in_reason(self.config, None)}
            raw_adv = parsed.get("should_advance", False)
            if isinstance(raw_adv, bool):
                should_advance = raw_adv
            elif isinstance(raw_adv, str):
                should_advance = raw_adv.strip().lower() in ("true", "1", "yes")
            else:
                should_advance = False
            if follow_up_count >= max_followups:
                should_advance = True

            graded = rubric.signals_from_grader(parsed.get("carl_signals"))
            rank = {level: idx for idx, level in enumerate(SIGNAL_LEVELS)}
            merged_carl: dict[str, str] = {}
            for dim in SIGNAL_SLOTS:
                llm_val = graded.get(dim, heuristic_carl[dim])
                prev_val = existing_carl.get(dim, "missing")
                merged_carl[dim] = SIGNAL_LEVELS[max(rank.get(prev_val, 0), rank[llm_val])]

            nudge_raw = parsed.get("coach_nudge") if isinstance(parsed.get("coach_nudge"), dict) else {}
            nudge_msg = str(nudge_raw.get("message") or "").strip()
            nudge_tone = "encourage" if nudge_raw.get("tone") == "encourage" else "tip"
            nudge = (
                {"tone": nudge_tone, "message": nudge_msg}
                if nudge_msg
                else build_heuristic_nudge(merged_carl, interview_type, rubric, candidate_answer)
            )
            out: dict[str, Any] = {
                "interviewer_reply": reply,
                "should_advance": should_advance,
                "carl_signals": merged_carl,
                "coach_nudge": nudge,
            }
            if conv_commit is not None:
                out["_conversation_commit"] = conv_commit
            return out
        except Exception as exc:
            logger.warning("LLM generate_practice_interviewer_reply failed (%s); using fallback", type(exc).__name__)
            if session_key:
                await self.close_session_worker(session_key)
            return {**_fallback(), "fallback_reason": built_in_reason(self.config, exc)}

    async def generate_practice_hint(
        self,
        *,
        interview_type: str,
        target_level: str,
        current_question: dict[str, Any],
        recent_turns: list[dict[str, Any]],
        workspace_context: dict[str, Any] | None = None,
        rubric: Rubric = CARL,
    ) -> str:
        """Return a supportive, non-spoiler coaching hint when the candidate asks for help."""
        fallback = rubric.offline_hint.format(
            focus=current_question.get("carl_focus") or "your own decisions & measurable impact"
        )
        if interview_type == "system_design":
            reqs = current_question.get("core_requirements") or []
            fallback = (
                f"Sketch the core request flow first (Client → API → Storage), then pick one key requirement "
                f"to nail in detail{': ' + reqs[0] if reqs else ''}."
            )
        elif interview_type == "coding":
            fallback = (
                "Start with a clear brute-force or invariant statement, verify it on a tiny example, "
                "and then ask what data structure eliminates the repeated work."
            )

        if not self.config.is_configured:
            return fallback

        raw_hist_lines = [
            f"{t.get('role', '').upper()}: {t.get('text', '')}"
            for t in recent_turns[-5:]
            if t.get("role") in ("interviewer", "candidate")
        ]
        # Bound hint history newest-first so long candidate answers never produce 30k+ char hint prompts.
        kept_hist: list[str] = []
        used_chars = 0
        for line in reversed(raw_hist_lines):
            clipped = line[:_HINT_HISTORY_MAX_CHARS] if not kept_hist and len(line) > _HINT_HISTORY_MAX_CHARS else line
            cost = len(clipped) + 1
            if kept_hist and used_chars + cost > _HINT_HISTORY_MAX_CHARS:
                break
            kept_hist.append(clipped)
            used_chars += cost
        kept_hist.reverse()
        history = "\n".join(kept_hist)
        ws_note = ""
        if workspace_context:
            if workspace_context.get("diagram_summary"):
                ws_note += f"\nCurrent diagram:\n{str(workspace_context['diagram_summary'])[:1500]}"
            if workspace_context.get("code"):
                ws_note += f"\nCurrent code:\n{str(workspace_context['code'])[:1500]}"

        sys_p = render_prompt(
            "practice_hint",
            behavioral_nudge=practice_hint_behavioral_nudge(rubric.grader, coached=rubric.coached),
        )
        usr_p = (
            f"Format: {interview_type} ({target_level})\n"
            f"Question: {current_question.get('prompt')}\n"
            f"Focus: {current_question.get('carl_focus') or ', '.join(current_question.get('core_requirements') or [])}\n"
            f"Recent exchange:\n{history}{ws_note}"
        )
        try:
            txt = await self._complete_text(
                model=self.config.fast_model,
                system=sys_p,
                user=usr_p,
                temperature=0.4,
                max_tokens=160,
                timeout=20.0,
                expect_json=False,
                use_warm_spare=True,
                operation="practice_hint",
            )
            return txt.strip() or fallback
        except Exception:
            return fallback

    async def generate_practice_debrief(
        self,
        *,
        role_title: str,
        company: str,
        interview_type: str,
        target_level: str,
        questions: list[dict[str, Any]],
        turns: list[dict[str, Any]],
        workspace_context: dict[str, Any] | None = None,
        rubric: Rubric = CARL,
    ) -> dict[str, Any]:
        """
        Generate a supportive coaching debrief in the session's rubric:
          - Leads with genuine strengths & confidence-building encouragement
          - Per-question signal breakdown + concrete Before -> After reframe
          - One targeted 45-second Retry Weak Part drill
        """
        # Build deterministic fallback debrief first (also guarantees schema safety)
        q_reviews_fb: list[dict[str, Any]] = []
        weakest_q: dict[str, Any] | None = None
        weakest_dim = rubric.drill_order[0]
        finish = f"your {rubric.name} finish" if rubric.named_framework else "how you land each answer"
        earned_strengths: list[str] = []
        anything_to_coach = False

        for q in questions:
            q_turns = [t for t in turns if t.get("question_id") == q.get("id") and t.get("role") == "candidate"]
            if not q_turns:
                continue
            carl = dict(q.get("carl_signals") or {"context": "partial", "action": "partial", "result": "missing", "learning": "missing"})
            focus_dim = rubric.drill_slot(carl)
            focus_label = rubric.signal(focus_dim).label if rubric.coached else "main point"
            if weakest_q is None or carl.get(focus_dim) == "missing":
                weakest_q = q
                weakest_dim = focus_dim
            q_title = q.get("title") or q.get("competency") or "Practice Question"
            words_said = sum(len(str(t.get("text") or "").split()) for t in q_turns)
            landed = [s.label for s in rubric.signals if rubric.coached and carl.get(s.slot) == "strong"]
            touched = [s.label for s in rubric.signals if rubric.coached and carl.get(s.slot) == "partial"]
            if landed:
                strength = f"Your {_join_labels(landed)} came through clearly."
                earned_strengths.append(f"{_join_labels(landed)} came through clearly in “{q_title}”.")
            elif touched:
                strength = f"You touched on {_join_labels(touched)}, which gives you something to build on."
            elif not rubric.coached and words_said >= 40:
                strength = "You gave this a full answer in your own words."
            else:
                strength = ""
            answered = bool(landed or touched or words_said >= 40)
            anything_to_coach = anything_to_coach or answered
            q_reviews_fb.append(
                {
                    "question_id": q.get("id"),
                    "title": q_title,
                    "prompt": q.get("prompt") or "",
                    "carl": carl,
                    "strength": strength,
                    "coaching_tip": (
                        f"Next pass, spend 15 extra seconds sharpening your {focus_label} so a hiring panel immediately hears your {target_level} impact."
                        if answered
                        else rubric.nothing_yet
                    ),
                    "carl_reframe": rubric.reframe_tip,
                }
            )

        weakest_label = rubric.signal(weakest_dim).label if rubric.coached else "main point"
        fallback_debrief: dict[str, Any] = {
            "overall_encouragement": (
                (
                    f"Good rep. Practicing out loud for {role_title} ({target_level}) builds real fluency, "
                    f"and tightening {finish} will make your answers land with more confidence."
                )
                if anything_to_coach
                else (
                    "These answers were too short to coach, so there is nothing to score yet. "
                    "Pick one real example for each question and talk it through for a minute or two; "
                    "then this debrief will have something to work with."
                )
            ),
            "top_strengths": earned_strengths[:3],
            "carl_primer": rubric.primer,
            "question_reviews": q_reviews_fb,
            "retry_drill": {
                "question_id": (weakest_q or (questions[0] if questions else {})).get("id"),
                "focus": f"Sharpen {weakest_label} in 45 seconds",
                "prompt": (
                    f"In 45 seconds, re-deliver just the {weakest_label.upper()} portion of your answer to: "
                    f"\"{(weakest_q or (questions[0] if questions else {})).get('prompt', 'your main story')}\""
                ),
            },
        }

        non_retry_turns = [t for t in turns if not t.get("is_retry")]
        if not self.config.is_configured or not any(t.get("role") == "candidate" for t in non_retry_turns):
            return fallback_debrief

        answered_ids = {
            str(t.get("question_id"))
            for t in non_retry_turns
            if t.get("role") == "candidate" and t.get("question_id")
        }
        answered_questions = [q for q in questions if str(q.get("id")) in answered_ids] or questions[:1]
        per_q_budget = max(1200, 10_000 // max(1, len(answered_questions)))

        transcript_sections: list[str] = []
        for q in answered_questions:
            qid = str(q.get("id") or "")
            q_lines = [
                f"{t['role'].upper()}: {t.get('text', '')}"
                for t in non_retry_turns
                if str(t.get("question_id") or "") == qid and t.get("role") in ("interviewer", "candidate")
            ]
            excerpt = _head_tail_excerpt("\n".join(q_lines), per_q_budget)
            transcript_sections.append(f"### [{qid}] {q.get('title', 'Question')}\n{excerpt}")

        ws_block = ""
        if workspace_context:
            if workspace_context.get("diagram_summary"):
                ws_block += f"\nFINAL DIAGRAM:\n{str(workspace_context['diagram_summary'])[:2000]}\n"
            if workspace_context.get("code"):
                ws_block += f"\nFINAL CODE:\n{str(workspace_context['code'])[:2500]}\n"
            if workspace_context.get("test_output"):
                ws_block += f"\nTEST RESULTS:\n{str(workspace_context['test_output'])[:1000]}\n"

        system_prompt = render_prompt(
            "practice_debrief",
            **practice_debrief_rubric_args(rubric.grader, rubric.signal_keys, coached=rubric.coached),
        )
        user_prompt = (
            f"Role: {role_title}" + (f" at {company}" if company else "") + f" | Target Level: {target_level} | Format: {interview_type}\n"
            f"Answered Questions:\n"
            + "\n".join(f"- {q.get('id')}: {q.get('title')} — {q.get('prompt')}" for q in answered_questions)
            + "\n\nPer-Question Transcript Excerpts:\n"
            + "\n\n".join(transcript_sections)
            + ws_block
        )

        def _norm_carl(raw_c: Any, default_c: dict[str, str]) -> dict[str, str]:
            graded = rubric.signals_from_grader(raw_c)
            return {dim: graded.get(dim, default_c.get(dim, "missing")) for dim in SIGNAL_SLOTS}

        try:
            debrief_model = self.config.calibration_model or self.config.fast_model
            is_high_effort = _is_reasoning_model(debrief_model) or debrief_model.lower().endswith(
                (":high", ":xhigh", ":max")
            )
            debrief_max_tokens = 3200 if is_high_effort else 1800
            debrief_timeout = 90.0 if is_high_effort else 60.0
            raw = await self._complete_text(
                model=debrief_model,
                system=system_prompt,
                user=user_prompt,
                temperature=0.35,
                max_tokens=debrief_max_tokens,
                timeout=debrief_timeout,
                expect_json=True,
                disable_fast_thinking=False,
                operation="practice_debrief",
            )
            try:
                parsed = _extract_json_object(raw)
            except json.JSONDecodeError as parse_exc:
                looks_truncated = _looks_like_truncated_json(raw, getattr(self, "_last_finish_reason", None))
                retry_max_tokens = min(4800, debrief_max_tokens + 1200) if looks_truncated else debrief_max_tokens
                logger.warning(
                    "LLM generate_practice_debrief returned malformed JSON (%s; truncated=%s); retrying once at temperature=0.0",
                    parse_exc,
                    looks_truncated,
                )
                raw_retry = await self._complete_text(
                    model=debrief_model,
                    system=system_prompt,
                    user=(
                        user_prompt
                        + "\n\nIMPORTANT: Your previous response had invalid JSON syntax. "
                        "Return ONLY one strictly valid raw JSON object matching the schema. "
                        "Use single quotes inside string values instead of unescaped double quotes."
                    ),
                    temperature=0.0,
                    max_tokens=retry_max_tokens,
                    timeout=debrief_timeout,
                    expect_json=True,
                    disable_fast_thinking=False,
                    operation="practice_debrief_retry",
                )
                parsed = _extract_json_object(raw_retry)
            overall = str(parsed.get("overall_encouragement") or "").strip()
            raw_reviews = parsed.get("question_reviews")
            if not overall or not isinstance(raw_reviews, list):
                return {**fallback_debrief, "fallback_reason": built_in_reason(self.config, None)}

            all_q_ids = {str(q.get("id")) for q in questions if q.get("id")}
            answered_q_by_id = {str(q.get("id")): q for q in answered_questions if q.get("id")}
            seen_review_qids: set[str] = set()
            clean_reviews: list[dict[str, Any]] = []
            for qr in raw_reviews:
                if not isinstance(qr, dict):
                    continue
                qid = str(qr.get("question_id") or "").strip()
                matched_q = answered_q_by_id.get(qid)
                if matched_q is None and len(answered_questions) == 1 and qid not in all_q_ids:
                    matched_q = answered_questions[0]
                    qid = str(matched_q.get("id") or "")
                if matched_q is None or qid in seen_review_qids:
                    continue
                seen_review_qids.add(qid)
                default_carl = dict(matched_q.get("carl_signals") or {})
                clean_reviews.append(
                    {
                        "question_id": qid,
                        "title": str(qr.get("title") or matched_q.get("title") or "Practice Question").strip(),
                        "prompt": str(qr.get("prompt") or matched_q.get("prompt") or "").strip(),
                        "carl": _norm_carl(qr.get("carl"), default_carl),
                        "strength": str(qr.get("strength") or "").strip(),
                        "coaching_tip": str(qr.get("coaching_tip") or "").strip(),
                        "carl_reframe": str(qr.get("carl_reframe") or "").strip(),
                    }
                )
            if not clean_reviews:
                return {**fallback_debrief, "fallback_reason": built_in_reason(self.config, None)}

            # An empty list is the model saying nothing earned praise; only a missing list is filled in.
            if isinstance(parsed.get("top_strengths"), list):
                clean_strengths = [str(s).strip() for s in parsed["top_strengths"] if isinstance(s, str) and s.strip()][:5]
            else:
                clean_strengths = list(fallback_debrief["top_strengths"])

            raw_drill = parsed.get("retry_drill") if isinstance(parsed.get("retry_drill"), dict) else {}
            drill_qid = str(raw_drill.get("question_id") or "").strip()
            drill_prompt = str(raw_drill.get("prompt") or "").strip()
            if not drill_prompt or drill_qid not in answered_q_by_id:
                clean_drill = dict(fallback_debrief["retry_drill"])
            else:
                clean_drill = {
                    "question_id": drill_qid,
                    "focus": str(raw_drill.get("focus") or "Sharpen clarity in 45 seconds").strip(),
                    "prompt": drill_prompt,
                }

            return {
                "overall_encouragement": overall,
                "top_strengths": clean_strengths,
                "carl_primer": str(parsed.get("carl_primer") or fallback_debrief["carl_primer"]).strip(),
                "question_reviews": clean_reviews,
                "retry_drill": clean_drill,
            }
        except Exception as exc:
            logger.warning("LLM generate_practice_debrief failed (%s); using heuristic debrief", exc)
            return {**fallback_debrief, "fallback_reason": built_in_reason(self.config, exc)}

    async def evaluate_practice_retry(
        self,
        *,
        role_title: str,
        target_level: str,
        drill_focus: str,
        drill_prompt: str,
        original_answer: str,
        retry_answer: str,
        rubric: Rubric = CARL,
    ) -> dict[str, Any]:
        """Give immediate, encouraging before-vs-after coaching on a 45-second Retry Weak Part drill."""
        from .practice import estimate_carl_heuristic

        rank = {"missing": 0, "partial": 1, "strong": 2}
        carl_before = estimate_carl_heuristic(original_answer, rubric=rubric)
        carl_after = estimate_carl_heuristic(retry_answer, rubric=rubric)
        signal_word = f"{rubric.name} signal" if rubric.named_framework else "signal"
        improved_dims = [
            rubric.signal(dim).label
            for dim in SIGNAL_SLOTS
            if rubric.coached and rank[carl_after[dim]] > rank[carl_before[dim]]
        ]
        missing_after = [
            rubric.signal(dim).label
            for dim in rubric.drill_order
            if rubric.coached and carl_after[dim] != "strong"
        ]

        if improved_dims:
            encourage_fb = f"Great rep — you clearly leveled up {', '.join(improved_dims)} in this 45-second pass!"
            improved_fb = (
                f"Compared to your first answer, {', '.join(improved_dims)} moved up in {signal_word} "
                "and reads much tighter for a panel."
            )
        elif sum(rank[v] for v in carl_after.values()) >= 6:
            encourage_fb = f"Strong, crisp delivery — you kept your {signal_word} intact in a tight 45-second window!"
            improved_fb = "You maintained clear ownership and measurable impact while keeping the pacing focused."
        else:
            encourage_fb = "Good effort getting another rep in — let's sharpen one concrete detail on the next pass."
            improved_fb = (
                "The retry stayed close to your original coverage; adding one more concrete detail "
                "will create a bigger before → after jump."
            )
        polish_fb = (
            f"For one final polish, add a crisp sentence strengthening {missing_after[0]}."
            if missing_after
            else "Lock this phrasing in — it's clear, specific, and tight."
        )

        fallback = {
            "encouragement": encourage_fb,
            "what_improved": improved_fb,
            "one_more_polish": polish_fb,
            "carl_after": carl_after,
        }
        if not self.config.is_configured:
            return fallback

        sys_p = render_prompt("practice_retry_eval", signal_schema=practice_signal_schema(rubric.signal_keys))
        usr_p = (
            f"Role: {role_title} ({target_level})\n"
            f"Retry Drill Focus: {drill_focus}\nDrill Prompt: {drill_prompt}\n\n"
            f"Original Answer:\n{original_answer[:2500]}\n\n"
            f"45-Second Retry Answer:\n{retry_answer[:2500]}"
        )
        try:
            raw = await self._complete_text(
                model=self.config.fast_model,
                system=sys_p,
                user=usr_p,
                temperature=0.35,
                max_tokens=380,
                timeout=25.0,
                expect_json=True,
            )
            parsed = _extract_json_object(raw)
            enc = str(parsed.get("encouragement") or "").strip()
            if not enc:
                return {**fallback, "fallback_reason": built_in_reason(self.config, None)}
            graded_after = rubric.signals_from_grader(parsed.get("carl_after"))
            norm_ca = {dim: graded_after.get(dim, carl_after[dim]) for dim in SIGNAL_SLOTS}
            return {
                "encouragement": enc,
                "what_improved": str(parsed.get("what_improved") or improved_fb).strip(),
                "one_more_polish": str(parsed.get("one_more_polish") or polish_fb).strip(),
                "carl_after": norm_ca,
            }
        except Exception as exc:
            logger.warning("LLM evaluate_practice_retry failed (%s); using fallback", exc)
            return {**fallback, "fallback_reason": built_in_reason(self.config, exc)}


_probe_guidance_for = probe_guidance_for
_calibration_rubric_for = calibration_rubric_for


def _join_labels(labels: list[str]) -> str:
    if len(labels) <= 2:
        return " and ".join(labels)
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


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


UNUSABLE_MODEL_REPLY = (
    "Your AI model answered, but not in a form Mockery could use. A more capable model usually fixes this."
)


def built_in_reason(cfg: LLMRuntimeConfig, exc: Exception | None) -> str:
    """Why a built-in reply is standing in for the connected model, in words the user can act on."""
    if exc is None or isinstance(exc, (json.JSONDecodeError, KeyError, TypeError, ValueError)):
        return UNUSABLE_MODEL_REPLY
    return f"Your AI did not respond. {_friendly_connection_hint(cfg, str(exc))}"


def _friendly_connection_hint(cfg: LLMRuntimeConfig, err_msg: str) -> str:
    low = err_msg.lower()
    if cfg.cli_backend:
        return f"Make sure `{cfg.cli_backend}` is signed in on your Mac (try running it once in Terminal)."
    if "401" in low or "unauthorized" in low or "invalid api key" in low or "authentication" in low:
        if cfg.provider == "omlx":
            return "oMLX requires an API key — paste your oMLX API key (or set OMLX_API_KEY) in Settings."
        return "Double-check that your API key is copied completely and has not expired."
    if "404" in low or ("model" in low and "not found" in low):
        return f"Model `{cfg.fast_model}` was not found by this provider — check the model name in Advanced settings."
    if "429" in low or "quota" in low or "rate limit" in low:
        return "Provider returned a rate limit or quota error — verify billing/free-tier quota."
    if cfg.provider in LOCAL_HTTP_PROVIDERS or "connection" in low or "refused" in low:
        if cfg.provider == "omlx":
            return "oMLX does not appear to be running. Open the oMLX menu-bar app (or run `omlx serve`) and load a model first."
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


def _resolve_omlx_probe_target() -> tuple[int, str]:
    """Read oMLX's port (OMLX_PORT -> ~/.omlx/settings.json -> 8000) and optional API key for local probing."""
    port = 8000
    api_key = os.getenv("OMLX_API_KEY", "").strip()
    settings_path = Path.home() / ".omlx" / "settings.json"
    if settings_path.is_file():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cand_port = raw.get("port")
                if cand_port is None and isinstance(raw.get("server"), dict):
                    cand_port = raw["server"].get("port")
                if isinstance(cand_port, int) and 1 <= cand_port <= 65535:
                    port = cand_port
                elif isinstance(cand_port, str) and cand_port.isdigit():
                    p = int(cand_port)
                    if 1 <= p <= 65535:
                        port = p
                if not api_key:
                    cand_key = raw.get("api_key")
                    if cand_key is None and isinstance(raw.get("auth"), dict):
                        cand_key = raw["auth"].get("api_key")
                    if isinstance(cand_key, str) and cand_key.strip():
                        api_key = cand_key.strip()
        except Exception:
            pass
    env_port = os.getenv("OMLX_PORT", "").strip()
    if env_port.isdigit():
        p = int(env_port)
        if 1 <= p <= 65535:
            port = p
    return port, api_key


async def _probe_omlx(timeout: float = 0.6) -> dict[str, Any]:
    port, api_key = _resolve_omlx_probe_target()
    base_url = f"http://localhost:{port}/v1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(f"http://127.0.0.1:{port}/v1/models", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    models = [
                        str(m.get("id", "")).strip()
                        for m in data["data"]
                        if isinstance(m, dict) and m.get("id")
                    ]
                    return {
                        "running": True,
                        "models": models,
                        "base_url": base_url,
                        "auth_required": False,
                    }
            elif resp.status_code in (401, 403):
                omlx_present = (
                    (Path.home() / ".omlx").exists()
                    or shutil.which("omlx") is not None
                    or Path("/Applications/oMLX.app").exists()
                )
                if omlx_present:
                    return {
                        "running": True,
                        "models": [],
                        "base_url": base_url,
                        "auth_required": True,
                    }
    except Exception:
        pass
    return {"running": False, "models": [], "base_url": base_url, "auth_required": False}


async def _probe_local_http_models(url: str, list_key: str, item_key: str, timeout: float = 0.6) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(url)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    str(m.get(item_key, "")).strip()
                    for m in (data.get(list_key) or [])
                    if isinstance(m, dict) and m.get(item_key)
                ]
                return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}


async def _probe_ollama(timeout: float = 0.6) -> dict[str, Any]:
    return await _probe_local_http_models("http://127.0.0.1:11434/api/tags", "models", "name", timeout)


async def _probe_lmstudio(timeout: float = 0.6) -> dict[str, Any]:
    return await _probe_local_http_models("http://127.0.0.1:1234/v1/models", "data", "id", timeout)


async def discover_available_providers() -> dict[str, Any]:
    """
    Fast (<600ms) local discovery of:
      - Installed subscription CLIs (`claude`, `codex`, `gemini`, `agy`, `pi`)
      - Running local OpenAI-compatible runtimes (oMLX, Ollama, LM Studio) and their installed models
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

    omlx_info, ollama_info, lmstudio_info = await asyncio.gather(
        _probe_omlx(), _probe_ollama(), _probe_lmstudio()
    )

    env_keys_present = [
        prov
        for prov, env_name in PROVIDER_ENV_KEYS.items()
        if bool(os.getenv(env_name, "").strip())
    ]

    return {
        "detected_clis": detected_clis,
        "local_servers": {
            "omlx": omlx_info,
            "ollama": ollama_info,
            "lmstudio": lmstudio_info,
        },
        "env_keys_present": env_keys_present,
        "system_memory_gb": get_system_memory_gb(),
    }

