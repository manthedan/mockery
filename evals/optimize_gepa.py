#!/usr/bin/env python3
"""Offline GEPA prompt-optimization CLI for Mockery Tier 1 prompts.

Strictly dev-only: never imported by runtime code under ``src/mockery/``.
Outputs candidate prompt, unified diff, and JSON evaluation report without
silently mutating ``src/mockery/prompts.py``.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics import (  # noqa: E402
    TARGET_TO_PROMPT_NAME,
    build_user_prompt_for_example,
    load_jsonl_dataset,
    render_system_prompt_for_example,
    score_example,
    split_dataset,
    validate_candidate_prompt,
    validate_dataset,
)
from mockery.llm import _fast_extra_body, _is_reasoning_model  # noqa: E402
from mockery.prompts import get_prompt_spec  # noqa: E402

DEFAULT_DATASETS: dict[str, Path] = {
    "triage_window": REPO_ROOT / "evals" / "datasets" / "triage_window.jsonl",
    "practice_interviewer": REPO_ROOT / "evals" / "datasets" / "practice_interviewer.jsonl",
}

TARGET_OBJECTIVES: dict[str, str] = {
    "triage_window": (
        "Optimize the system prompt for fast, high-precision interview turn triage. "
        "Maximize strict raw-JSON output, accurate single-mic speaker diarization and rapid-handoff "
        "splitting, verbatim word preservation (fixing only clear STT proper-noun typos), accurate "
        "card boundary detection without over-splitting follow-ups, and calibrated confidence. "
        "MUST keep the `{diarize_rules}` placeholder verbatim."
    ),
    "practice_interviewer": (
        "Optimize the system prompt for a live supportive AI mock interviewer. "
        "Maximize strict raw-JSON output, natural concise spoken replies (1-3 sentences) grounded in "
        "the candidate's answer and workspace context, accurate `should_advance` timing, accurate and "
        "strictly monotonic `carl_signals`, supportive encouraging coaching tone, and zero code/algorithm spoilers. "
        "MUST keep `{interview_type_label}`, `{role_title}`, `{target_level}`, and `{tone_instruction}` verbatim."
    ),
}

# Match exact runtime parameters in src/mockery/llm.py
TARGET_TEMPERATURES: dict[str, float] = {
    "triage_window": 0.0,
    "practice_interviewer": 0.4,
}

SECRET_TOKEN_RE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._\-]+|sk-[A-Za-z0-9_\-]{8,}|(?:api[_-]?key|token|authorization)\s*[=:]\s*['\"]?[A-Za-z0-9._\-]{6,}['\"]?)",
    re.I,
)


def compute_max_tokens(target: str, example: dict[str, Any]) -> int:
    """Match runtime max_tokens calculation in src/mockery/llm.py."""
    if target == "triage_window":
        n_utts = len(example.get("incoming_utterances") or [])
        return max(450, 220 * max(1, n_utts))
    if target == "practice_interviewer":
        return 480
    raise ValueError(f"Unsupported eval target: {target!r}")


def _sanitize_error_message(exc: BaseException) -> str:
    msg = f"{exc.__class__.__name__}: {exc}"
    msg = re.sub(r"[\r\n]+", " ", msg).strip()
    msg = SECRET_TOKEN_RE.sub("[REDACTED]", msg)
    return msg[:220]


def _is_loopback_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1")


def resolve_task_endpoint_and_key(
    cli_base_url: str | None,
    cli_api_key: str | None,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """
    Resolve task endpoint URL and API key without forwarding an unrelated env key
    when `--task-base-url` explicitly targets a different host/endpoint.
    """
    environ = env if env is not None else dict(os.environ)

    gepa_url = (environ.get("GEPA_TASK_BASE_URL") or "").strip()
    llm_url = (environ.get("LLM_BASE_URL") or "").strip()
    default_omlx_url = "http://localhost:8000/v1"

    if cli_base_url is not None and cli_base_url.strip():
        base_url = cli_base_url.strip()
        url_source = "cli"
    elif gepa_url:
        base_url = gepa_url
        url_source = "gepa_env"
    elif llm_url:
        base_url = llm_url
        url_source = "llm_env"
    else:
        base_url = default_omlx_url
        url_source = "default_omlx"

    if cli_api_key is not None and cli_api_key.strip():
        return base_url, cli_api_key.strip()

    if url_source == "gepa_env":
        key = (environ.get("GEPA_TASK_API_KEY") or "").strip()
        if not key and _is_loopback_url(base_url):
            key = (environ.get("OMLX_API_KEY") or "").strip()
        return base_url, key or "local-no-key"

    if url_source == "llm_env":
        key = (environ.get("LLM_API_KEY") or "").strip()
        if not key and _is_loopback_url(base_url):
            key = (environ.get("OMLX_API_KEY") or "").strip()
        return base_url, key or "local-no-key"

    if url_source == "default_omlx":
        key = (environ.get("OMLX_API_KEY") or "").strip()
        return base_url, key or "local-no-key"

    # CLI explicitly overrode --task-base-url without --task-api-key:
    # NEVER forward unrelated OMLX_API_KEY or LLM_API_KEY to a different host.
    if gepa_url and base_url.rstrip("/") == gepa_url.rstrip("/"):
        key = (environ.get("GEPA_TASK_API_KEY") or "").strip()
        if key:
            return base_url, key
    if llm_url and base_url.rstrip("/") == llm_url.rstrip("/"):
        key = (environ.get("LLM_API_KEY") or "").strip()
        if key:
            return base_url, key
    if _is_loopback_url(base_url) and base_url.rstrip("/") == default_omlx_url:
        key = (environ.get("OMLX_API_KEY") or "").strip()
        if key:
            return base_url, key
    return base_url, "local-no-key"


def preflight_gepa_dependencies(reflection_model: Any) -> None:
    """Verify optional GEPA + reflection dependencies before issuing any task LM requests."""
    missing: list[str] = []
    if importlib.util.find_spec("gepa") is None:
        missing.append("gepa")
    else:
        try:
            from gepa.optimize_anything import (  # noqa: F401
                EngineConfig,
                GEPAConfig,
                ReflectionConfig,
                optimize_anything,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Installed `gepa` package is missing `gepa.optimize_anything` ({_sanitize_error_message(exc)}). "
                "Upgrade with: uv sync --extra gepa"
            ) from exc

    if isinstance(reflection_model, str) and importlib.util.find_spec("litellm") is None:
        missing.append("litellm")

    if missing:
        raise RuntimeError(
            f"Missing optional GEPA reflection dependencies: {', '.join(missing)}. "
            "Install them in your dev environment with: uv sync --extra gepa"
        )


def make_openai_compatible_task_runner(
    *,
    target: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    timeout: float = 45.0,
) -> Callable[[str, str, dict[str, Any]], str]:
    """Build a synchronous OpenAI-compatible runner (works with oMLX, Ollama, LM Studio, OpenAI, etc.)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "local-no-key", base_url=base_url, timeout=timeout)
    is_reasoning = _is_reasoning_model(model)
    extra = _fast_extra_body(model)

    def _run(system_prompt: str, user_prompt: str, example: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": compute_max_tokens(target, example),
        }
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if extra:
            kwargs["extra_body"] = extra
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    return _run


def make_dry_run_runner(
    mock_responses: dict[str, str] | None = None,
) -> Callable[[str, str, dict[str, Any]], str]:
    """Zero-network runner returning supplied mock outputs or each example's reference_response."""
    overrides = mock_responses or {}

    def _run(_system_prompt: str, _user_prompt: str, example: dict[str, Any]) -> str:
        ex_id = str(example.get("id") or "")
        if ex_id in overrides:
            return overrides[ex_id]
        return str(example.get("reference_response") or "{}")

    return _run


def evaluate_candidate_on_examples(
    *,
    target: str,
    candidate_text: str,
    examples: list[dict[str, Any]],
    runner: Callable[[str, str, dict[str, Any]], str],
) -> dict[str, Any]:
    ok, errors = validate_candidate_prompt(target, candidate_text)
    if not ok:
        return {
            "mean_score": 0.0,
            "usable_count": 0,
            "invocation_error_count": 0,
            "backend_failed": False,
            "valid_prompt": False,
            "prompt_errors": errors,
            "results": [
                {
                    "example_id": ex.get("id"),
                    "score": 0.0,
                    "invocation_error": False,
                    "side_info": {
                        "example_id": ex.get("id"),
                        "subscores": {},
                        "failures": list(errors),
                        "suggestions": ["Restore all required prompt placeholders verbatim."],
                    },
                }
                for ex in examples
            ],
        }

    results: list[dict[str, Any]] = []
    usable_scores: list[float] = []
    invocation_error_count = 0

    for ex in examples:
        sys_prompt = render_system_prompt_for_example(target, ex, override_text=candidate_text)
        user_prompt = build_user_prompt_for_example(target, ex)
        try:
            raw_output = runner(sys_prompt, user_prompt, ex)
        except Exception as exc:
            invocation_error_count += 1
            safe_err = _sanitize_error_message(exc)
            results.append(
                {
                    "example_id": ex.get("id"),
                    "score": None,
                    "invocation_error": True,
                    "error": safe_err,
                    "side_info": {
                        "example_id": ex.get("id"),
                        "subscores": {},
                        "failures": [f"Task backend invocation failed ({safe_err})"],
                        "suggestions": ["Check task model endpoint connectivity, auth, and model ID."],
                    },
                }
            )
        else:
            score, side_info = score_example(target, raw_output, ex)
            usable_scores.append(score)
            results.append(
                {
                    "example_id": ex.get("id"),
                    "score": score,
                    "invocation_error": False,
                    "side_info": side_info,
                }
            )

    usable_count = len(usable_scores)
    mean_score = round(sum(usable_scores) / usable_count, 4) if usable_count else 0.0
    backend_failed = bool(examples) and usable_count == 0 and invocation_error_count == len(examples)
    return {
        "mean_score": mean_score,
        "usable_count": usable_count,
        "invocation_error_count": invocation_error_count,
        "backend_failed": backend_failed,
        "valid_prompt": True,
        "prompt_errors": [],
        "results": results,
    }


def _extract_candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        if "system_prompt" in candidate:
            return str(candidate["system_prompt"])
        if len(candidate) == 1:
            return str(next(iter(candidate.values())))
    return str(candidate)


def _extract_best_val_score(result: Any) -> float | None:
    scores = getattr(result, "val_aggregate_scores", None)
    best_idx = getattr(result, "best_idx", None)
    if isinstance(scores, (list, tuple)) and isinstance(best_idx, int) and 0 <= best_idx < len(scores):
        val = scores[best_idx]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    best_score = getattr(result, "best_score", None)
    if isinstance(best_score, (int, float)) and not isinstance(best_score, bool):
        return float(best_score)
    return None


def _ensure_trailing_newline(text: str) -> str:
    return text if (not text or text.endswith("\n")) else (text + "\n")


def build_prompt_diff(prompt_name: str, baseline_text: str, candidate_text: str) -> str:
    if baseline_text == candidate_text:
        return ""
    return "".join(
        difflib.unified_diff(
            _ensure_trailing_newline(baseline_text).splitlines(keepends=True),
            _ensure_trailing_newline(candidate_text).splitlines(keepends=True),
            fromfile=f"src/mockery/prompts.py::{prompt_name} (baseline)",
            tofile=f"src/mockery/prompts.py::{prompt_name} (gepa_candidate)",
        )
    )


def write_run_artifacts(
    *,
    output_dir: Path,
    prompt_name: str,
    baseline_text: str,
    candidate_text: str,
    report: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline_prompt.txt"
    candidate_path = output_dir / "candidate_prompt.txt"
    diff_path = output_dir / "prompt.diff"
    report_path = output_dir / "report.json"

    diff_text = build_prompt_diff(prompt_name, baseline_text, candidate_text)
    baseline_path.write_text(baseline_text, encoding="utf-8")
    candidate_path.write_text(candidate_text, encoding="utf-8")
    diff_path.write_text(diff_text, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "baseline_prompt": baseline_path,
        "candidate_prompt": candidate_path,
        "prompt_diff": diff_path,
        "report": report_path,
    }


def run_gepa_optimization(
    *,
    target: str,
    seed_prompt: str,
    trainset: list[dict[str, Any]],
    valset: list[dict[str, Any]],
    runner: Callable[[str, str, dict[str, Any]], str],
    reflection_model: Any,
    max_metric_calls: int,
) -> tuple[str, dict[str, Any]]:
    """Run live GEPA optimization via `gepa.optimize_anything` (lazy-imported)."""
    preflight_gepa_dependencies(reflection_model)
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    consecutive_infra_errors = 0

    def evaluator(candidate: Any, example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        nonlocal consecutive_infra_errors
        cand_text = _extract_candidate_text(candidate)
        ok, errors = validate_candidate_prompt(target, cand_text)
        if not ok:
            return 0.0, {
                "example_id": example.get("id"),
                "failures": errors,
                "suggestions": ["Preserve all required runtime `{placeholders}` verbatim without extra braces or unknown keys."],
            }
        sys_prompt = render_system_prompt_for_example(target, example, override_text=cand_text)
        user_prompt = build_user_prompt_for_example(target, example)
        try:
            raw_output = runner(sys_prompt, user_prompt, example)
        except Exception as exc:
            consecutive_infra_errors += 1
            safe_err = _sanitize_error_message(exc)
            if consecutive_infra_errors >= 3:
                raise RuntimeError(
                    f"Aborting GEPA run after {consecutive_infra_errors} consecutive task backend errors: {safe_err}"
                ) from exc
            return 0.0, {
                "example_id": example.get("id"),
                "invocation_error": True,
                "failures": [f"Transient task backend error ({safe_err})"],
                "suggestions": ["Transient infrastructure error; do not change prompt structure for this failure."],
            }
        consecutive_infra_errors = 0
        return score_example(target, raw_output, example)

    spec = get_prompt_spec(TARGET_TO_PROMPT_NAME[target])
    background = (
        f"Prompt name: {spec.name}\n"
        f"Description: {spec.description}\n"
        f"Required placeholders (must stay verbatim): {', '.join(f'{{{k}}}' for k in spec.input_keys)}\n"
        f"Expected output format: {spec.output_format}\n"
        "Target runtime executes on fast local Apple Silicon (oMLX) or low-latency cloud models."
    )

    config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=max_metric_calls),
        reflection=ReflectionConfig(reflection_lm=reflection_model),
    )

    result = optimize_anything(
        seed_candidate={"system_prompt": seed_prompt},
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=TARGET_OBJECTIVES[target],
        background=background,
        config=config,
    )

    best_raw = getattr(result, "best_candidate", None)
    best_text = _extract_candidate_text(best_raw) if best_raw is not None else seed_prompt
    gepa_meta = {
        "best_val_score": _extract_best_val_score(result),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
    }
    return best_text, gepa_meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GEPA prompt evaluation and optimization harness for Mockery.",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(TARGET_TO_PROMPT_NAME.keys()),
        help="Tier 1 prompt target to evaluate or optimize.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline deterministic check using dataset reference/mock outputs (zero network calls, no gepa required).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate baseline prompt against --task-model without running GEPA optimization.",
    )
    parser.add_argument(
        "--mock-responses",
        type=Path,
        default=None,
        help="Optional JSON file mapping example_id -> raw model response for --dry-run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to JSONL dataset (defaults to evals/datasets/<target>.jsonl).",
    )
    parser.add_argument(
        "--task-model",
        default=(
            os.getenv("GEPA_TASK_MODEL")
            or os.getenv("LLM_FAST_MODEL")
            or "mlx-community/Qwen2.5-7B-Instruct-4bit"
        ),
        help="Fast/local task model executed on each dataset example (e.g. oMLX local model).",
    )
    parser.add_argument(
        "--task-base-url",
        default=None,
        help="OpenAI-compatible base URL for task model (defaults to GEPA_TASK_BASE_URL, LLM_BASE_URL, or http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--task-api-key",
        default=None,
        help="API key for the task model endpoint (never inherits unrelated keys when --task-base-url is overridden).",
    )
    parser.add_argument(
        "--reflection-model",
        default=os.getenv("GEPA_REFLECTION_MODEL") or "openai/gpt-5.1",
        help="Frontier reflection LLM identifier passed to GEPA ReflectionConfig.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=40,
        help="GEPA evaluation budget (EngineConfig.max_metric_calls).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write baseline_prompt.txt, candidate_prompt.txt, prompt.diff, and report.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    target: str = args.target
    prompt_name = TARGET_TO_PROMPT_NAME[target]
    spec = get_prompt_spec(prompt_name)
    baseline_text = spec.default_text

    dataset_path = args.dataset or DEFAULT_DATASETS[target]
    rows = load_jsonl_dataset(dataset_path)
    validate_dataset(target, rows)
    trainset, valset = split_dataset(rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (REPO_ROOT / "evals" / "runs" / f"{target}_{timestamp}")

    resolved_base_url: str | None = None
    if args.dry_run:
        mock_map: dict[str, str] | None = None
        if args.mock_responses is not None:
            raw_mock = json.loads(args.mock_responses.read_text(encoding="utf-8"))
            if not isinstance(raw_mock, dict):
                raise ValueError("--mock-responses must be a JSON object mapping example_id -> response string.")
            mock_map = {str(k): str(v) for k, v in raw_mock.items()}
        runner = make_dry_run_runner(mock_map)
        mode = "dry_run"
    else:
        mode = "eval_only" if args.eval_only else "gepa_optimize"
        if mode == "gepa_optimize":
            preflight_gepa_dependencies(args.reflection_model)
        resolved_base_url, resolved_api_key = resolve_task_endpoint_and_key(
            args.task_base_url,
            args.task_api_key,
        )
        runner = make_openai_compatible_task_runner(
            target=target,
            model=args.task_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            temperature=TARGET_TEMPERATURES[target],
        )

    baseline_train = evaluate_candidate_on_examples(
        target=target,
        candidate_text=baseline_text,
        examples=trainset,
        runner=runner,
    )
    baseline_val = evaluate_candidate_on_examples(
        target=target,
        candidate_text=baseline_text,
        examples=valset,
        runner=runner,
    )

    if baseline_train["backend_failed"] or baseline_val["backend_failed"]:
        elapsed_s = round(time.perf_counter() - t0, 3)
        report: dict[str, Any] = {
            "status": "backend_error",
            "target": target,
            "prompt_name": prompt_name,
            "mode": mode,
            "dataset": str(dataset_path),
            "counts": {"total": len(rows), "train": len(trainset), "val": len(valset)},
            "config": {
                "task_model": "dry-run-reference" if args.dry_run else args.task_model,
                "task_base_url": resolved_base_url,
                "reflection_model": None if mode != "gepa_optimize" else args.reflection_model,
                "max_metric_calls": None if mode != "gepa_optimize" else args.max_metric_calls,
            },
            "baseline": {
                "train_mean_score": baseline_train["mean_score"],
                "val_mean_score": baseline_val["mean_score"],
                "train": baseline_train,
                "val": baseline_val,
            },
            "candidate": None,
            "gepa": {},
            "elapsed_seconds": elapsed_s,
        }
        artifacts = write_run_artifacts(
            output_dir=output_dir,
            prompt_name=prompt_name,
            baseline_text=baseline_text,
            candidate_text=baseline_text,
            report=report,
        )
        print(
            json.dumps(
                {
                    "status": "backend_error",
                    "mode": mode,
                    "target": target,
                    "error": "Task backend produced zero usable evaluations on baseline split.",
                    "output_dir": str(output_dir),
                    "artifacts": {k: str(v) for k, v in artifacts.items()},
                    "elapsed_seconds": elapsed_s,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    candidate_text = baseline_text
    candidate_train = baseline_train
    candidate_val = baseline_val
    gepa_meta: dict[str, Any] = {}
    rejected_invalid_candidate = False
    candidate_validation_errors: list[str] = []

    if mode == "gepa_optimize":
        proposed_text, gepa_meta = run_gepa_optimization(
            target=target,
            seed_prompt=baseline_text,
            trainset=trainset,
            valset=valset,
            runner=runner,
            reflection_model=args.reflection_model,
            max_metric_calls=args.max_metric_calls,
        )
        cand_ok, cand_errors = validate_candidate_prompt(target, proposed_text)
        if not cand_ok:
            rejected_invalid_candidate = True
            candidate_validation_errors = cand_errors
        else:
            candidate_text = proposed_text
            candidate_train = evaluate_candidate_on_examples(
                target=target,
                candidate_text=candidate_text,
                examples=trainset,
                runner=runner,
            )
            candidate_val = evaluate_candidate_on_examples(
                target=target,
                candidate_text=candidate_text,
                examples=valset,
                runner=runner,
            )

    elapsed_s = round(time.perf_counter() - t0, 3)
    report = {
        "status": "ok",
        "target": target,
        "prompt_name": prompt_name,
        "mode": mode,
        "dataset": str(dataset_path),
        "counts": {"total": len(rows), "train": len(trainset), "val": len(valset)},
        "config": {
            "task_model": "dry-run-reference" if args.dry_run else args.task_model,
            "task_base_url": resolved_base_url,
            "reflection_model": None if mode != "gepa_optimize" else args.reflection_model,
            "max_metric_calls": None if mode != "gepa_optimize" else args.max_metric_calls,
        },
        "baseline": {
            "train_mean_score": baseline_train["mean_score"],
            "val_mean_score": baseline_val["mean_score"],
            "train": baseline_train,
            "val": baseline_val,
        },
        "candidate": {
            "train_mean_score": candidate_train["mean_score"],
            "val_mean_score": candidate_val["mean_score"],
            "changed": candidate_text != baseline_text,
            "rejected_invalid_candidate": rejected_invalid_candidate,
            "validation_errors": candidate_validation_errors,
            "train": candidate_train,
            "val": candidate_val,
        },
        "gepa": gepa_meta,
        "elapsed_seconds": elapsed_s,
    }

    artifacts = write_run_artifacts(
        output_dir=output_dir,
        prompt_name=prompt_name,
        baseline_text=baseline_text,
        candidate_text=candidate_text,
        report=report,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "mode": mode,
                "target": target,
                "baseline_train_score": baseline_train["mean_score"],
                "baseline_val_score": baseline_val["mean_score"],
                "candidate_train_score": candidate_train["mean_score"],
                "candidate_val_score": candidate_val["mean_score"],
                "rejected_invalid_candidate": rejected_invalid_candidate,
                "output_dir": str(output_dir),
                "artifacts": {k: str(v) for k, v in artifacts.items()},
                "elapsed_seconds": elapsed_s,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
