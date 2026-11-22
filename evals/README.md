# Offline Prompt Evaluation & GEPA Tuning Harness

> **Dev-only boundary:** Nothing under `evals/` or the optional `gepa` dependency is ever imported by runtime code in `src/mockery/`. All runtime prompts live cleanly in [`src/mockery/prompts.py`](../src/mockery/prompts.py).

## Tier 1 Optimization Targets

| Target (`--target`) | Prompt Registry Key | Why Optimize Offline with GEPA |
| --- | --- | --- |
| `triage_window` | `triage_window` | Latency-critical JSON structured output running on small/local models: speaker diarization, `<700ms` mixed-speaker handoff splitting, STT typo cleanup without paraphrasing, and card boundary accuracy without over-splitting. |
| `practice_interviewer` | `practice_interviewer_reply` | Live AI practice interviewer JSON contract: concise spoken replies (1–3 sentences), workspace grounding (whiteboard/code/tests), `should_advance` pacing, monotonic CARL signals, supportive tone, and zero coding spoilers. |

## 1. Offline Smoke Check (`--dry-run`, zero API keys / zero `gepa` needed)

Verify the synthetic datasets, prompt rendering, deterministic metrics, and artifact output completely offline:

```bash
uv run python -m evals.optimize_gepa --target triage_window --dry-run
uv run python -m evals.optimize_gepa --target practice_interviewer --dry-run
```

## 2. Evaluate Baseline Against Local oMLX / Ollama (`--eval-only`)

Measure your current local model (for example, oMLX on `http://localhost:8000/v1`) before spending reflection budget:

```bash
uv run python -m evals.optimize_gepa \
  --target triage_window \
  --eval-only \
  --task-model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --task-base-url http://localhost:8000/v1
```

## 3. Run GEPA Reflective Prompt Optimization

Install the optional dev-only `gepa` extra first (installs both `gepa` and `litellm` for string-valued reflection models without adding either to the default runtime install):

```bash
uv sync --extra gepa
```

Then pair your fast local `task_lm` (oMLX / Ollama / LM Studio / DeepSeek Flash, OR Mockery's `cli://` transports such as `cli://claude`, `cli://codex`, `cli://agy`) with a frontier `reflection_lm` (either a LiteLLM string like `openai/gpt-5.1` or a zero-API-key local CLI transport like `cli://claude/sonnet`, `cli://codex/gpt-6.1-sol`, `cli://agy/gemini-4-neon:high`):

```bash
# Via local HTTP server (oMLX) + LiteLLM reflection model:
uv run python -m evals.optimize_gepa \
  --target practice_interviewer \
  --task-model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --task-base-url http://localhost:8000/v1 \
  --reflection-model openai/gpt-5.1 \
  --max-metric-calls 40

# Or via Mockery's cli:// transports (zero API keys, no LiteLLM required):
uv run python -m evals.optimize_gepa \
  --target practice_interviewer \
  --task-base-url cli://claude \
  --task-model haiku \
  --reflection-model cli://claude/sonnet \
  --max-metric-calls 32
```

## 4. Reviewing & Promoting Candidate Prompts

The harness **never** silently mutates `src/mockery/prompts.py`. Every run writes four artifacts under `evals/runs/<target>_<timestamp>/` (gitignored):

- `baseline_prompt.txt` — exact seed prompt from `PROMPT_REGISTRY`
- `candidate_prompt.txt` — best GEPA candidate (verified to preserve all required `{placeholders}`; invalid candidates are rejected back to baseline)
- `prompt.diff` — unified diff between baseline and candidate
- `report.json` — train/val mean scores, per-example subscores, and Actionable Side Information (ASI) failure traces

To promote a winning candidate:

1. Inspect `report.json` to confirm `val_mean_score` improved without overfitting `train_mean_score`.
2. Review `prompt.diff` for tone, safety, and placeholder preservation.
3. Update `default_text` in [`src/mockery/prompts.py`](../src/mockery/prompts.py) and re-run `uv run pytest -v`.

## Privacy Rule

- All checked-in datasets under `evals/datasets/` are strictly synthetic and PII-free. Never commit real resumes, personal emails, phone numbers, or private interview transcripts into `evals/datasets/`.
- When passing a custom `--dataset`, remember that example transcripts and workspace snippets are sent to `--task-base-url` (and diagnostic failure summaries are sent to `--reflection-model` during GEPA reflection). Always redact PII before evaluating against remote endpoints, or run against a local loopback server (`http://localhost:8000/v1`).
