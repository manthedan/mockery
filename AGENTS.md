# Mockery Agent Guidelines

Keep rules deterministic in code/linters; this file holds only high-level workflow and judgment calls.

## Inner Loop Verification (Run Before Committing)
- Fast gate (`ruff`, architecture/DOM-ID/state-chart lint, JS syntax, unit + FSM + fuzz tests):
  ```bash
  uv run python scripts/check.py
  ```
- Browser UAT + Bombadil-style property-based UI explorer (when touching `src/mockery/static/` or API contracts):
  ```bash
  uv run python scripts/check.py --ui
  ```
- Regenerate visual state charts after changing `state_machines.py` or `state_machines.js`:
  ```bash
  uv run python scripts/lint_arch.py --write-charts
  ```

## Shared Navigation Artifacts
- Feature/UI vocabulary & route map: `docs/feature-map.md`
- Auto-generated FSM state charts: `docs/state-charts.md`

## Core Judgment Rules
- **Local-first privacy:** Never send unredacted resume PII to external LLMs; keep all tests, UAT seeds, and eval fixtures strictly synthetic.
- **No workaround comments:** Fix root causes or add a deterministic rule in `scripts/lint_arch.py` instead of writing excuse comments.
- **Prompts stay deliberate:** Only modify `src/mockery/prompts.py` when explicitly requested or backed by held-out eval gains in `evals/`.
- **Trunk-based default:** Commit clean, verified milestones directly to `main` unless explicitly asked for a branch or PR.
