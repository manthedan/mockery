# Mock Helper — Local Apple Silicon Mock Interview Copilot

A local-first 45-minute mock interview assistant for macOS (Apple Silicon), built to sharpen **live probing depth** and **post-interview calibration**.

---

## Core Architecture

1. **Consent-First Capture & Shared Sample Clock:**
   - Explicit pre-capture **Candidate Consent Gate** (with a 1-click verbal consent script, headphones reminder, optional local 16kHz WAV archiving for offline re-transcription, and a clean **Candidate Copy `.md`** export).
   - **Google Meet Mode (Phase 1):** Routes your **Mic (`Interviewer`)** and isolated **Google Meet Chrome Tab Audio (`Candidate`)** through a **single shared `AudioContext`** (`AudioWorklet` downsampling to 16kHz PCM) with cross-channel acoustic echo-bleed suppression if speaker audio leaks into the mic.
   - **Single-Mic Seam (Phase 2 Ready):** Routes room audio through a `SpeakerDiarizer` protocol (`src/mock_helper/sources.py`) ready for drop-in **NVIDIA Nemotron 3 Diarization** or **Streaming Sortformer**—using `(id, rev)` utterance upserts so delayed speaker-label corrections update turns cleanly in place.
2. **Upsertable Canonical Utterance Log (`id`, `rev`, `final`) + Derived Cards:**
   - Every utterance is stored in an append-only / revision-upsertable log (`InterviewSession.utterances` + crash-safe `sessions/<id>.events.jsonl`), while cards hold `utterance_ids` as a non-destructive derived view.
   - Draft partials (`final=False`) stream directly into the active card so **Dive Deeper sees an answer while it is still in progress**.
   - **Behavioral Mode** uses Question Cards (`Q1..Qn`); **System Design & Coding Modes** use Phase Cards (`P1..Pn`). Includes `<kbd>N</kbd>` manual card split, 1-click merge, and full-context end-of-session **Regroup**.
3. **Speculative Pre-Warmed "Dive Deeper" (`deepseek-flash` non-thinking):**
   - Whenever a substantive candidate turn (`>= 25 words`) finishes, Mock Helper speculatively pre-generates follow-up probes in the background (automatically passing `thinking: {"type": "disabled"}` for `deepseek-flash` and filtering out already-asked questions). Pressing <kbd>D</kbd> displays cached probes in **0ms**.
4. **Blind Self-Calibration Before AI Reveal:**
   - Before showing the AI's read, the Final Calibration tab prompts you to lock in **your own Hire Signal, Perceived Level, and Gut Rationale**.
   - Talk-to-listen time/word ratios and per-card durations are computed deterministically in Python from utterance timestamps, and every LLM claim is required to cite `[MM:SS, utt_id]`.

---

## Quick Start

```bash
uv run mock-helper
```

Opens `http://127.0.0.1:8765` (`mlx-community/parakeet-tdt-0.6b-v3` is already pre-cached on this Mac; you can A/B switch to `parakeet-tdt-0.6b-v2` in **Settings**).

## State Machine Introspection (DX & AX)

All session, per-card probe, VAD speech-turn, ASR checkpoint, and background job lifecycles are formalized in `src/mock_helper/state_machines.py` and published live on every `/api/status` and WebSocket `session_state` payload (`session.state_machines`):

- **CLI topology & Mermaid export:** `uv run mock-helper --dump-machines`
- **HTTP introspection:** `GET /api/machines`
- **Browser / Agent DOM inspection:** `window.__MOCK_HELPER_STATE__`

## Running Tests

```bash
uv run pytest -v
```
