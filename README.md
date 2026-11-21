# Mockery

**A calm, local-first mock interview studio for Apple Silicon Macs.**

Mockery listens to your mock interview in real time — running **100% offline speech recognition and speaker separation directly on your Mac** — organizes the conversation into clean question/phase cards, surfaces sharp follow-up probes while the candidate is still talking, and runs a **blind post-interview calibration audit** against your own gut verdict.

- **Private by default:** Speech-to-text ([Parakeet TDT 0.6B v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) on MLX) and voice separation ([WeSpeaker](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) on CoreML) run locally on your Mac. Audio never leaves your machine.
- **Zero-friction setup:** Works out of the box in **Transcript-Only mode** with no API key, or connects in one click to a provider you already have — **Claude Code CLI**, **Codex CLI**, **Gemini CLI**, **oMLX / Ollama / LM Studio**, or a **Google Gemini (free tier)**, **Anthropic**, **OpenAI**, **DeepSeek**, or **OpenRouter** API key.
- **Two capture modes:**
  - **Google Meet / Remote:** Captures your mic (`You`) and your Chrome Meet tab (`Candidate`) on a shared audio clock with automatic echo-bleed suppression.
  - **In-Person (Single Mic):** Acoustic two-speaker diarization separates voices automatically in the same room.

---

## Quick Start (macOS Apple Silicon)

> **Requirements:** A Mac with Apple Silicon (M1, M2, M3, or M4) and Google Chrome (recommended for Google Meet tab audio).

### Option 1 — One-Line Installer (Installs `Mockery.app` in Applications)

Open **Terminal** and run:

```bash
curl -fsSL https://raw.githubusercontent.com/manthedan/mockery/main/install.sh | sh
```

Then open **Mockery** from **Applications** or **Spotlight** (⌘ Space → `Mockery`).

- Mockery sits quietly in your menu bar and opens the local studio at `http://127.0.0.1:8765`.
- On your very first launch, Mockery downloads the offline speech engine (~600 MB) once — after that, transcription works 100% offline.
- A guided welcome screen helps you connect an AI assistant in seconds (or skip straight to Transcript-Only mode).

### Option 2 — Run with `uv` (Developers)

```bash
git clone https://github.com/manthedan/mockery.git
cd mockery
uv run mockery
```

To build a double-clickable `.app` or `.dmg` locally:

```bash
uv run python scripts/build_macos_app.py --install   # installs /Applications/Mockery.app
uv run python scripts/build_macos_app.py --dmg       # creates dist/Mockery-macOS-arm64.dmg
```

---

## Connecting an AI Assistant (Optional)

Speech transcription, speaker separation, cards, talk ratios, and Markdown exports all run **100% locally with zero API key**.

To enable **live follow-up probes** and **blind calibration**, Mockery's onboarding wizard & **Settings** support:

| Option | Cost / Auth | Best For |
| --- | --- | --- |
| **Detected Local CLIs** (`claude`, `codex`, `gemini`, `agy`, `pi`) | Existing subscription — **no API key** | Auto-detected if installed on your Mac (one click) |
| **Google Gemini API** | Generous free tier via [Google AI Studio](https://aistudio.google.com/apikey) | Easiest cloud setup without a credit card |
| **Anthropic Claude / OpenAI / DeepSeek / OpenRouter** | Bring your own API key | Favorite frontier or budget cloud models |
| **[oMLX](https://github.com/jundot/omlx) / Ollama / LM Studio** | **100% offline & free** | Private local LLMs on Apple Silicon (auto-lists your loaded models; 16 GB+ Mac recommended) |
| **Transcript-Only** | Free, zero setup | Pure local transcription, notes, and Markdown export |

Use **Settings → Test connection** at any time to verify your setup before an interview.

---

## How a Session Works

1. **Pick role, format, and audio mode** (*Behavioral*, *System design*, *Coding*, or *General*; *Google Meet* or *One microphone*).
2. **Start interview:** Live draft transcripts stream at ~50 ms latency while final turns lock in with full-context accuracy.
3. **Live follow-ups (`D`):** Mockery speculatively pre-warms 3 tailored follow-up probes after substantive candidate answers so pressing <kbd>D</kbd> reveals them instantaneously. Press <kbd>1</kbd>–<kbd>3</kbd> to check off probes as you ask them, or <kbd>N</kbd> to start a new card.
4. **Blind calibration:** Before revealing the AI debrief, Mockery asks you to lock in your own **Hire Signal**, **Level**, and **Gut Rationale**. Then it streams a timestamp-cited audit (`[MM:SS]`) comparing your call against what was actually said.
5. **Export:** Download an **Interviewer Debrief** or a scrubbed **Candidate Copy** in clean Markdown, or click **Settings → Reveal in Finder** to open your saved sessions folder.

---

## Speech Language Settings

Mockery defaults to **`parakeet-tdt-0.6b-v2`** for the fastest, cleanest English transcription. Need another language? Open **Settings → Speech model** and switch to **`parakeet-tdt-0.6b-v3`** (25 European languages).

---

## Security & Privacy

- Mockery binds strictly to `127.0.0.1` by default and verifies same-origin requests.
- If you intentionally expose it on a local network (`mockery --host 0.0.0.0`), Mockery automatically requires a cryptographically random access token (or `--lan-token <secret>`).
- API keys are saved locally with `0600` file permissions in `.env.local` (never transmitted anywhere except to your chosen provider).
- Saved transcripts live in `~/Library/Application Support/Mockery/sessions` (or `./sessions` when running inside a local git checkout).

---

## Development & Testing

```bash
uv run pytest -v
```

State machine introspection (CLI, HTTP, and browser):

```bash
uv run mockery --dump-machines   # JSON + Mermaid diagrams
# HTTP: GET /api/machines
# Browser console: window.__MOCKERY_STATE__
```

---

## License

[MIT](LICENSE)
