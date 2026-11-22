# Mockery v0.2 launch film: pre-production brief

Status: blueprint only. Nothing here has been shot or rendered.

| | |
| --- | --- |
| **Runtime** | 58 s master. A 45 s cutdown is specified at the end. |
| **One line** | Practice the interview out loud. Then run one for someone else. |
| **Structure** | Two acts, two voices. Act 1 is the candidate's side in Dark. Act 2 is the interviewer's side in Light. |
| **Deliverables** | 16:9 master (3840×2160, 60 fps, H.264 + ProRes), 1:1 and 9:16 crops, burned-in captions on all three, `.srt` for the master. |
| **Where it runs** | README top, release site hero, GitHub release notes, social posts. Assume it autoplays muted. |

## Creative direction

The film should feel like the product: calm, editorial, nothing glowing.

- **Captions carry the story.** Every beat reads with the sound off.
- **The only voices are the ones in the room.** No narrator by default. You hear the interviewer ask, the candidate answer, and room tone. An optional narrator script is included for a voiced variant.
- **Two voices, two weights.** Lines for the candidate's side are set in full ink. Lines for the interviewer's side are set in the muted ink. This is the same idea as the two quote marks in the logo.
- **Yellow means evidence.** The marker highlight appears twice only: on "out loud" in the opening card and on the cited transcript line in beat 8.
- **Movement.** Slow push-ins (100% → 106%) and hard cuts. No zoom whips, no 3D device tilts, no cursor trails, no particle or gradient backgrounds, no risers.

### Type and colour (the app's tokens, as used in `site/styles.css`)

| Use | Dark (Act 1) | Light (Act 2) |
| --- | --- | --- |
| Card background | `#141413` | `#fbfaf7` |
| Caption, primary voice | `#efeeea` | `#1c1b18` |
| Caption, second voice | `#939189` | `#716e67` |
| Small labels | `#b4b2ab` | `#55534c` |
| Marker highlight | `rgba(244, 196, 64, 0.26)` | `rgba(250, 206, 62, 0.5)` |

- Captions: serif, the family used for spoken words in the app (`"New York", ui-serif, "Iowan Old Style", Palatino, Georgia`). Newsreader (OFL) is the fallback if the edit machine has no New York. Sentence case, no all-caps, no drop shadow.
- Labels and keycaps: system sans (SF Pro).
- Caption position: lower third, left-aligned to the app's content margin, on a solid card-coloured strip so it never sits on top of UI text.
- Brand files: `docs/assets/wordmark-on-dark.svg`, `docs/assets/wordmark-on-light.svg`, `docs/assets/logo-mark.svg`.

## Stage the demo

Everything on screen is synthetic: Jordan Vance, FinScale Payments, a Pied Piper job listing, and a scripted webhook-design interview. Never record against your real Mockery data.

```bash
# 1. Serve the seeded demo studio (throwaway data dir, deleted on Ctrl-C).
uv run python docs/marketing/render_assets.py stage          # http://127.0.0.1:8765

# 2. Open it in a clean, chrome-less window with an empty browser profile.
open -na "Google Chrome" --args --app=http://127.0.0.1:8765 \
  --window-size=1440,900 --user-data-dir="$TMPDIR/mockery-shoot"
```

`stage` seeds the same data as `scripts/verify_ui.py` (`seed_demo_data_dir`) plus the Copilot session defined in `docs/marketing/render_assets.py`. It refuses to start if port 8765 is taken, so quit your own Mockery first.

| Seeded item | ID | Used in |
| --- | --- | --- |
| Behavioral session, question 2 of 3 | `prac_beh_active` | Beat 1 |
| Completed debrief with retry drill | `prac_debrief_retry` | Beat 2 |
| System design whiteboard | `prac_sd_active` | Beat 3 |
| Coding studio, `TTLLRUCache` | `prac_code_active` | Beat 4 |
| Candidate profile, `Jordan_Vance_Resume.pdf` | profile modal | Beat 5 |
| Copilot review, 3 cards, 24 min | `20261120_101500_fin2` | Beats 7 and 8 |

**No model is needed for any scripted action.** Opening sessions, starting the retry timer, dragging a node, running the JavaScript tests, flipping a redaction chip, switching Copilot tabs and clicking a citation were all dry-run against this data with no reachable model. Do not press Send, Speak, Evaluate retry or Ask AI to review during a take; those call a model and their output is not scripted.

### Capture settings

- Window 1440×900 CSS px on a Retina display (2880×1800 pixels), 60 fps, cursor visible at default size.
- macOS: Do Not Disturb on, menu bar and Dock hidden, wallpaper irrelevant (record the window only).
- Theme: Mockery follows the macOS appearance. Shoot beats 1–5 in Dark and beats 7–8 in Light. Switch appearance between acts, then reload the window.
- Record each beat as its own clip with two seconds of stillness at head and tail.
- For frame-identical retakes, drive the clicks through `scripts/verify_ui.py`'s `CDPClient` using the hooks listed per beat. `capture_shots()` in `docs/marketing/render_assets.py` is a working example.

## Storyboard

Timings are for the 58 s master. "Hook" is the element to click or the selector to drive.

### Act 1: the candidate (Dark)

#### Beat 0 · 0:00–0:04 · Opening card

- **Screen:** `#141413` card. Line appears word by word: **Practice the interview out loud.** The marker highlight wipes in behind "out loud" on the last word.
- **Audio:** Room tone. A breath, then Jordan mid-sentence, low in the mix: "…so I audited three teams' retry handlers…"
- **Caption:** the card text is the caption.
- **Narrator (optional):** "Say it out loud first."

#### Beat 1 · 0:04–0:12 · Behavioral, live answer

- **Scenario:** Home → AI Mock Practice → open `prac_beh_active`.
- **Hook:** `[data-open-practice="prac_beh_active"]`
- **Screen:** Header reads "Staff Backend Engineer — Core Payments Ledger · Pied Piper", badge "Q2 of 3". The interviewer's question in serif, then Jordan's answer with its CARL chips: C and A green, R amber, L empty. Push in slowly on the chips. End on the "Speak answer" button without clicking it.
- **Audio:** Interviewer voice (the app's local voice, recorded separately): "Tell me about a time you had to drive cross-team alignment on a technical direction." Then Jordan: "I audited three teams' retry handlers and built a two-day drop-in adapter, so all three adopted deterministic idempotency."
- **Caption:** Answer out loud. See what landed.
- **Narrator (optional):** "Mockery asks. You answer, in your own voice."

#### Beat 2 · 0:12–0:21 · Debrief and 45-second retry

- **Scenario:** Home → open `prac_debrief_retry` → jump to the retry drill → start the timer.
- **Hooks:** `[data-open-practice="prac_debrief_retry"]`, `[data-jump-retry]`, `#practiceRetryTimerBtn`
- **Screen:** Debrief view. Hold on Q2 "Pushing Back on an Aggressive Launch Date" with R amber and L empty, and the line "Quantify prevented exposure and add one process takeaway." Click "Jump to 45s retry drill ↓". The drill card shows "Quantify risk saved & add an org-level Learning" and a clock at 0:45. Click "Start timer". Hold while it counts 0:45 → 0:41.
- **Audio:** Soft click on the timer. Jordan, steadier than in beat 1: "Capping the rollout kept one point four million dollars of modeled exposure off the table, and we cleared forty-two million in clean volume that weekend. Every launch kickoff now starts with that checklist."
- **Caption:** One retry. Forty-five seconds. While the feedback is fresh.
- **Narrator (optional):** "Then one more try at the part that needed it."

#### Beat 3 · 0:21–0:27 · System design whiteboard

- **Scenario:** Home → open `prac_sd_active`.
- **Hook:** `[data-open-practice="prac_sd_active"]`
- **Screen:** The whiteboard: Merchant SDKs → Global API Gateway → Ledger Core API → Spanner Ledger DB and Settlement Stream, with labelled edges (HTTPS, gRPC, ACID commit, Outbox event). Drag "Settlement Stream" about 40 px down so its edge re-routes. Requirements counter reads 1/3.
- **Audio:** Jordan: "Every mutation carries an idempotency key, committed with the ledger rows and an outbox record."
- **Caption:** Draw the system while you talk.
- **Narrator (optional):** "Sketch the design. Get asked about what you drew."

#### Beat 4 · 0:27–0:33 · Coding studio

- **Scenario:** Home → open `prac_code_active` → run the tests.
- **Hooks:** `[data-open-practice="prac_code_active"]`, `[data-cs-run]` (or ⌘↵ in the editor)
- **Screen:** `TTLLRUCache` in JavaScript. Press ⌘↵. Test results show "All assertions passed" and "2/2 assertions passed". Show a ⌘↵ keycap overlay for one second.
- **Audio:** Two key taps. No voice.
- **Caption:** Run real tests, right on your Mac.
- **Narrator (optional):** "Write it, run it, explain it."

#### Beat 5 · 0:33–0:38 · Resume privacy redaction

- **Scenario:** Home → Profile. Before rolling, click the Name chip once so it reads Visible.
- **Hooks:** `#profileBtn`, `#resumePiiChips [data-toggle-span-id="pii_0"]`
- **Screen:** "Your resume stays on this Mac". Under "What leaves your Mac", the preview starts with "Jordan Vance". Click the Name chip: it flips to Hidden and the name in the preview becomes a neutral `Name` placeholder beside `Email`, `Phone` and `URL`.
- **Audio:** One soft click. Room tone.
- **Caption:** Your resume is redacted on your Mac before any model reads it.
- **Narrator (optional):** "What leaves your Mac is your call."

### The turn

#### Beat 6 · 0:38–0:41 · Second card

- **Screen:** Hard cut from the dark app to a `#fbfaf7` card. Line in the muted second-voice ink: **Then run one for someone else.**
- **Audio:** Room tone changes: a second, closer voice clears their throat. This is the interviewer's side now.
- **Caption:** the card text is the caption.
- **Narrator (optional):** "Then sit on the other side of the table."

### Act 2: the interviewer (Light)

#### Beat 7 · 0:41–0:48 · Live follow-up probes

- **Scenario:** Home → Interviewer Copilot → open "Senior Backend Engineer (L5) · System Design · 3 cards · 24m" → card 3 → Follow-ups.
- **Hooks:** `#homeModeSwitch` (Interviewer Copilot), `[data-load-session="20261120_101500_fin2"]`, `[data-jump-qa="qa_p3"]`, `#tabProbes`
- **Screen:** Transcript on the left with You and Candidate turns. Card 3 "Dead-Letter Queue & Replay" on the right with three follow-ups. Push in on the first: "Size the dead-letter store for a three-day outage at forty thousand events a second. Where does it live?" Show a `D` keycap overlay as the panel settles.
- **Audio:** Candidate (a different voice from Jordan): "Honestly, I haven't sized the storage for a multi-day outage. I'd want to model that." Beat. Interviewer, reading the probe: "So size it for three days. Where does it live?"
- **Caption:** Three follow-ups, grounded in what was just said.
- **Narrator (optional):** "While they talk, Mockery listens on your Mac and drafts the next question."

#### Beat 8 · 0:48–0:54 · Blind calibration

- **Scenario:** Same session → Calibration → click the first citation.
- **Hooks:** `#tabCalibration`, `.cite-link[data-cite-utt="utt_02"]`
- **Screen:** The audit opens with the blind call and "supports **Lean Hire for Senior**". Card grades: P1 Strong, P2 Strong, P3 Mixed. Click the `00:24` citation: the transcript jumps to the candidate's scoping question and marks it.
- **Audio:** One soft click. No voice.
- **Caption:** Commit to your call first. Then check it against the transcript.
- **Narrator (optional):** "Make your call first. Then see what the transcript says."

#### Beat 9 · 0:54–0:58 · End card

- **Screen:** Light card. Wordmark (`wordmark-on-light.svg`), then:
  - A calm, local-first mock interview studio for Apple Silicon.
  - `curl -fsSL https://raw.githubusercontent.com/manthedan/mockery/main/install.sh | sh`
  - github.com/manthedan/mockery
  - Speech stays on your Mac · v0.2.0 · MIT
- **Audio:** Room tone out over the last second. No sting.
- **Narrator (optional):** "Mockery. It all runs on your Mac."

## Audio plan

| Element | Source | Notes |
| --- | --- | --- |
| Interviewer voice, Act 1 | The app's own local voice, recorded from Practice (voice preview or the Replay control) | Use Kokoro or Phonon if a pack is installed, otherwise the built-in macOS voice. It is the product's real sound. |
| Jordan (candidate, Act 1) | Voice actor or a volunteer reading the lines above | Lines are scripted and fictional. Do not improvise real work history. |
| Interviewer and candidate, Act 2 | Two more readers | Two clearly different timbres, to sell "two voices, kept apart". |
| Room tone | Recorded on set | Quiet room, slightly different tone per act. |
| Music | None in the master | If a platform needs it: one felt piano or a single sustained pad, under −24 LUFS, no build. |
| UI sounds | Real key and trackpad clicks only | No synthetic whooshes. |

Mix dialogue to −16 LUFS integrated. Every spoken line also appears in the `.srt`.

## Caption sheet

| Beat | In | Out | Caption |
| --- | --- | --- | --- |
| 0 | 0:00 | 0:04 | Practice the interview out loud. |
| 1 | 0:05 | 0:12 | Answer out loud. See what landed. |
| 2 | 0:13 | 0:21 | One retry. Forty-five seconds. While the feedback is fresh. |
| 3 | 0:22 | 0:27 | Draw the system while you talk. |
| 4 | 0:28 | 0:33 | Run real tests, right on your Mac. |
| 5 | 0:34 | 0:38 | Your resume is redacted on your Mac before any model reads it. |
| 6 | 0:38 | 0:41 | Then run one for someone else. |
| 7 | 0:42 | 0:48 | Three follow-ups, grounded in what was just said. |
| 8 | 0:49 | 0:54 | Commit to your call first. Then check it against the transcript. |
| 9 | 0:54 | 0:58 | A calm, local-first mock interview studio for Apple Silicon. |

## 45-second cutdown

| Change | Saves |
| --- | --- |
| Drop beat 4 (Coding studio) | 6 s |
| Beat 2: cut straight to the retry card, hold the clock for 2 s | 3 s |
| Beat 1: start on Jordan's answer, interviewer question as caption only | 2 s |
| Beat 7: drop the interviewer's spoken probe | 2 s |

Result: 45 s. Beats 0, 6 and 9 are never shortened.

## Crops

- **1:1** Centre on the conversation column for beats 1, 2, 5, 7 and 8, and on the canvas for beat 3. Captions move to a strip below the frame.
- **9:16** Stack: app crop on top (same focal areas), caption strip beneath at 2× size. Beat 4 is dropped in vertical; code is unreadable at that width.

## Pre-flight checklist

- [ ] `render_assets.py stage` is the instance in the window. The home screen lists only the seeded sessions.
- [ ] Fresh Chrome profile: no extensions, bookmarks, autofill or account avatar.
- [ ] Do Not Disturb on. No real names, emails, calendar popups or Meet calls anywhere in frame.
- [ ] Retina capture confirmed at 2880×1800 before the first take.
- [ ] No take includes a model-generated response.
- [x] Synthetic target employer uses Pied Piper (no real employer names on screen).
- [ ] Captions checked against the caption sheet. Install command checked character by character against `README.md`.
- [ ] Ctrl-C the stage server when done. It deletes its data directory.
