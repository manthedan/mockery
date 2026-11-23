# Mockery Feature Map

Shared vocabulary mapping user-facing surfaces to exact DOM landmarks, JS modules, FastAPI routes, and state machines.

---

## 1. Home / Launchpad (`#setupLaunchpad`)
- **Primary modules:** `src/mockery/static/app.js`, `src/mockery/static/practice_studio.js`, `src/mockery/static/styles.css`
- **Mode switch:** `#homeModeSwitch` (AI Mock Practice vs Live Interviewer Copilot), `#homePracticeHistorySection` (`#homePracticeList`), `#homeSessionList`
- **Role & target alignment:** `#setupRoleInput` (`<textarea rows="1">` with newline stripping via `bindRoleHeadline()`), `#homeJdTargetLine` (`#homeApplyJdRoleBtn`), `#setupTypePills`
- **Practice launch controls (`#practiceLaunchRows`, `#practiceLaunchCta`):** `#practiceSourcePills`, `#practiceLevelPills` (role-neutral seniority from `src/mockery/levels.py`: Entry, Mid, Senior, Lead, Manager), `#practiceQuestionPicker` (bank question, auto-select, or "Write my own question…" revealing `#practiceCustomQuestionInput`; `RampartRedactor.strip_contact_details` removes email addresses, phone numbers, and ID or card numbers from it at session start and leaves names and everything else as written; `#practiceKeepQuestionCheckbox` saves it to My questions on start, `#practiceQuestionListDetails` / `#practiceQuestionListInput` / `#practiceSaveQuestionListBtn` add a pasted list, and `#practiceRemoveMyQuestionBtn` removes the saved question that is selected), `#practiceNudgesCheckbox`, `#practiceVoiceCheckbox`, `#practiceVoiceSelect`, `#practicePreviewVoiceBtn`, `#practiceStartBtn`
- **Copilot launch controls (`#copilotLaunchRows`, `#copilotLaunchCta`):** `#setupModePills`, `#setupRubricInput`, `#setupSaveAudioProxy`, `#setupStartBtn`, `#setupSimulateBtn`
- **Backend routes:** `GET /api/status`, `GET /api/practice/sessions`, `POST /api/practice/start`, `GET /api/sessions`

## 2. Onboarding Modal (`#onboardingModal`)
- **Primary modules:** `src/mockery/static/app.js`, `src/mockery/static/profile_addons.js`
- **Key landmarks:** `#onboardingModalTitle`, `.onboard-detected-list` (2-column grid when >=3 local CLIs detected), `#onboardSkipBtn`, `#closeOnboardingBtn`, voice & add-on pack selection
- **Backend routes:** `GET /api/providers/discover`, `POST /api/settings/llm/test`, `POST /api/settings/llm`, `GET /api/addons`, `POST /api/addons/{pack_id}/install`

## 3. Profile, Resume Redaction & Question Banks (`#profileModal`)
- **Primary modules:** `src/mockery/static/profile_addons.js`, `src/mockery/routers/profile.py`, `src/mockery/profiles.py`, `src/mockery/redactor.py`, `src/mockery/banks/__init__.py`
- **Thinking models:** a model that reasons before it answers (common with local servers) can spend a call's whole output allowance on the reasoning and return nothing. `_complete_messages` in `llm.py` reads how much each reply spent on thinking (`_reasoning_tokens_spent`), keeps a per-model allowance (`_reasoning_allowance`, capped by `REASONING_ALLOWANCE_CAP`) that is added to every later call, and asks again with more room when a reply was cut off for that reason.
- **Contact details:** everything a Practice call sends to a model passes through `InterviewCopilotLLM.practice_text_filter` (set in `server.py` to `RampartRedactor.strip_contact_details`), which removes email addresses, phone numbers, Social Security numbers and card numbers. Stored sessions keep what the candidate said.
- **Built-in replies:** when a connected model fails or returns something unusable, Practice substitutes deterministic text (`_fallback` paths in `llm.py`) and says so: the interviewer turn, debrief and retry feedback carry `fallback_reason` (from `built_in_reason`), shown as `.built-in-note` with an Open Settings link plus a toast. No reason is set when no AI is connected. Built-in text praises only what an answer earned (`build_heuristic_nudge`, `said_too_little` in `practice.py`).
- **Answer frameworks (`src/mockery/rubrics.py`):** every answer is graded on four signal slots; the rubric names them. Behavioral rounds choose CARL, STAR or None on the launch form (`#practiceFrameworkRow`, `#practiceFrameworkPills`, stored in `localStorage` as `mockery_practice_framework`, sent as `answer_framework`); General rounds use their own signals (Direct, Specific, Fit, Concise); System design and Coding keep theirs. The choice is saved on the session (`config.answer_framework`; sessions without it are CARL) and returned as `session.rubric`, which the meter, badges, debrief and exported report all read. None stores no signals and hides the meter and score. Grading vocabulary per rubric lives in `prompts.py` (`practice_reply_rubric_args`, `practice_debrief_rubric_args`); the offline scorer in `practice.py` scores each slot by its rubric meaning.
- **Bundled banks (`src/mockery/banks/*.json`, one category each):** `behavioral` (role-neutral), `general` (motivation, fit, and practical questions), `system_design` (per-level depth in `playbook.level_extensions`), `coding` (`difficulty` plus runnable suites verified by `tests/test_bank_reference_solutions.py`). A Practice session draws only from the bank matching its format.
- **My questions (`<data_root>/banks/my_<format>.json`):** the candidate's own saved questions, one hand-editable pack per format, listed first in the Practice picker. A saved question is used only when picked, and is then asked exactly as written. The file is recognised by its name, so editing its contents never turns it into a bank pack; writes take a lock file (`.my_questions.lock`) and re-read a file that changed underneath them. Other `*.json` packs dropped into `<data_root>/banks/` join the curated bank.
- **Key landmarks:** `#profileBtn`, `#homeEditProfileBtn`, `#resumeEmptyState`, `#resumePiiChips`, `#resumeRedactedPreview`, JD list & active selector, Question Banks browser
- **Backend routes (`routers/profile.py`):** `GET /api/profile`, `POST /api/profile/resume/file`, `POST /api/profile/resume/text`, `POST /api/profile/resume/redact`, `POST /api/profile/resume/toggle/{span_id}`, `DELETE /api/profile/resume`, `POST /api/profile/jd`, `POST /api/profile/jd/{jd_id}/select`, `DELETE /api/profile/jd/{jd_id}`, `GET /api/banks`, `POST /api/banks/mine`, `DELETE /api/banks/mine/{category}/{question_id}`
- **State machine:** `ResumeRedactionState` (`src/mockery/state_machines.py`) + `createRedactionQueueCoordinator` (`src/mockery/static/state_machines.js`)

## 4. Settings & Add-On Packs (`#settingsModal`)
- **Primary modules:** `src/mockery/static/app.js`, `src/mockery/static/profile_addons.js`, `src/mockery/routers/addons.py`, `src/mockery/addons.py`, `src/mockery/pyodide_pack.py`
- **Key landmarks:** `#settingsBtn`, `#practiceSettingsBtn`, provider/model config, feature modules, downloadable packs (Phonon, Kokoro, Rampart, Pyodide Python)
- **Backend routes (`routers/addons.py` & `server.py`):** `GET /api/addons`, `POST /api/addons/modules`, `POST /api/addons/{pack_id}/install`, `DELETE /api/addons/{pack_id}`, `GET /api/addons/pyodide/{asset_path}`, `GET /api/tts/status`, `POST /api/tts/speak`, `POST /api/tts/stream`, `POST /api/asr/transcribe`, `POST /api/settings/llm`, `POST /api/open-data-dir`
- **State machine:** `AddonPackState` + `createPackOpTracker`

## 5. AI Mock Practice Studio (`#practiceWorkspace`)
- **Primary modules:** `src/mockery/static/practice_studio.js`, `src/mockery/routers/practice.py`, `src/mockery/practice.py`, `src/mockery/tts.py`, `src/mockery/speech_text.py` (rewrites inline code and signatures into spoken words for the voice only; on-screen text is untouched), `src/mockery/asr.py`
- **Header & rail:** `#practiceHomeBtn`, `#practiceConvToggleBtn`, `#practiceRoleTitle`, `#practiceMetaPill`, `#practiceProgressBadge`, `#practiceWorkspaceSaveStatus`, `#practiceBargeInBtn`, `#practiceVoiceToggleBtn`, `#practiceHintBtn`, `#practiceNextQBtn`, `#practiceDebriefBtn`
- **Conversation & composer (`#practiceActiveView`):** `#practiceFeed`, `#practiceComposerWrap`, `#practiceMicBtn`, `#practiceMicStatus`, `#practiceAnswerInput`, `#practiceSendBtn`, `#practiceCoachMessage`
- **Hands-free answers (default; `#practiceHandsFreeCheckbox` on the launch form, `#practiceAnswerModeBtn` in session, stored in `localStorage` as `mockery_practice_hands_free`):** the dock is the mic alone (`.is-voice-first`); `Space` or `M` starts a take and the next press transcribes and sends it; `#practiceDiscardTakeBtn` / `#practiceRetryDiscardBtn` throw a take away; in Behavioral and General rounds the mic opens by itself when the interviewer finishes speaking. The typing box returns when the mic is refused, the transcriber is in error, or a send was cancelled or failed. A take runs up to 6 minutes and is transcribed in pieces cut at pauses (`src/mockery/static/voice_clip.js`); reaching that limit puts the words in the box and never sends them. Only a press sends a take. "Type instead" (`#practiceAnswerModeBtn`, `#practiceRetryModeBtn`) closes an open or pending mic. A long interviewer message is spoken in sentence-bounded pieces under the voice's limit (`tts.max_chars` in `/api/status`), and the mic opens only after the last one.
- **Split-studio layout:** `#practiceStudioSlot`, `#practiceSplitHandle` (`PointerEvent` drag resize + persisted width), `[data-conv-collapsed]` attribute on `#practiceWorkspace`
- **Backend routes (`routers/practice.py`):** `GET /api/practice/sessions`, `GET /api/practice/sessions/{session_id}`, `POST /api/practice/start`, `POST /api/practice/{session_id}/turn`, `POST /api/practice/{session_id}/hint`, `POST /api/practice/{session_id}/next`, `POST /api/practice/{session_id}/workspace`, `POST /api/practice/{session_id}/debrief`, `POST /api/practice/{session_id}/retry`, `GET /api/practice/{session_id}/markdown`
- **State machines:** `PracticePhase`, `TTSBackendState`, `practice_op`, `practice_mic`, `practice_workspace_save`

## 6. System Design Whiteboard (`#practiceStudioSlot` in `system_design` mode)
- **Primary module:** `src/mockery/static/design_studio.js`
- **Capabilities:** Full-pane SVG dot grid, node palette (`SVC`, `DB`, `CACHE`, etc.), orthogonal edge routing, inline inspector (`[data-sd-edit-label]`, `[data-sd-edit-note]`, `[data-sd-edit-edge-label]`), keyboard delete (`Backspace`/`Delete` after focus handoff), undo (`[data-sd-undo]`), debounced workspace autosave to `POST /api/practice/{session_id}/workspace`

## 7. Coding Studio (`#practiceStudioSlot` in `coding` mode)
- **Primary modules:** `src/mockery/static/coding_studio.js`, `src/mockery/static/js_sandbox_worker.js`, `src/mockery/static/py_sandbox_worker.js`
- **Capabilities:** Language tabs (`[data-cs-lang="javascript"|"python"|"pseudocode"]`), Tab indent / Esc→Tab keyboard escape, isolated Web Worker test execution (JS built-in; Python via Pyodide pack at `/api/addons/pyodide/{asset_path}`), autosave to `POST /api/practice/{session_id}/workspace`

## 8. Coaching Debrief & 45-Second Retry Drill (`#practiceDebriefView`)
- **Primary modules:** `src/mockery/static/practice_studio.js`, `src/mockery/llm.py` (`generate_practice_debrief`, `evaluate_practice_retry`)
- **Key landmarks:** `#practiceDebriefContent`, `#practiceRetryCard`, `#practiceRetryHead`, `#practiceRetryClock`, `#practiceRetryTimerBtn`, `#practiceRetryPrompt`, `#practiceRetryRep`, `#practiceRetryMicBtn`, `#practiceRetryInput`, `#practiceRetrySubmitBtn`
- **Backend routes:** `POST /api/practice/{session_id}/debrief`, `POST /api/practice/{session_id}/retry`, `GET /api/practice/{session_id}/markdown`
- **State machines:** `PracticePhase` (`debriefing` → `completed`), `practice_retry_timer`

## 9. Live Interviewer Copilot (`#workspaceGrid`)
- **Primary modules:** `src/mockery/static/app.js`, `src/mockery/grouper.py`, `src/mockery/diarizer.py`, `src/mockery/vad.py`, `src/mockery/sources.py`
- **Backend routes:** `WS /ws/session`, `POST /api/sessions/{session_id}/load`, `GET /api/sessions/{session_id}/markdown`
- **State machines:** `SessionPhase`, `CardProbeState`, `VADTurnState`, `ASRModelState`, `DiarizerEnrollmentState`

## 10. Global Notifications
- **Landmark:** `#toastBanner` (`.toast`, anchored at `top: 62px; pointer-events: none` so it never blocks bottom composers or footers)
