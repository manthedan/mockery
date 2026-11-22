# Mockery State Charts (Auto-Generated)

Generated directly from `src/mockery/state_machines.py` and `src/mockery/static/state_machines.js`.
Do not edit manually — regenerate with `uv run python scripts/lint_arch.py --write-charts`.

## Backend State Machines (`src/mockery/state_machines.py`)

### `session_phase`

```mermaid
stateDiagram-v2
  [*] --> ready
  ready --> capturing : start_capture
  capturing --> awaiting_self_rating : flush_and_end
  awaiting_self_rating --> capturing : start_capture (resume)
  awaiting_self_rating --> rated : submit_self_rating
  rated --> calibrating : generate_calibration
  calibrating --> calibrated : calibration_ready
  calibrated --> capturing : start_capture (resume)
```

### `vad_turn`

```mermaid
stateDiagram-v2
  [*] --> silence
  silence --> onset
  onset --> silence
  onset --> speaking
  onset --> trailing_pause
  speaking --> silence
  speaking --> trailing_pause
  trailing_pause --> silence
  trailing_pause --> speaking
```

### `asr_model`

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> loading
  idle --> switching
  loading --> error
  loading --> ready
  loading --> switching
  switching --> error
  switching --> ready
  switching --> switching
  ready --> switching
  error --> loading
  error --> switching
```

### `practice_phase`

```mermaid
stateDiagram-v2
  [*] --> active : start_practice
  active --> active : submit_turn / advance_question (has_more)
  active --> awaiting_debrief : advance_question (last)
  active --> debriefing : generate_debrief (early)
  awaiting_debrief --> debriefing : generate_debrief
  debriefing --> completed : debrief_ready
  debriefing --> active : debrief_error (early)
  debriefing --> awaiting_debrief : debrief_error
  completed --> debriefing : recover_debrief
  completed --> completed : submit_retry
```

### `addon_pack`

```mermaid
stateDiagram-v2
  [*] --> not_installed
  not_installed --> installing : install
  installing --> installed : ok
  installing --> error : failed
  error --> installing : retry_install
  installed --> uninstalling : uninstall
  uninstalling --> not_installed : ok
  uninstalling --> error : failed
```

### `diarizer_enrollment`

```mermaid
stateDiagram-v2
  [*] --> unloaded
  unloaded --> cold_start
  cold_start --> manual_calibrated
  cold_start --> single_speaker
  single_speaker --> cold_start
  single_speaker --> manual_calibrated
  single_speaker --> two_speakers
  two_speakers --> cold_start
  two_speakers --> manual_calibrated
  manual_calibrated --> cold_start
  manual_calibrated --> manual_calibrated
```

## Frontend Client State Machines (`src/mockery/static/state_machines.js`)

### `practice_mic`

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> requesting_mic
  idle --> transcribing
  recording --> idle
  recording --> transcribing
  requesting_mic --> idle
  requesting_mic --> recording
  transcribing --> idle
```

### `practice_op`

```mermaid
stateDiagram-v2
  [*] --> idle
  advancing --> exiting
  advancing --> idle
  debriefing --> exiting
  debriefing --> idle
  exiting --> idle
  idle --> advancing
  idle --> debriefing
  idle --> exiting
  idle --> opening
  idle --> requesting_hint
  idle --> starting
  idle --> submitting_retry
  idle --> submitting_turn
  opening --> exiting
  opening --> idle
  requesting_hint --> exiting
  requesting_hint --> idle
  starting --> exiting
  starting --> idle
  submitting_retry --> exiting
  submitting_retry --> idle
  submitting_turn --> exiting
  submitting_turn --> idle
```

### `practice_retry_timer`

```mermaid
stateDiagram-v2
  [*] --> idle
  expired --> idle
  expired --> running
  idle --> running
  running --> expired
  running --> idle
  running --> warning
  warning --> expired
  warning --> idle
```

### `practice_workspace_save`

```mermaid
stateDiagram-v2
  [*] --> saved
  dirty --> saved
  dirty --> saving
  error --> dirty
  error --> saved
  error --> saving
  saved --> dirty
  saved --> saving
  saving --> dirty
  saving --> error
  saving --> saved
```
