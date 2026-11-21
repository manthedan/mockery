#!/usr/bin/env python3
"""
Reusable YouTube Mock-Interview Eval & Cached-Trace Replay Harness for mockery.

Inspired by `scratch/triage-lab`:
1. Caches VAD + Parakeet-MLX ASR segments and YouTube captions per video.
2. Feeds all raw single-mic segments into `InterviewSession` + `QAGrouper` (blind single-mic
   simulation where the interviewer never touches Tab) and runs sliding-window `triage_window`
   + deterministic confidence-gated `apply_turn_triage`.
3. Persists raw per-window LLM traces in `runs/<slug>/window_NN.json` so deterministic applier
   changes can be iterated in milliseconds via `--replay` without re-calling the LLM.
4. Optionally runs live `suggest_probes` on each resulting Q&A card (`--probes`) and scores
   a 100-point rubric written to `runs/<slug>/SUMMARY.md`.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mockery.asr import ParakeetEngine
from mockery.grouper import QAGrouper
from mockery.llm import InterviewCopilotLLM, LLMRuntimeConfig
from mockery.models import (
    LLM_PRESETS,
    InterviewSession,
    SessionConfig,
    TurnTriagePlan,
    Utterance,
    fmt_ms,
)
from mockery.state_machines import validate_session_invariants
from mockery.vad import ChannelVADSegmenter, SileroVADRuntime

KNOWN_VIDEO_CHAPTERS: dict[str, dict[str, Any]] = {
    "-Y6YmuqGLpY": {
        "role_title": "Forward Deployed Engineer (Rippling Behavioral Mock)",
        "interview_type": "behavioral",
        "chapters": [
            {"ms": 0, "title": "Intro"},
            {"ms": 45_000, "title": "Pushing back on clients"},
            {"ms": 373_000, "title": "Solving technical problems"},
            {"ms": 654_000, "title": "Cross-functional collaboration"},
        ],
    }
}


def _slugify(val: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", val).strip("_") or "run"


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ensure_audio_and_asr(
    *,
    video_id: str,
    cache_dir: Path,
    max_seconds: int,
    parakeet_model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seg_path = cache_dir / "parakeet_segments.json"
    yt_path = cache_dir / "yt_transcript.json"
    wav_path = cache_dir / "interview_16k.wav"

    yt_cues: list[dict[str, Any]] = []
    if yt_path.exists():
        yt_cues = json.loads(yt_path.read_text(encoding="utf-8"))
    else:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

            raw_cues = YouTubeTranscriptApi().fetch(video_id)
            yt_cues = [
                {
                    "start": float(c.start),
                    "duration": float(c.duration),
                    "text": str(c.text),
                }
                for c in raw_cues
            ]
            yt_path.write_text(json.dumps(yt_cues, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[eval] Warning: could not fetch YouTube captions ({exc}); continuing without captions.")

    if seg_path.exists():
        asr_data = json.loads(seg_path.read_text(encoding="utf-8"))
        return asr_data, yt_cues

    if not wav_path.exists():
        m4a_path = cache_dir / "video.m4a"
        if not m4a_path.exists():
            url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"[eval] Downloading audio for {video_id}...")
            subprocess.run(
                ["uvx", "yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", str(m4a_path), url],
                check=True,
            )
        ff_cmd = ["ffmpeg", "-y", "-i", str(m4a_path), "-ac", "1", "-ar", "16000", "-f", "wav"]
        if max_seconds > 0:
            ff_cmd.extend(["-t", str(max_seconds)])
        ff_cmd.append(str(wav_path))
        subprocess.run(ff_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[eval] Running Silero VAD + Parakeet ({parakeet_model}) on {wav_path}...")
    with wave.open(str(wav_path), "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())
        audio_sec = wf.getnframes() / float(wf.getframerate())

    vad = ChannelVADSegmenter(SileroVADRuntime())
    t0 = time.perf_counter()
    final_segs = []
    partial_count = 0
    chunk_bytes = 3200  # 100ms of 16kHz PCM16
    for off in range(0, len(raw_pcm), chunk_bytes):
        for seg in vad.push_pcm_int16(raw_pcm[off : off + chunk_bytes]):
            if seg.kind == "partial":
                partial_count += 1
            else:
                final_segs.append(seg)
    final_segs.extend(vad.flush())
    vad_wall_ms = (time.perf_counter() - t0) * 1000.0

    t_load = time.perf_counter()
    asr = ParakeetEngine(model_id=parakeet_model)
    asr.ensure_loaded()
    asr_load_ms = (time.perf_counter() - t_load) * 1000.0

    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    t_asr = time.perf_counter()
    for idx, seg in enumerate(final_segs):
        res = asr.transcribe_pcm(seg.audio)
        latencies.append(res.inference_ms)
        records.append(
            {
                "idx": idx,
                "utterance_id": seg.utterance_id,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "dur_sec": round((seg.end_ms - seg.start_ms) / 1000.0, 2),
                "inference_ms": round(res.inference_ms, 1),
                "text": res.text,
            }
        )
    asr_total_s = time.perf_counter() - t_asr
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    asr_data = {
        "video_id": video_id,
        "audio_seconds": round(audio_sec, 1),
        "vad_wall_ms": round(vad_wall_ms, 1),
        "asr_load_ms": round(asr_load_ms, 1),
        "asr_model": parakeet_model,
        "asr_total_wall_sec": round(asr_total_s, 2),
        "realtime_factor": round(audio_sec / asr_total_s, 1) if asr_total_s > 0 else 0.0,
        "final_segment_count": len(records),
        "non_empty_segments": sum(1 for r in records if r["text"].strip()),
        "partial_emissions": partial_count,
        "avg_inference_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_inference_ms": round(p95, 1),
        "parakeet_word_count": sum(len(r["text"].split()) for r in records),
        "segments": records,
    }
    seg_path.write_text(json.dumps(asr_data, indent=2), encoding="utf-8")
    return asr_data, yt_cues


def compute_asr_match(asr_data: dict[str, Any], yt_cues: list[dict[str, Any]], max_seconds: int) -> dict[str, Any]:
    pk_words = _normalize_words(" ".join(s["text"] for s in asr_data["segments"]))
    if not yt_cues:
        return {"has_captions": False, "word_similarity": None, "pk_words": len(pk_words), "yt_words": 0}
    yt_filtered = [c for c in yt_cues if max_seconds <= 0 or float(c["start"]) < max_seconds]
    yt_words = _normalize_words(" ".join(str(c["text"]) for c in yt_filtered))
    matcher = difflib.SequenceMatcher(None, yt_words, pk_words, autojunk=False)
    ratio = matcher.ratio()
    return {
        "has_captions": True,
        "word_similarity": round(ratio * 100.0, 2),
        "pk_words": len(pk_words),
        "yt_words": len(yt_words),
    }


def score_run(
    *,
    session: InterviewSession,
    asr_data: dict[str, Any],
    asr_match: dict[str, Any],
    chapters: list[dict[str, Any]],
    triage_stats: dict[str, Any],
    probe_results: list[dict[str, Any]],
) -> dict[str, Any]:
    # 1. ASR Fidelity (25 pts)
    non_empty = asr_data.get("non_empty_segments", 0)
    total_segs = max(1, asr_data.get("final_segment_count", 1))
    drop_free_ratio = non_empty / total_segs
    sim_pct = asr_match.get("word_similarity")
    if sim_pct is not None:
        # 95%+ word similarity vs YouTube captions earns full similarity points (15 pts)
        sim_pts = min(15.0, max(0.0, (sim_pct - 75.0) / 20.0 * 15.0))
    else:
        sim_pts = 12.0
    drop_pts = 5.0 * drop_free_ratio
    rtf = float(asr_data.get("realtime_factor", 0.0))
    speed_pts = 5.0 if rtf >= 15.0 else (3.0 if rtf >= 5.0 else 1.5)
    asr_score = round(sim_pts + drop_pts + speed_pts, 1)

    # 2. Single-Mic Auto-Diarization & Handoff Splitting (25 pts)
    metrics = session.compute_metrics()
    invariants = validate_session_invariants(session)
    inv_pts = 5.0 if not invariants else 0.0

    # Speaker balance sanity (interviewer 8%-38% of words in behavioral mock)
    i_pct = metrics.interviewer_talk_pct
    if 8.0 <= i_pct <= 38.0:
        bal_pts = 8.0
    elif 4.0 <= i_pct <= 50.0:
        bal_pts = 5.0
    else:
        bal_pts = 1.0

    # Question-turn speaker attribution accuracy: utterances containing primary interviewer cues
    primary_q_utts = [
        u
        for u in session.utterances
        if re.search(
            r"\b(tell me about a time|walk me through|can you introduce yourself|describe a complex|next question|different question|last question|push back on|work across)\b",
            u.text,
            re.I,
        )
    ]
    q_int_hits = sum(1 for u in primary_q_utts if u.speaker == "interviewer")
    q_attr_ratio = q_int_hits / len(primary_q_utts) if primary_q_utts else 1.0
    q_attr_pts = round(7.0 * q_attr_ratio, 1)

    # Handoff split bonus: detected mixed-speaker VAD chunks cleanly split
    splits_applied = triage_stats.get("mixed_splits_applied", 0)
    handoff_pts = 5.0 if splits_applied >= 3 else (3.0 if splits_applied >= 1 else 1.0)
    diar_score = round(min(25.0, inv_pts + bal_pts + q_attr_pts + handoff_pts), 1)

    # 3. Chapter / Card Boundary Alignment (30 pts)
    # Match ground-truth non-intro chapters (and intro if present) against qa_groups start timestamps
    chapter_matches: list[dict[str, Any]] = []
    if chapters:
        gt_question_chapters = [c for c in chapters if int(c["ms"]) > 0]
        cards = session.qa_groups
        matched_card_ids: set[str] = set()
        hits = 0
        for ch in gt_question_chapters:
            target_ms = int(ch["ms"])
            best_card = None
            best_err = 10**9
            for qa in cards:
                err = abs(qa.started_ms - target_ms)
                if err < best_err:
                    best_err = err
                    best_card = qa
            is_hit = best_card is not None and best_err <= 25_000
            if is_hit and best_card is not None:
                hits += 1
                matched_card_ids.add(best_card.id)
            chapter_matches.append(
                {
                    "chapter": ch["title"],
                    "target_mm_ss": fmt_ms(target_ms),
                    "matched_card": f"{best_card.badge_prefix()}{best_card.index}: {best_card.title}" if best_card else "—",
                    "card_mm_ss": fmt_ms(best_card.started_ms) if best_card else "—",
                    "error_sec": round(best_err / 1000.0, 1) if best_card else None,
                    "hit": is_hit,
                }
            )
        recall = hits / len(gt_question_chapters) if gt_question_chapters else 1.0
        # Penalize severe over-splitting (expected ~= len(chapters) or len(gt_question_chapters) + 1 for intro)
        extra_cards = max(0, len(cards) - len(chapters) - 1)
        precision_pts = max(0.0, 10.0 - 2.5 * extra_cards)
        recall_pts = 20.0 * recall
        card_score = round(min(30.0, recall_pts + precision_pts), 1)
    else:
        card_score = 24.0 if 2 <= len(session.qa_groups) <= 8 else 14.0

    # 4. Live Dive-Deeper Probes (20 pts)
    if probe_results:
        live_probes = [p for p in probe_results if p.get("model_used") != "local-heuristic"]
        valid_3 = sum(1 for p in live_probes if len(p.get("probes", [])) == 3)
        avg_lat = sum(p.get("latency_ms", 0.0) for p in probe_results) / len(probe_results)
        # Entity grounding check: at least 2 content words (>=5 chars) from candidate text echoed in probes
        grounded = 0
        for p in probe_results:
            cand_vocab = {w for w in _normalize_words(p.get("candidate_sample", "")) if len(w) >= 5}
            probe_vocab = set(_normalize_words(" ".join(pr["question"] + " " + pr["why"] for pr in p.get("probes", []))))
            if len(cand_vocab & probe_vocab) >= 2:
                grounded += 1
        ground_ratio = grounded / len(probe_results)
        lat_pts = 5.0 if avg_lat <= 6000 else (3.5 if avg_lat <= 12000 else 2.0)
        probe_score = round(
            min(
                20.0,
                (7.0 * (valid_3 / len(probe_results)))
                + (8.0 * ground_ratio)
                + lat_pts,
            ),
            1,
        )
    else:
        probe_score = None

    max_possible = 100.0 if probe_score is not None else 80.0
    total = round(asr_score + diar_score + card_score + (probe_score or 0.0), 1)
    return {
        "total": total,
        "max_possible": max_possible,
        "asr_score": asr_score,
        "diarization_score": diar_score,
        "card_alignment_score": card_score,
        "probe_score": probe_score,
        "invariants": invariants,
        "chapter_matches": chapter_matches,
        "question_attribution": {"hits": q_int_hits, "total": len(primary_q_utts)},
    }


def render_summary_markdown(
    *,
    args: argparse.Namespace,
    session: InterviewSession,
    asr_data: dict[str, Any],
    asr_match: dict[str, Any],
    triage_stats: dict[str, Any],
    probe_results: list[dict[str, Any]],
    scores: dict[str, Any],
) -> str:
    m = session.compute_metrics()
    lines = [
        f"# Mock Helper YouTube Eval Summary — `{args.video_id}`",
        "",
        f"- **Provider / Fast Model:** `{args.provider}` / `{args.fast_model}`",
        f"- **Mode:** Single-mic blind auto-diarization (`in_person`, `window_size={args.window_size}`, `replay={args.replay}`)",
        f"- **Rubric Score:** **{scores['total']} / {scores['max_possible']}**",
        f"  - ASR Fidelity: **{scores['asr_score']} / 25** (Parakeet vs YouTube similarity: {asr_match.get('word_similarity') or 'N/A'}%, {asr_data['non_empty_segments']}/{asr_data['final_segment_count']} non-empty, {asr_data['realtime_factor']}x RT, avg {asr_data['avg_inference_ms']}ms)",
        f"  - Single-Mic Diarization & Handoff Splits: **{scores['diarization_score']} / 25** (Interviewer {m.interviewer_talk_pct}% / Candidate {m.candidate_talk_pct}% words; {triage_stats['mixed_splits_applied']} mixed-speaker VAD handoffs split; {triage_stats['speaker_flips']} flips; {triage_stats['reviews_queued']} medium/low review chips; invariants OK: `{not scores['invariants']}`)",
        f"  - Chapter / Card Alignment: **{scores['card_alignment_score']} / 30** ({len(session.qa_groups)} cards created)",
        f"  - Live Dive-Deeper Probes: **{scores['probe_score'] if scores['probe_score'] is not None else 'N/A'} / 20**",
        f"- **Triage Wall Time:** {triage_stats['triage_wall_sec']}s across {triage_stats['windows']} windows (avg {triage_stats['avg_window_sec']}s/window)",
        "",
        "## Chapter Alignment vs. Ground Truth",
        "",
        "| Ground-Truth Chapter | Target | Matched Card | Card Start | Error | Hit (<=25s) |",
        "|---|---|---|---|---|---|",
    ]
    for cm in scores["chapter_matches"]:
        err_s = f"{cm['error_sec']}s" if cm["error_sec"] is not None else "—"
        lines.append(
            f"| {cm['chapter']} | `{cm['target_mm_ss']}` | {cm['matched_card']} | `{cm['card_mm_ss']}` | {err_s} | {'PASS' if cm['hit'] else 'MISS'} |"
        )

    lines.extend(["", "## Resulting Q&A Cards", ""])
    for qa, stat in zip(session.qa_groups, m.per_card):
        utts = session.utterances_for_card(qa)
        lead_mm = fmt_ms(qa.started_ms)
        lines.append(
            f"- **{stat.badge} [`{lead_mm}`] {qa.title}** — {len(utts)} turns | Interviewer {stat.interviewer_words}w / Candidate {stat.candidate_words}w"
        )

    if session.pending_reviews:
        lines.extend(["", "## Queued Medium/Low-Confidence Review Chips", ""])
        for r in session.pending_reviews:
            lines.append(
                f"- `[{r.confidence}]` `{r.kind}` on `{r.utterance_id}` — {r.rationale}"
            )

    if probe_results:
        lines.extend(["", "## Live Dive-Deeper Probe Samples", ""])
        for p in probe_results:
            lines.append(f"### {p['badge']}: {p['title']} *(latency: {p['latency_ms']}ms, model: `{p['model_used']}`)*")
            lines.append(f"- **Quick Read:** {p['quick_read']}")
            for pr in p["probes"]:
                lines.append(f"  - **[{pr['angle']}]** {pr['question']} *(Why: {pr['why']})*")
            lines.append("")

    return "\n".join(lines) + "\n"


async def run_eval(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    preset = LLM_PRESETS.get(args.provider, LLM_PRESETS["agy"])
    fast_model = args.fast_model or preset["fast_model"]
    args.fast_model = fast_model

    asr_data, yt_cues = ensure_audio_and_asr(
        video_id=args.video_id,
        cache_dir=cache_dir,
        max_seconds=args.max_seconds,
        parakeet_model=args.parakeet_model,
    )
    asr_match = compute_asr_match(asr_data, yt_cues, args.max_seconds)

    meta = KNOWN_VIDEO_CHAPTERS.get(args.video_id, {})
    chapters = meta.get("chapters", [])
    if args.chapters_json:
        chapters = json.loads(Path(args.chapters_json).read_text(encoding="utf-8"))

    run_slug = _slugify(f"{args.provider}_{fast_model}_w{args.window_size}")
    run_dir = cache_dir / "runs" / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure isolated runtime config (never overwrites repo .env.local)
    llm_cfg = LLMRuntimeConfig(run_dir / "_cfg")
    llm_cfg.provider = args.provider
    llm_cfg.base_url = preset["base_url"]
    llm_cfg.fast_model = fast_model
    llm_cfg.calibration_model = preset["calibration_model"]
    llm = InterviewCopilotLLM(llm_cfg)

    if not args.replay and not llm_cfg.is_configured:
        raise RuntimeError(
            f"Provider '{args.provider}' is not configured/available. Check CLI installation or API key."
        )

    session = InterviewSession(
        config=SessionConfig(
            role_title=meta.get("role_title", f"YouTube Mock Interview ({args.video_id})"),
            interview_type=meta.get("interview_type", "behavioral"),
            mode="in_person",
            auto_diarize=True,
        )
    )
    grouper = QAGrouper()

    raw_segs = [s for s in asr_data["segments"] if s.get("text", "").strip()]
    if args.max_seconds > 0:
        raw_segs = [s for s in raw_segs if int(s["start_ms"]) < args.max_seconds * 1000]

    window_size = max(1, args.window_size)
    t_triage_start = time.perf_counter()
    mixed_splits = 0
    speaker_flips = 0
    reviews_queued = 0
    window_count = 0

    try:
        for w_idx, offset in enumerate(range(0, len(raw_segs), window_size)):
            window_count += 1
            chunk = raw_segs[offset : offset + window_size]
            window_utts: list[Utterance] = []
            for seg in chunk:
                # Blind single-mic simulation: every raw segment arrives tagged as 'candidate'
                utt = Utterance(
                    id=str(seg["utterance_id"]),
                    rev=1,
                    final=True,
                    speaker="candidate",
                    text=str(seg["text"]).strip(),
                    start_ms=int(seg["start_ms"]),
                    end_ms=int(seg["end_ms"]),
                )
                grouper.upsert_and_route(session, utt)
                window_utts.append(utt)

            trace_path = run_dir / f"window_{w_idx:02d}.json"
            if args.replay and trace_path.exists():
                trace_obj = json.loads(trace_path.read_text(encoding="utf-8"))
                plans = [TurnTriagePlan.model_validate(p) for p in trace_obj.get("plans", [])]
                w_sec = float(trace_obj.get("latency_sec", 0.0))
                src_tag = "replay"
            else:
                t_w = time.perf_counter()
                plans = await llm.triage_window(session, window_utts, diarize=True)
                w_sec = round(time.perf_counter() - t_w, 2)
                trace_path.write_text(
                    json.dumps(
                        {
                            "window_idx": w_idx,
                            "utterance_ids": [u.id for u in window_utts],
                            "latency_sec": w_sec,
                            "plans": [p.model_dump() for p in plans],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                src_tag = "live"

            w_splits = 0
            w_cards = 0
            for plan in plans:
                res = grouper.apply_turn_triage(
                    session,
                    plan,
                    allow_speaker_mutation=True,
                    allow_card_split=True,
                )
                if len(res.split_utterance_ids) >= 2:
                    mixed_splits += 1
                    w_splits += 1
                if res.speaker_changed:
                    speaker_flips += 1
                if res.queued_review_id:
                    reviews_queued += 1
                if res.opened_new_qa_id:
                    w_cards += 1

            inv_errors = validate_session_invariants(session)
            assert not inv_errors, f"Session invariant violation after window {w_idx}: {inv_errors}"
            print(
                f"[window {w_idx:02d}/{((len(raw_segs)-1)//window_size):02d} | {src_tag} {w_sec:4.1f}s] "
                f"plans={len(plans)}/{len(window_utts)} splits={w_splits} new_cards={w_cards} total_cards={len(session.qa_groups)}"
            )

        triage_wall_sec = round(time.perf_counter() - t_triage_start, 2)
        triage_stats = {
            "windows": window_count,
            "triage_wall_sec": triage_wall_sec,
            "avg_window_sec": round(triage_wall_sec / max(1, window_count), 2),
            "mixed_splits_applied": mixed_splits,
            "speaker_flips": speaker_flips,
            "reviews_queued": reviews_queued,
        }

        probe_results: list[dict[str, Any]] = []
        probes_path = run_dir / "probes.json"
        if args.probes:
            if args.replay and probes_path.exists():
                probe_results = json.loads(probes_path.read_text(encoding="utf-8"))
            else:
                for qa in session.qa_groups:
                    utts = session.utterances_for_card(qa)
                    cand_words = sum(len(u.text.split()) for u in utts if u.speaker == "candidate")
                    if cand_words < 20:
                        continue
                    t_p = time.perf_counter()
                    resp = await llm.suggest_probes(session, qa)
                    lat_ms = round((time.perf_counter() - t_p) * 1000.0, 1)
                    qa.dive_deeper_history.append(resp)
                    probe_results.append(
                        {
                            "qa_id": qa.id,
                            "badge": f"{qa.badge_prefix()}{qa.index}",
                            "title": qa.title,
                            "latency_ms": lat_ms,
                            "model_used": resp.model_used,
                            "quick_read": resp.quick_read,
                            "probes": [p.model_dump() for p in resp.probes],
                            "candidate_sample": " ".join(u.text for u in utts if u.speaker == "candidate")[:1200],
                        }
                    )
                    print(
                        f"[probes {qa.badge_prefix()}{qa.index}] {lat_ms:.0f}ms | {qa.title} -> {len(resp.probes)} probes"
                    )
                probes_path.write_text(json.dumps(probe_results, indent=2), encoding="utf-8")

        scores = score_run(
            session=session,
            asr_data=asr_data,
            asr_match=asr_match,
            chapters=chapters,
            triage_stats=triage_stats,
            probe_results=probe_results,
        )

        summary_md = render_summary_markdown(
            args=args,
            session=session,
            asr_data=asr_data,
            asr_match=asr_match,
            triage_stats=triage_stats,
            probe_results=probe_results,
            scores=scores,
        )
        (run_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
        (run_dir / "session.json").write_text(session.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "transcript.md").write_text(session.to_markdown(audience="interviewer"), encoding="utf-8")
        (run_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

        print("\n" + "=" * 72)
        print(summary_md)
        print(f"Artifacts saved to: {run_dir}")
    finally:
        await llm.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube mock-interview eval & replay harness for mockery")
    parser.add_argument("--video-id", default="-Y6YmuqGLpY", help="YouTube video ID")
    parser.add_argument("--max-seconds", type=int, default=1097, help="Clip audio at N seconds (0 = full video)")
    parser.add_argument("--cache-dir", default="", help="Directory holding cached audio/ASR/runs")
    parser.add_argument("--provider", default="agy", choices=sorted(LLM_PRESETS.keys()))
    parser.add_argument("--fast-model", default="", help="Override fast model spec (e.g. gemini-4-hydrogen:low)")
    parser.add_argument("--parakeet-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--window-size", type=int, default=10, help="Utterances per triage_window call")
    parser.add_argument("--chapters-json", default="", help="Optional JSON file with [{ms, title}, ...] ground truth")
    parser.add_argument("--replay", action="store_true", help="Replay cached window traces deterministically (0 LLM calls)")
    parser.add_argument("--probes", action="store_true", help="Run live Dive Deeper probes on each Q&A card")
    args = parser.parse_args()
    if not args.cache_dir:
        args.cache_dir = str(PROJECT_ROOT / "eval_cache" / args.video_id)
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
