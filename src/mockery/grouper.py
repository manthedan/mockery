from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import (
    InterviewSession,
    PendingReviewItem,
    QAGroup,
    SpeakerRole,
    TurnSegment,
    TurnTriagePlan,
    Utterance,
)

SMALL_TALK_RE = re.compile(
    r"\b(can you hear me|hear me okay|mic check|audio working|how are you|nice to meet you|good morning|good afternoon|hey there)\b",
    re.IGNORECASE,
)

BEHAVIORAL_PIVOT_RE = re.compile(
    r"\b(next question|moving on|move on to|shifting gears|shift gears|another question|different topic|switch gears|let's switch to|second question|third question|final question|last question|tell me about another|for my next)\b",
    re.IGNORECASE,
)

PHASE_PIVOT_RE = re.compile(
    r"\b(high[- ]level design|api design|data model|database schema|deep dive|bottlenecks|scaling|failure modes|edge cases|time complexity|space complexity|test cases|let's implement|start coding|walk through an example|next phase|moving on)\b",
    re.IGNORECASE,
)

FOLLOWUP_CONTINUATION_RE = re.compile(
    r"^(and\b|so\b|wait\b|quick follow|follow[- ]?up|going deeper|diving deeper|on that point|you mentioned|when you said|earlier you said|in that system|for that service|why did you|how did you measure)",
    re.IGNORECASE,
)

BACKCHANNEL_RE = re.compile(
    r"^(yeah|yep|yes|uh[- ]?huh|mm[- ]?hmm|got it|gotcha|makes sense|i see|okay|ok|right|sure|interesting|cool|awesome|great|go ahead|please continue|take your time)[.!?, ]*$",
    re.IGNORECASE,
)


@dataclass
class GroupingDecision:
    qa_id: str | None
    utterance_id: str
    opened_new_qa: bool = False
    needs_async_topic_check: bool = False
    should_speculate_probes: bool = False
    ignored_stale: bool = False


@dataclass
class TurnTriageApplyResult:
    mutated: bool = False
    speaker_changed: bool = False
    split_utterance_ids: list[str] = field(default_factory=list)
    opened_new_qa_id: str | None = None
    queued_review_id: str | None = None


def _text_overlap_ratio(original: str, candidate: str) -> float:
    """Dice word-multiset overlap in [0, 1] used to guard light STT cleanup against hallucination."""
    w1 = re.findall(r"[a-z0-9]+", original.lower())
    w2 = re.findall(r"[a-z0-9]+", candidate.lower())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    c1 = Counter(w1)
    c2 = Counter(w2)
    shared = sum(min(c1[w], c2[w]) for w in c1.keys() & c2.keys())
    return (2.0 * shared) / (len(w1) + len(w2))


def _collapse_adjacent_segments(segments: list[TurnSegment]) -> list[TurnSegment]:
    collapsed: list[TurnSegment] = []
    for seg in segments:
        txt = re.sub(r"\s+", " ", seg.text.strip())
        if not txt:
            continue
        if collapsed and collapsed[-1].speaker == seg.speaker:
            collapsed[-1] = TurnSegment(
                speaker=seg.speaker,
                text=f"{collapsed[-1].text} {txt}".strip(),
            )
        else:
            collapsed.append(TurnSegment(speaker=seg.speaker, text=txt))
    return collapsed


def _allocate_segment_spans(start_ms: int, end_ms: int, segments: list[TurnSegment]) -> list[tuple[int, int]]:
    total_ms = max(0, end_ms - start_ms)
    weights = [max(1, len(seg.text)) for seg in segments]
    total_w = sum(weights)
    spans: list[tuple[int, int]] = []
    cursor = start_ms
    acc_w = 0
    for idx, w in enumerate(weights):
        acc_w += w
        if idx == len(weights) - 1:
            seg_end = max(cursor, end_ms)
        else:
            seg_end = start_ms + int(round(total_ms * (acc_w / total_w)))
            seg_end = max(cursor, min(end_ms, seg_end))
        spans.append((cursor, seg_end))
        cursor = seg_end
    return spans


def make_concise_title(text: str, max_words: int = 9) -> str:
    raw = re.sub(r"\s+", " ", text.strip())
    if not raw:
        return "Discussion Thread"

    # If the interviewer opens with praise/filler before asking the actual question, focus on the question sentence.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
    chosen = raw
    if len(sentences) > 1:
        for sent in reversed(sentences):
            if len(sent.split()) >= 5 and (
                "?" in sent
                or re.search(
                    r"\b(want to ask|ask you about|describe|tell me|walk me through|can you|could you|how would|what was|explain)\b",
                    sent,
                    re.I,
                )
            ):
                chosen = sent
                break

    cleaned = re.sub(
        r"^(so|okay|alright|great|awesome|cool|thanks|thank you|well|um|uh)[,.\s]+",
        "",
        chosen,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if not words:
        return "Discussion Thread"
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


class QAGrouper:
    """
    Maintains `QAGroup` cards as a non-destructive derived view (`utterance_ids`)
    over the canonical `InterviewSession.utterances` log.
    Supports both Behavioral Q&A cards (`Q1..Qn`) and System Design / Coding Phase cards (`P1..Pn`).
    """

    @staticmethod
    def _card_kind(session: InterviewSession) -> Literal["question", "phase"]:
        return "phase" if session.config.interview_type in ("system_design", "coding") else "question"

    def sync_card_kinds(self, session: InterviewSession) -> None:
        """Keeps all derived cards aligned with `session.config.interview_type` (`Q` vs `P`)."""
        kind = self._card_kind(session)
        for qa in session.qa_groups:
            qa.card_kind = kind

    def upsert_and_route(self, session: InterviewSession, incoming: Utterance) -> GroupingDecision:
        incoming.text = incoming.text.strip()
        stored, is_new, applied = session.upsert_utterance_with_status(incoming)
        if not applied:
            owner = self._find_owner_card(session, stored.id)
            return GroupingDecision(
                qa_id=owner.id if owner else None,
                utterance_id=stored.id,
                ignored_stale=True,
            )

        word_count = len(stored.text.split())
        kind = self._card_kind(session)

        # Case A: Existing utterance revision (`rev` increment on draft/final or Phase 2 speaker flip)
        if not is_new:
            owner = self._find_owner_card(session, stored.id)
            if owner is not None:
                owner.updated_ms = max(owner.updated_ms, stored.end_ms)
                if owner.main_question == "" and stored.speaker == "interviewer":
                    owner.main_question = stored.text

                # Re-evaluate card split/topic shift when an interviewer draft finalizes mid-card!
                if stored.final and stored.speaker == "interviewer" and stored.id in owner.utterance_ids:
                    pos = owner.utterance_ids.index(stored.id)
                    if pos > 0:
                        umap = session.utterance_map()
                        prior_utts = [umap[uid] for uid in owner.utterance_ids[:pos] if uid in umap]
                        cand_words_before = sum(
                            len(u.text.split()) for u in prior_utts if u.speaker == "candidate"
                        )
                        pivot_re = PHASE_PIVOT_RE if kind == "phase" else BEHAVIORAL_PIVOT_RE
                        if cand_words_before >= 10 and pivot_re.search(stored.text):
                            new_qa = self.split_utterance_to_new_card(
                                session,
                                owner.id,
                                stored.id,
                                new_title=make_concise_title(stored.text),
                            )
                            if new_qa is not None:
                                return GroupingDecision(
                                    qa_id=new_qa.id,
                                    utterance_id=stored.id,
                                    opened_new_qa=True,
                                )

                        prev_utt = prior_utts[-1] if prior_utts else None
                        answering_clarification = bool(
                            prev_utt and prev_utt.speaker == "candidate" and prev_utt.text.rstrip().endswith("?")
                        )
                        needs_check = (
                            kind == "question"
                            and not answering_clarification
                            and not (word_count <= 6 and BACKCHANNEL_RE.match(stored.text))
                            and not FOLLOWUP_CONTINUATION_RE.search(stored.text)
                            and cand_words_before >= 40
                            and (word_count >= 10 or "?" in stored.text)
                        )
                        if needs_check:
                            return GroupingDecision(
                                qa_id=owner.id,
                                utterance_id=stored.id,
                                needs_async_topic_check=True,
                            )

                speculate = (
                    stored.final
                    and stored.speaker == "candidate"
                    and session.card_word_counts(owner)["candidate"] >= 25
                )
                return GroupingDecision(
                    qa_id=owner.id,
                    utterance_id=stored.id,
                    should_speculate_probes=speculate,
                )

            # Still in warmup_utterance_ids and no Q&A cards exist yet — promote if it grew into Q1
            # (still respecting small-talk guard so audio-check drafts don't open Q1 on finalization).
            if not session.qa_groups and stored.id in session.warmup_utterance_ids:
                is_small_talk = word_count <= 9 and bool(SMALL_TALK_RE.search(stored.text))
                if not is_small_talk and self._qualifies_to_open_first_card(stored, word_count):
                    session.warmup_utterance_ids.remove(stored.id)
                    title = (
                        "Problem & Requirements"
                        if kind == "phase" and stored.speaker == "candidate"
                        else make_concise_title(stored.text)
                    )
                    qa = self._create_card(session, stored, title=title, card_kind=kind)
                    speculate = (
                        stored.final
                        and stored.speaker == "candidate"
                        and word_count >= 25
                    )
                    return GroupingDecision(
                        qa_id=qa.id,
                        utterance_id=stored.id,
                        opened_new_qa=True,
                        should_speculate_probes=speculate,
                    )

            return GroupingDecision(qa_id=None, utterance_id=stored.id)

        # Case B: Brand-new `utterance_id`
        if not session.qa_groups:
            if word_count <= 9 and SMALL_TALK_RE.search(stored.text):
                session.warmup_utterance_ids.append(stored.id)
                return GroupingDecision(qa_id=None, utterance_id=stored.id)

            if self._qualifies_to_open_first_card(stored, word_count):
                default_title = (
                    "Problem & Requirements"
                    if kind == "phase" and word_count < 8
                    else (
                        "Introduction & Background"
                        if stored.speaker == "candidate"
                        else make_concise_title(stored.text)
                    )
                )
                qa = self._create_card(session, stored, title=default_title, card_kind=kind)
                speculate = (
                    stored.final
                    and stored.speaker == "candidate"
                    and word_count >= 25
                )
                return GroupingDecision(
                    qa_id=qa.id,
                    utterance_id=stored.id,
                    opened_new_qa=True,
                    should_speculate_probes=speculate,
                )

            session.warmup_utterance_ids.append(stored.id)
            return GroupingDecision(qa_id=None, utterance_id=stored.id)

        current_qa = session.qa_groups[-1]

        # Candidate speech always attaches to the active Q&A / Phase card
        if stored.speaker == "candidate":
            self._attach(session, current_qa, stored)
            speculate = (
                stored.final and session.card_word_counts(current_qa)["candidate"] >= 25
            )
            return GroupingDecision(
                qa_id=current_qa.id,
                utterance_id=stored.id,
                should_speculate_probes=speculate,
            )

        # Interviewer speech while a card is active
        counts = session.card_word_counts(current_qa)
        cand_words = counts["candidate"]
        card_utts = session.utterances_for_card(current_qa)
        last_utt = card_utts[-1] if card_utts else None

        # Short backchannel stays in current card
        if word_count <= 6 and BACKCHANNEL_RE.match(stored.text):
            self._attach(session, current_qa, stored)
            return GroupingDecision(qa_id=current_qa.id, utterance_id=stored.id)

        pivot_re = PHASE_PIVOT_RE if kind == "phase" else BEHAVIORAL_PIVOT_RE

        # Explicit behavioral pivot or system-design/coding phase shift after any candidate exchange -> open new card!
        if cand_words >= 10 and pivot_re.search(stored.text):
            qa = self._create_card(
                session,
                stored,
                title=make_concise_title(stored.text),
                card_kind=kind,
            )
            return GroupingDecision(qa_id=qa.id, utterance_id=stored.id, opened_new_qa=True)

        # Candidate hasn't answered substantially yet -> interviewer is still framing the question
        if cand_words < 25:
            self._attach(session, current_qa, stored)
            if not current_qa.main_question:
                current_qa.main_question = stored.text
                if current_qa.title.startswith(("Question ", "Phase ")):
                    current_qa.title = make_concise_title(stored.text)
            return GroupingDecision(qa_id=current_qa.id, utterance_id=stored.id)

        # Answering a candidate's clarifying question stays in current card
        if last_utt and last_utt.speaker == "candidate" and last_utt.text.rstrip().endswith("?"):
            self._attach(session, current_qa, stored)
            return GroupingDecision(qa_id=current_qa.id, utterance_id=stored.id)

        # Direct follow-up probe stays in current card
        if FOLLOWUP_CONTINUATION_RE.search(stored.text) or word_count <= 9:
            self._attach(session, current_qa, stored)
            return GroupingDecision(qa_id=current_qa.id, utterance_id=stored.id)

        # In phase mode (system design / coding), stay in current phase unless explicit shift or manual hotkey;
        # in behavioral/general mode, trigger non-blocking LLM topic-shift check on finalized substantive turns.
        self._attach(session, current_qa, stored)
        needs_check = (
            kind == "question"
            and stored.final
            and cand_words >= 40
            and (word_count >= 10 or "?" in stored.text)
        )
        return GroupingDecision(
            qa_id=current_qa.id,
            utterance_id=stored.id,
            needs_async_topic_check=needs_check,
        )

    def ingest_utterance(
        self,
        session: InterviewSession,
        *,
        speaker: Literal["interviewer", "candidate"],
        text: str,
        start_ms: int,
        end_ms: int,
        utt_id: str | None = None,
        rev: int = 1,
        final: bool = True,
    ) -> GroupingDecision:
        utt = (
            Utterance(id=utt_id, rev=rev, final=final, speaker=speaker, text=text, start_ms=start_ms, end_ms=end_ms)
            if utt_id
            else Utterance(rev=rev, final=final, speaker=speaker, text=text, start_ms=start_ms, end_ms=end_ms)
        )
        return self.upsert_and_route(session, utt)

    def force_new_card(self, session: InterviewSession, title: str = "") -> QAGroup:
        kind = self._card_kind(session)
        prefix_word = "Phase" if kind == "phase" else "Question"
        now_ms = 0
        hoisted_id: str | None = None

        umap = session.utterance_map()
        if session.qa_groups:
            prev = session.qa_groups[-1]
            now_ms = prev.updated_ms
            prev_utts = session.utterances_for_card(prev)
            if len(prev_utts) >= 2 and prev_utts[-1].speaker == "interviewer":
                cand_before = sum(len(u.text.split()) for u in prev_utts[:-1] if u.speaker == "candidate")
                if cand_before >= 10:
                    hoisted_id = prev.utterance_ids.pop()
                    prev.speculative_cache = None
                    rem = [umap[uid] for uid in prev.utterance_ids if uid in umap]
                    if rem:
                        prev.updated_ms = max(u.end_ms for u in rem)

        idx = len(session.qa_groups) + 1
        if hoisted_id and hoisted_id in umap:
            u = umap[hoisted_id]
            qa = QAGroup(
                index=idx,
                card_kind=kind,
                title=title.strip() or make_concise_title(u.text),
                main_question=u.text,
                utterance_ids=[hoisted_id],
                started_ms=u.start_ms,
                updated_ms=u.end_ms,
            )
        else:
            qa = QAGroup(
                index=idx,
                card_kind=kind,
                title=title.strip() or f"{prefix_word} {idx}",
                main_question="",
                utterance_ids=[],
                started_ms=now_ms,
                updated_ms=now_ms,
            )
        session.qa_groups.append(qa)
        self._reindex(session)
        return qa

    def split_utterance_to_new_card(
        self,
        session: InterviewSession,
        qa_id: str,
        utterance_id: str,
        new_title: str | None = None,
    ) -> QAGroup | None:
        umap = session.utterance_map()
        kind = self._card_kind(session)
        for i, qa in enumerate(session.qa_groups):
            if qa.id != qa_id:
                continue
            if utterance_id not in qa.utterance_ids:
                return None
            pos = qa.utterance_ids.index(utterance_id)
            if pos == 0:
                return None
            moved_ids = qa.utterance_ids[pos:]
            qa.utterance_ids = qa.utterance_ids[:pos]
            qa.speculative_cache = None
            rem_tail = umap.get(qa.utterance_ids[-1]) if qa.utterance_ids else None
            if rem_tail:
                qa.updated_ms = rem_tail.end_ms
            lead = umap.get(moved_ids[0])
            tail = umap.get(moved_ids[-1])
            new_qa = QAGroup(
                index=qa.index + 1,
                card_kind=kind,
                title=(new_title or (make_concise_title(lead.text) if lead else "Topic")).strip(),
                main_question=lead.text if (lead and lead.speaker == "interviewer") else "",
                utterance_ids=moved_ids,
                started_ms=lead.start_ms if lead else qa.updated_ms,
                updated_ms=tail.end_ms if tail else qa.updated_ms,
            )
            session.qa_groups.insert(i + 1, new_qa)
            self._reindex(session)
            return new_qa
        return None

    def merge_with_previous(self, session: InterviewSession, qa_id: str) -> QAGroup | None:
        umap = session.utterance_map()
        for i, qa in enumerate(session.qa_groups):
            if qa.id == qa_id and i > 0:
                prev = session.qa_groups[i - 1]
                prev.utterance_ids.extend(qa.utterance_ids)
                prev.utterance_ids.sort(key=lambda uid: (umap[uid].start_ms, uid) if uid in umap else (0, uid))
                prev.speculative_cache = None
                for hist in qa.dive_deeper_history:
                    prev.dive_deeper_history.append(hist.model_copy(update={"qa_id": prev.id}))
                prev.dive_deeper_history.sort(key=lambda h: h.generated_at_ms)
                if qa.notes:
                    prev.notes = f"{prev.notes}\n{qa.notes}".strip()
                prev.updated_ms = max(prev.updated_ms, qa.updated_ms)
                session.qa_groups.pop(i)
                self._reindex(session)
                return prev
        return None

    def apply_regroup_boundaries(
        self,
        session: InterviewSession,
        boundaries: list[dict[str, Any]],
    ) -> bool:
        """
        Applies full-context card boundaries `[{"start_utterance_id": ..., "title": ...}, ...]`
        over all non-warmup utterances without losing or duplicating a single utterance ID.
        """
        warm_set = set(session.warmup_utterance_ids)
        ordered_ids = [u.id for u in session.utterances if u.id not in warm_set]
        if not ordered_ids or not boundaries:
            return False

        id_to_pos = {uid: idx for idx, uid in enumerate(ordered_ids)}
        valid_starts: list[tuple[int, str]] = []
        seen_pos: set[int] = set()
        for item in boundaries:
            uid = str(item.get("start_utterance_id", ""))
            title = str(item.get("title", "")).strip() or "Discussion Thread"
            if uid in id_to_pos and id_to_pos[uid] not in seen_pos:
                pos = id_to_pos[uid]
                seen_pos.add(pos)
                valid_starts.append((pos, title))

        if not valid_starts:
            return False
        valid_starts.sort(key=lambda x: x[0])
        if valid_starts[0][0] != 0:
            valid_starts.insert(0, (0, session.qa_groups[0].title if session.qa_groups else "Opening Question"))

        old_cards = list(session.qa_groups)
        old_by_lead = {
            qa.utterance_ids[0]: qa for qa in old_cards if qa.utterance_ids
        }
        umap = session.utterance_map()
        kind = self._card_kind(session)
        rebuilt: list[QAGroup] = []

        for idx, (start_pos, title) in enumerate(valid_starts):
            end_pos = valid_starts[idx + 1][0] if idx + 1 < len(valid_starts) else len(ordered_ids)
            chunk_ids = ordered_ids[start_pos:end_pos]
            if not chunk_ids:
                continue
            lead_id = chunk_ids[0]
            prior = old_by_lead.get(lead_id)
            lead_u = umap.get(lead_id)
            tail_u = umap.get(chunk_ids[-1])
            rebuilt.append(
                QAGroup(
                    id=prior.id if prior else QAGroup(index=idx + 1, title=title).id,
                    index=idx + 1,
                    card_kind=kind,
                    title=title,
                    main_question=lead_u.text if (lead_u and lead_u.speaker == "interviewer") else "",
                    utterance_ids=chunk_ids,
                    dive_deeper_history=[],
                    speculative_cache=None,
                    notes="",
                    started_ms=lead_u.start_ms if lead_u else 0,
                    updated_ms=tail_u.end_ms if tail_u else 0,
                )
            )

        if not rebuilt:
            return False

        # Migrate notes & probe history from ALL original cards to their best-overlapping rebuilt card
        rebuilt_sets = [set(card.utterance_ids) for card in rebuilt]
        for old_idx, old_qa in enumerate(old_cards):
            if not old_qa.notes.strip() and not old_qa.dive_deeper_history:
                continue
            old_set = set(old_qa.utterance_ids)
            if not old_set:
                best_idx = min(old_idx, len(rebuilt) - 1)
            else:
                best_idx = 0
                best_overlap = -1
                for r_idx, r_set in enumerate(rebuilt_sets):
                    overlap = len(old_set & r_set)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = r_idx
            target = rebuilt[best_idx]
            if old_qa.notes.strip():
                target.notes = f"{target.notes}\n{old_qa.notes.strip()}".strip() if target.notes else old_qa.notes.strip()
            for hist in old_qa.dive_deeper_history:
                migrated = hist.model_copy(update={"qa_id": target.id})
                target.dive_deeper_history.append(migrated)

        for card in rebuilt:
            card.dive_deeper_history.sort(key=lambda h: h.generated_at_ms)

        session.qa_groups = rebuilt
        return True

    def apply_turn_triage(
        self,
        session: InterviewSession,
        plan: TurnTriagePlan,
        *,
        allow_speaker_mutation: bool = True,
        allow_card_split: bool = True,
        force_apply: bool = False,
    ) -> TurnTriageApplyResult:
        """
        Deterministic, confidence-gated applier for `TurnTriagePlan`.
        - High-confidence (or `force_apply=True`) ops mutate canonical utterances / cards safely.
        - Medium/low-confidence ops that would flip speaker, split a mixed turn, or split a card
          are queued into `session.pending_reviews` for 1-click user confirmation without mutating.
        - Light STT proper-noun cleanup is guarded by `_text_overlap_ratio >= 0.65` to prevent
          hallucinated paraphrasing.
        """
        result = TurnTriageApplyResult()
        orig = session.utterance_map().get(plan.utterance_id)
        if orig is None or not orig.final:
            return result

        segments = _collapse_adjacent_segments(plan.segments)
        if not segments:
            segments = [TurnSegment(speaker=orig.speaker, text=orig.text)]

        owner = self._find_owner_card(session, orig.id)
        in_warmup = orig.id in session.warmup_utterance_ids
        is_high_conf = plan.confidence == "high" or force_apply

        # Case 1: Mixed-speaker single-mic turn split (2+ alternating speaker segments)
        if allow_speaker_mutation and len(segments) >= 2:
            combined_text = " ".join(s.text for s in segments)
            if _text_overlap_ratio(orig.text, combined_text) >= 0.60:
                if not is_high_conf:
                    review = self._queue_pending_review(
                        session,
                        PendingReviewItem(
                            utterance_id=orig.id,
                            qa_id=owner.id if owner else None,
                            kind="split_mixed_turn",
                            suggested_segments=segments,
                            suggested_title=plan.suggested_title,
                            rationale=plan.rationale or "Possible rapid speaker handoff inside one segment.",
                            confidence=plan.confidence,
                        ),
                    )
                    result.queued_review_id = review.id
                    return result

                spans = _allocate_segment_spans(orig.start_ms, orig.end_ms, segments)
                first_start, first_end = spans[0]
                updated_first = orig.model_copy(
                    update={
                        "rev": orig.rev + 1,
                        "speaker": segments[0].speaker,
                        "text": segments[0].text,
                        "start_ms": first_start,
                        "end_ms": first_end,
                    }
                )
                session.upsert_utterance_with_status(updated_first)
                if segments[0].speaker != orig.speaker:
                    result.speaker_changed = True

                created_utts: list[Utterance] = [updated_first]
                child_ids: list[str] = []
                for seg, (seg_start, seg_end) in zip(segments[1:], spans[1:]):
                    child = Utterance(
                        rev=1,
                        final=True,
                        speaker=seg.speaker,
                        text=seg.text,
                        start_ms=seg_start,
                        end_ms=seg_end,
                    )
                    session.upsert_utterance_with_status(child)
                    created_utts.append(child)
                    child_ids.append(child.id)

                if owner is not None:
                    pos = owner.utterance_ids.index(orig.id)
                    owner.utterance_ids[pos + 1 : pos + 1] = child_ids
                    owner.speculative_cache = None
                    owner.updated_ms = max(owner.updated_ms, orig.end_ms)
                    if not owner.main_question:
                        first_int = next((u for u in session.utterances_for_card(owner) if u.speaker == "interviewer"), None)
                        if first_int:
                            owner.main_question = first_int.text
                elif in_warmup:
                    pos = session.warmup_utterance_ids.index(orig.id)
                    session.warmup_utterance_ids[pos + 1 : pos + 1] = child_ids

                session.pending_reviews = [r for r in session.pending_reviews if r.utterance_id != orig.id]
                result.mutated = True
                result.speaker_changed = True
                result.split_utterance_ids = [u.id for u in created_utts]

                # Handle card boundary opening on the interviewer segment when requested (or opening Q1 from warmup)
                int_candidates = [u for u in created_utts if u.speaker == "interviewer"]
                pivot_utt = (
                    max(int_candidates, key=lambda u: len(u.text.split()))
                    if int_candidates
                    else created_utts[0]
                )
                if allow_card_split and plan.card_action == "open_new_card":
                    new_qa = self._open_card_at_utterance(
                        session,
                        pivot_utt.id,
                        suggested_title=plan.suggested_title or make_concise_title(pivot_utt.text),
                    )
                    if new_qa is not None:
                        result.opened_new_qa_id = new_qa.id
                elif in_warmup and not session.qa_groups:
                    for u in created_utts:
                        wc = len(u.text.split())
                        is_small = wc <= 9 and bool(SMALL_TALK_RE.search(u.text))
                        if not is_small and self._qualifies_to_open_first_card(u, wc):
                            new_qa = self._open_card_at_utterance(
                                session,
                                u.id,
                                suggested_title=plan.suggested_title or make_concise_title(u.text),
                            )
                            if new_qa is not None:
                                result.opened_new_qa_id = new_qa.id
                            break
                return result
            else:
                # Hallucinated split rejected by overlap guard — fall back to single segment with original text
                segments = [TurnSegment(speaker=segments[0].speaker, text=orig.text)]

        # Case 2: Single-speaker turn (optional speaker flip, guarded STT cleanup, optional card split)
        seg = segments[0]
        target_speaker: SpeakerRole = seg.speaker if allow_speaker_mutation else orig.speaker
        wants_speaker_flip = allow_speaker_mutation and target_speaker != orig.speaker
        wants_text_clean = (
            allow_speaker_mutation
            and seg.text != orig.text
            and _text_overlap_ratio(orig.text, seg.text) >= 0.65
        )
        wants_card_split = allow_card_split and plan.card_action == "open_new_card"

        if not is_high_conf and (wants_speaker_flip or wants_card_split):
            review_kind: Literal["flip_speaker", "split_card"] = (
                "flip_speaker" if wants_speaker_flip else "split_card"
            )
            review = self._queue_pending_review(
                session,
                PendingReviewItem(
                    utterance_id=orig.id,
                    qa_id=owner.id if owner else None,
                    kind=review_kind,
                    suggested_speaker=target_speaker if wants_speaker_flip else None,
                    suggested_title=plan.suggested_title if wants_card_split else "",
                    rationale=plan.rationale or "Low-confidence triage suggestion — confirm to apply.",
                    confidence=plan.confidence,
                ),
            )
            result.queued_review_id = review.id
            return result

        updated = orig
        if wants_speaker_flip or wants_text_clean:
            updated = orig.model_copy(
                update={
                    "rev": orig.rev + 1,
                    "speaker": target_speaker,
                    "text": seg.text if wants_text_clean else orig.text,
                }
            )
            session.upsert_utterance_with_status(updated)
            result.mutated = True
            result.speaker_changed = wants_speaker_flip
            if owner is not None:
                owner.speculative_cache = None
                if not owner.main_question and updated.speaker == "interviewer":
                    owner.main_question = updated.text
                if (
                    owner.utterance_ids
                    and owner.utterance_ids[0] == updated.id
                    and updated.speaker == "interviewer"
                    and owner.title in ("Introduction & Background", "Discussion Thread")
                ):
                    owner.title = make_concise_title(updated.text)

        if is_high_conf:
            session.pending_reviews = [r for r in session.pending_reviews if r.utterance_id != orig.id]

        if wants_card_split:
            new_qa = self._open_card_at_utterance(
                session,
                updated.id,
                suggested_title=plan.suggested_title or make_concise_title(updated.text),
            )
            if new_qa is not None:
                result.mutated = True
                result.opened_new_qa_id = new_qa.id
            elif owner is not None and owner.utterance_ids and owner.utterance_ids[0] == updated.id and plan.suggested_title:
                owner.title = plan.suggested_title
                result.mutated = True
        elif in_warmup and not session.qa_groups and wants_speaker_flip:
            wc = len(updated.text.split())
            is_small = wc <= 9 and bool(SMALL_TALK_RE.search(updated.text))
            if not is_small and self._qualifies_to_open_first_card(updated, wc):
                new_qa = self._open_card_at_utterance(
                    session,
                    updated.id,
                    suggested_title=plan.suggested_title or make_concise_title(updated.text),
                )
                if new_qa is not None:
                    result.mutated = True
                    result.opened_new_qa_id = new_qa.id

        return result

    def flip_utterance_speaker(self, session: InterviewSession, utterance_id: str) -> Utterance | None:
        """Manually flips an utterance between `interviewer` and `candidate` at `rev + 1`."""
        orig = session.utterance_map().get(utterance_id)
        if orig is None:
            return None
        flipped: SpeakerRole = "candidate" if orig.speaker == "interviewer" else "interviewer"
        updated = orig.model_copy(update={"rev": orig.rev + 1, "speaker": flipped})
        session.upsert_utterance_with_status(updated)
        session.pending_reviews = [
            r for r in session.pending_reviews if not (r.utterance_id == utterance_id and r.kind == "flip_speaker")
        ]
        owner = self._find_owner_card(session, utterance_id)
        if owner is not None:
            owner.speculative_cache = None
            first_int = next((u for u in session.utterances_for_card(owner) if u.speaker == "interviewer"), None)
            owner.main_question = first_int.text if first_int else ""
        return updated

    def resolve_pending_review(
        self,
        session: InterviewSession,
        review_id: str,
        *,
        accept: bool,
    ) -> TurnTriageApplyResult:
        """Accepts or dismisses a confidence-gated `PendingReviewItem`."""
        review = next((r for r in session.pending_reviews if r.id == review_id), None)
        if review is None:
            return TurnTriageApplyResult()
        session.pending_reviews = [r for r in session.pending_reviews if r.id != review_id]
        if not accept:
            return TurnTriageApplyResult(mutated=True)

        orig = session.utterance_map().get(review.utterance_id)
        if orig is None:
            return TurnTriageApplyResult(mutated=True)

        if review.kind == "split_mixed_turn" and review.suggested_segments:
            plan = TurnTriagePlan(
                utterance_id=orig.id,
                segments=review.suggested_segments,
                card_action="open_new_card" if review.suggested_title else "keep_in_current",
                suggested_title=review.suggested_title,
                confidence="high",
                rationale=review.rationale,
            )
        elif review.kind == "flip_speaker" and review.suggested_speaker:
            plan = TurnTriagePlan(
                utterance_id=orig.id,
                segments=[TurnSegment(speaker=review.suggested_speaker, text=orig.text)],
                card_action="keep_in_current",
                confidence="high",
                rationale=review.rationale,
            )
        else:
            plan = TurnTriagePlan(
                utterance_id=orig.id,
                segments=[TurnSegment(speaker=orig.speaker, text=orig.text)],
                card_action="open_new_card",
                suggested_title=review.suggested_title or make_concise_title(orig.text),
                confidence="high",
                rationale=review.rationale,
            )
        return self.apply_turn_triage(
            session,
            plan,
            allow_speaker_mutation=True,
            allow_card_split=True,
            force_apply=True,
        )

    def _open_card_at_utterance(
        self,
        session: InterviewSession,
        utterance_id: str,
        *,
        suggested_title: str,
    ) -> QAGroup | None:
        owner = self._find_owner_card(session, utterance_id)
        if owner is not None:
            return self.split_utterance_to_new_card(
                session,
                qa_id=owner.id,
                utterance_id=utterance_id,
                new_title=suggested_title,
            )

        if utterance_id in session.warmup_utterance_ids:
            pos = session.warmup_utterance_ids.index(utterance_id)
            moved_ids = session.warmup_utterance_ids[pos:]
            session.warmup_utterance_ids = session.warmup_utterance_ids[:pos]
            umap = session.utterance_map()
            lead = umap.get(moved_ids[0])
            tail = umap.get(moved_ids[-1])
            kind = self._card_kind(session)
            new_qa = QAGroup(
                index=1,
                card_kind=kind,
                title=suggested_title.strip() or (make_concise_title(lead.text) if lead else "Opening Question"),
                main_question=lead.text if (lead and lead.speaker == "interviewer") else "",
                utterance_ids=moved_ids,
                started_ms=lead.start_ms if lead else 0,
                updated_ms=tail.end_ms if tail else 0,
            )
            session.qa_groups.insert(0, new_qa)
            self._reindex(session)
            return new_qa

        return None

    @staticmethod
    def _queue_pending_review(session: InterviewSession, item: PendingReviewItem) -> PendingReviewItem:
        for idx, existing in enumerate(session.pending_reviews):
            if existing.utterance_id == item.utterance_id and existing.kind == item.kind:
                session.pending_reviews[idx] = item.model_copy(update={"id": existing.id})
                return session.pending_reviews[idx]
        session.pending_reviews.append(item)
        return item

    @staticmethod
    def _qualifies_to_open_first_card(utt: Utterance, word_count: int) -> bool:
        if utt.speaker == "interviewer" and (word_count >= 5 or "?" in utt.text):
            return True
        if utt.speaker == "candidate" and word_count >= 14:
            return True
        return False

    @staticmethod
    def _find_owner_card(session: InterviewSession, utt_id: str) -> QAGroup | None:
        for qa in session.qa_groups:
            if utt_id in qa.utterance_ids:
                return qa
        return None

    @staticmethod
    def _create_card(
        session: InterviewSession,
        first_utt: Utterance,
        *,
        title: str,
        card_kind: Literal["question", "phase"],
    ) -> QAGroup:
        qa = QAGroup(
            index=len(session.qa_groups) + 1,
            card_kind=card_kind,
            title=title,
            main_question=first_utt.text if first_utt.speaker == "interviewer" else "",
            utterance_ids=[first_utt.id],
            started_ms=first_utt.start_ms,
            updated_ms=first_utt.end_ms,
        )
        session.qa_groups.append(qa)
        return qa

    @staticmethod
    def _attach(session: InterviewSession, qa: QAGroup, utt: Utterance) -> None:
        if utt.id not in qa.utterance_ids:
            qa.utterance_ids.append(utt.id)
            umap = session.utterance_map()
            qa.utterance_ids.sort(key=lambda uid: (umap[uid].start_ms, uid) if uid in umap else (0, uid))
        qa.updated_ms = max(qa.updated_ms, utt.end_ms)

    @staticmethod
    def _reindex(session: InterviewSession) -> None:
        for idx, qa in enumerate(session.qa_groups, start=1):
            qa.index = idx
