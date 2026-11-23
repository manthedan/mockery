"""Candidate Profile, Resume Redaction, Job Listings, and Question Banks router."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ..banks import MAX_MY_QUESTION_CHARS, MAX_MY_QUESTIONS, MyQuestionsError, split_pasted_questions
from ..levels import DEFAULT_LEVEL
from ..profiles import ResumeRedactionPrefs, StaleResumeDocumentError
from ..redactor import (
    clean_job_listing_text,
    extract_document_text,
    fetch_job_listing,
    parse_job_title_and_company,
)

MAX_RESUME_UPLOAD_BYTES = 10 * 1024 * 1024


class ResumeTextUploadRequest(BaseModel):
    filename: str = "resume.txt"
    text: str


class JobListingImportRequest(BaseModel):
    url: str = ""
    text: str = ""
    title: str = ""
    company: str = ""
    target_level: str = DEFAULT_LEVEL


MyQuestionCategory = Literal["behavioral", "general", "system_design", "coding"]


class MyQuestionsAddRequest(BaseModel):
    """Save one question as written, or (`as_list`) every question in a pasted list."""

    category: MyQuestionCategory
    text: str = Field(max_length=MAX_MY_QUESTIONS * MAX_MY_QUESTION_CHARS)
    as_list: bool = False


def build_profile_router(runtime: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/profile")
    async def get_candidate_profile() -> dict[str, Any]:
        return runtime.profiles.public_dict()

    @router.post("/api/profile/resume/file")
    async def upload_resume_file(file: UploadFile = File(...)) -> dict[str, Any]:
        raw_bytes = await file.read(MAX_RESUME_UPLOAD_BYTES + 1)
        if len(raw_bytes) > MAX_RESUME_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Resume file exceeds 10 MB limit.")
        filename = Path(file.filename or "resume.txt").name
        try:
            extracted = await asyncio.to_thread(extract_document_text, filename, raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read resume file: {exc}") from exc
        if not extracted.strip():
            raise HTTPException(
                status_code=400,
                detail="No selectable text found in that file (if it is a scanned image PDF, paste the text directly).",
            )
        await asyncio.to_thread(
            runtime.profiles.set_resume,
            filename=filename,
            raw_text=extracted,
        )
        return runtime.profiles.public_dict()

    @router.post("/api/profile/resume/text")
    async def upload_resume_text(body: ResumeTextUploadRequest) -> dict[str, Any]:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
        if len(body.text.encode("utf-8")) > MAX_RESUME_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Resume text exceeds 10 MB limit.")
        await asyncio.to_thread(
            runtime.profiles.set_resume,
            filename=Path(body.filename or "resume.txt").name,
            raw_text=body.text.strip(),
        )
        return runtime.profiles.public_dict()

    @router.post("/api/profile/resume/redact")
    async def update_resume_redaction(body: ResumeRedactionPrefs) -> dict[str, Any]:
        updated = await asyncio.to_thread(runtime.profiles.update_redaction_prefs, body)
        if updated is None:
            raise HTTPException(status_code=404, detail="Upload a resume first before configuring redaction.")
        return runtime.profiles.public_dict()

    @router.post("/api/profile/resume/toggle/{span_id}")
    async def toggle_resume_span_redaction(
        span_id: str,
        doc_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            updated = await asyncio.to_thread(
                runtime.profiles.toggle_span_redaction,
                span_id,
                expected_doc_id=doc_id,
            )
        except StaleResumeDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="Resume or span not found.")
        return runtime.profiles.public_dict()

    @router.delete("/api/profile/resume")
    async def delete_resume() -> dict[str, Any]:
        runtime.profiles.clear_resume()
        return runtime.profiles.public_dict()

    @router.post("/api/profile/jd")
    async def import_job_listing(body: JobListingImportRequest) -> dict[str, Any]:
        url = body.url.strip()
        text = body.text.strip()
        title = body.title.strip()
        company = body.company.strip()

        if url and not text:
            try:
                fetched = await fetch_job_listing(url)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Could not fetch job listing URL: {exc}") from exc
            text = fetched["text"]
            title = title or fetched["title"]
            company = company or fetched["company"]
            url = fetched["url"]
        elif text:
            text = clean_job_listing_text(text)
            if not title or not company:
                inferred_title, inferred_company = parse_job_title_and_company(title, url=url, text=text)
                title = title or inferred_title
                company = company or inferred_company
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a job listing URL or paste the job description text.",
            )

        if not text.strip():
            raise HTTPException(status_code=400, detail="Job listing text is empty after parsing.")

        jd = runtime.profiles.add_job_listing(
            title=title or "Target Role",
            company=company,
            url=url,
            text=text,
            target_level=body.target_level or DEFAULT_LEVEL,
        )
        out = runtime.profiles.public_dict()
        out["imported_jd"] = jd.model_dump()
        return out

    @router.post("/api/profile/jd/{jd_id}/select")
    async def select_job_listing(jd_id: str) -> dict[str, Any]:
        jd = runtime.profiles.select_job_listing(jd_id)
        if jd is None:
            raise HTTPException(status_code=404, detail="Job listing not found")
        return runtime.profiles.public_dict()

    @router.delete("/api/profile/jd/{jd_id}")
    async def delete_job_listing(jd_id: str) -> dict[str, Any]:
        if not runtime.profiles.delete_job_listing(jd_id):
            raise HTTPException(status_code=404, detail="Job listing not found")
        return runtime.profiles.public_dict()

    @router.get("/api/banks")
    async def list_question_banks(
        category: str | None = Query(None),
        level: str | None = Query(None),
        competency: str | None = Query(None),
        match_active_jd: bool = Query(True),
    ) -> dict[str, Any]:
        active_jd = runtime.profiles.active_jd() if match_active_jd else None
        jd_text = f"{active_jd.title} {active_jd.company} {active_jd.text}" if active_jd else None
        questions = runtime.banks.list_questions(
            category=category,
            level=level,
            competency=competency,
            jd_text=jd_text,
        )
        packs = [{k: v for k, v in p.items() if k != "questions"} for p in runtime.banks.list_packs()]
        return {
            "packs": packs,
            "questions": questions,
            "matched_jd_id": active_jd.id if active_jd else None,
        }

    @router.post("/api/banks/mine")
    async def add_my_questions(body: MyQuestionsAddRequest) -> dict[str, Any]:
        pasted = split_pasted_questions(body.text) if body.as_list else [body.text.strip()]
        prompts: list[str] = []
        removed_details: list[str] = []
        for question in pasted:
            cleaned, removed = runtime.redactor.strip_contact_details(question)
            if cleaned.strip():
                prompts.append(cleaned)
            removed_details.extend(name for name in removed if name not in removed_details)
        if not prompts:
            raise HTTPException(status_code=422, detail="Type or paste at least one question to save.")
        try:
            added, already_saved = await asyncio.to_thread(runtime.banks.add_my_questions, body.category, prompts)
        except MyQuestionsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"added": added, "already_saved": already_saved, "removed_details": removed_details}

    @router.delete("/api/banks/mine/{category}/{question_id}")
    async def delete_my_question(category: MyQuestionCategory, question_id: str) -> dict[str, Any]:
        try:
            deleted = await asyncio.to_thread(runtime.banks.delete_my_question, category, question_id)
        except MyQuestionsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="That question is not in My questions.")
        return {"ok": True}

    return router
