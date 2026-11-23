from __future__ import annotations

import asyncio
import hashlib
import html
import inspect
import io
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import threading
import unicodedata
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
from xml.etree import ElementTree as ET

import httpx
import numpy as np

from .downloads import DownloadProgress, make_hf_progress_tqdm_class
from .tts import optimal_ort_threads

logger = logging.getLogger("mockery.redactor")

RAMPART_REPO_ID = "nationaldesignstudio/rampart"
RAMPART_FILES = ("onnx/model_q4.onnx", "tokenizer.json", "config.json")
RAMPART_APPROX_MB = 15
MAX_DOCX_XML_BYTES = 10 * 1024 * 1024
MAX_JD_FETCH_BYTES = 1_500_000
MAX_SPAN_CACHE_ENTRIES = 8

LOCATION_LABELS = {"CITY", "STATE", "ZIP_CODE"}
ADDRESS_LABELS = {"BUILDING_NUMBER", "STREET_NAME", "SECONDARY_ADDRESS", "STREET_ADDRESS"}
NAME_LABELS = {"GIVEN_NAME", "SURNAME", "PERSON_NAME"}
# Contact details and identifiers removed from text the candidate wrote themselves (a custom
# practice question). Names, employers and places stay: the candidate chose to put them there.
CONTACT_DETAIL_NAMES = {
    "EMAIL": "email address",
    "PHONE": "phone number",
    "SSN": "Social Security number",
    "CREDIT_CARD": "card number",
}

_PDF_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
)

# Common resume/tech words that should never be treated as employer names or person names
_TECH_STOPWORDS = {
    "summary", "experience", "education", "skills", "projects", "publications",
    "certifications", "awards", "languages", "technologies", "frameworks",
    "tools", "interests", "leadership", "work experience", "technical skills",
    "professional experience", "selected projects", "open source", "profile",
    "engineering projects", "independent work", "personal projects", "side projects",
    "additional experience", "relevant experience", "core competencies",
    "react", "typescript", "javascript", "python", "golang", "rust", "java",
    "kubernetes", "docker", "postgres", "postgresql", "redis", "graphql",
    "aws", "gcp", "azure", "linux", "macos", "ios", "android", "next.js",
    "node.js", "fastapi", "pytorch", "tensorflow", "mlx", "onnx", "sql",
    "present", "current", "remote", "hybrid", "full-stack", "backend", "frontend",
    "software engineer", "senior software engineer", "staff software engineer",
    "principal engineer", "engineering manager", "tech lead", "software developer",
    "product manager", "data scientist", "machine learning engineer",
    "bachelor", "master", "b.s.", "m.s.", "ph.d.", "phd", "bs", "ms", "ba", "ma",
    "computer science", "electrical engineering", "mathematics", "gpa",
}

# Deterministic recognizers (America.gov Rampart Layer 1 + resume-specific contact patterns)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b")
_URL_RE = re.compile(
    r"(?:https?://[^\s<>()\"'\],;]+|"
    r"\b(?:www\.|linkedin\.com/in/|github\.com/|gitlab\.com/|x\.com/|twitter\.com/|bsky\.app/profile/)[^\s<>()\"'\],;]+)",
    re.IGNORECASE,
)
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?!\w)"
)
_CC_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|1?\d\d?)\b"
)
_STREET_RE = re.compile(
    r"\b\d{1,6}[ \t]+(?:[NSEW]\.?[ \t]+)?[A-Z][A-Za-z0-9'.\-]+(?:[ \t]+[A-Z][A-Za-z0-9'.\-]+){0,3}[ \t]+"
    r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Rd|Road|Dr(?:ive)?|Ln|Lane|Ct|Court|Way|Pl(?:ace)?|Pkwy|Parkway|Cir(?:cle)?|Ter(?:race)?)\.?"
    r"(?:,?[ \t]*(?:Apt|Suite|Ste|Unit|#)[ \t]*[A-Za-z0-9\-]+)?\b"
)
_ZIP_RE = re.compile(r"\b(?:[A-Z]{2})[ \t]+(\d{5}(?:-\d{4})?)\b")

_UNIVERSITY_RE = re.compile(
    r"\b(?:"
    r"University[ \t]+of[ \t]+[A-Z][A-Za-z&\-]+(?:[ \t]+(?:of[ \t]+)?[A-Z][A-Za-z&\-]+){0,3}"
    r"|(?:[A-Z][A-Za-z'.\-]+[ \t]+){1,4}(?:University|Institute[ \t]+of[ \t]+Technology|Polytechnic[ \t]+Institute|College|School[ \t]+of[ \t]+[A-Z][A-Za-z]+)"
    r"|(?:MIT|CMU|UCLA|UCSD|UCSB|UCSC|UC[ \t]+Berkeley|Stanford|Harvard|Yale|Princeton|Cornell|Caltech|NYU|USC|UPenn|Georgia[ \t]+Tech)"
    r")\b"
)

_DATE_RANGE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2}\s*(?:[-–—]|to)\s*"
    r"(?:Present|Current|Now|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2})",
    re.IGNORECASE,
)

_ROLE_WORDS_RE = re.compile(
    r"\b(?:engineer|developer|architect|manager|director|lead|founder|co-founder|cto|vp|"
    r"intern|scientist|researcher|analyst|designer|consultant|specialist|administrator|sre|devops)\b",
    re.IGNORECASE,
)
_CITY_STATE_RE = re.compile(
    r"\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?,[ \t]*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\b"
)
_CITY_STATE_CAPTURE_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?),[ \t]*[A-Z]{2}\b")
_CITY_STATE_EXACT_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2}$")
_HEADER_SPLIT_RE = re.compile(r"\s*[|•·—–]\s*")
_MD_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+")
_NAME_TOKEN_PUNCT = frozenset("'.-’")


def _unicode_equiv_word_pattern(word: str) -> str:
    """Build a regex matching a word regardless of NFC vs NFD Unicode normalization."""
    parts: list[str] = []
    for ch in unicodedata.normalize("NFC", word):
        nfd = unicodedata.normalize("NFD", ch)
        if nfd != ch:
            parts.append(f"(?:{re.escape(ch)}|{re.escape(nfd)})")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _is_name_token(word: str) -> bool:
    """Unicode-aware capitalized name token check (handles NFC & NFD accented letters like María Núñez)."""
    norm = unicodedata.normalize("NFC", word)
    if not (2 <= len(norm) <= 25):
        return False
    first = norm[0]
    if not first.isalpha() or not first.isupper():
        return False
    return all(
        ch.isalpha() or ch in _NAME_TOKEN_PUNCT or unicodedata.category(ch) == "Mn"
        for ch in norm[1:]
    )
_NAME_PARTICLES = {
    "al", "ben", "bin", "da", "de", "del", "della", "di", "du",
    "el", "ibn", "la", "le", "mac", "mc", "van", "von",
}
_ZIP_CLEAN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{3,10}$")
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
_SEP_SPLIT_RE = re.compile(r"\s*(?:\||—|–|·|•|\t)\s*")
_WS_RE = re.compile(r"\s+")
_NON_DIGIT_RE = re.compile(r"\D")


def _luhn_valid(digits: str) -> bool:
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total = 0
    rev = digits[::-1]
    for i, ch in enumerate(rev):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def extract_document_text(filename: str, raw_bytes: bytes) -> str:
    """Extract readable plain text from PDF, DOCX, TXT, MD, or RTF bytes entirely locally."""
    if not raw_bytes:
        return ""
    ext = Path(filename or "").suffix.lower()

    if ext == ".pdf" or raw_bytes[:4] == b"%PDF":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages: list[str] = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append(txt.strip())
            text = "\n\n".join(pages)
            return _normalize_extracted_whitespace(text)
        except Exception as exc:
            raise ValueError(
                f"Could not read PDF file (it may be corrupted or password-protected): {exc}"
            ) from exc

    if ext == ".docx" or raw_bytes[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                if "word/document.xml" in zf.namelist():
                    with zf.open("word/document.xml") as zentry:
                        xml_bytes = zentry.read(MAX_DOCX_XML_BYTES + 1)
                    if len(xml_bytes) > MAX_DOCX_XML_BYTES:
                        raise ValueError("DOCX document XML exceeds safe decompression size limit.")
                    root = ET.fromstring(xml_bytes)
                    paras: list[str] = []
                    for p in root.iter():
                        if p.tag.endswith("}p"):
                            chunks = [
                                node.text
                                for node in p.iter()
                                if node.tag.endswith("}t") and node.text
                            ]
                            line = "".join(chunks).strip()
                            if line:
                                paras.append(line)
                    return _normalize_extracted_whitespace("\n".join(paras))
        except (zipfile.BadZipFile, ET.ParseError):
            pass

    decoded = raw_bytes.decode("utf-8", errors="replace")
    if ext == ".rtf" or decoded.lstrip().startswith("{\\rtf"):
        decoded = re.sub(r"\\[a-z]+-?\d* ?", " ", decoded)
        decoded = re.sub(r"[{}]", " ", decoded)
    return _normalize_extracted_whitespace(decoded)


def _normalize_extracted_whitespace(text: str) -> str:
    text = text.translate(_PDF_LIGATURES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------- SSRF-Safe Job Listing Fetcher ----------------


def _resolve_public_http_url(url: str) -> tuple[str, str, int, str]:
    """Validate URL scheme/host, resolve DNS once, ensure every IP is public, and return a pinned IP."""
    clean = (url or "").strip()
    parsed = urlparse(clean)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please provide a valid http:// or https:// job listing URL.")
    if parsed.username or parsed.password:
        raise ValueError("Refusing to fetch URLs containing embedded credentials.")
    host = parsed.hostname.lower().strip("[]")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local") or host.endswith(".internal"):
        raise ValueError("Refusing to fetch private or local network URLs.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname '{host}': {exc}") from exc

    pinned_ip: str | None = None
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip_str = sockaddr[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if not ip_obj.is_global:
            raise ValueError("Refusing to fetch private or non-public IP address.")
        if pinned_ip is None:
            pinned_ip = ip_str
    if not pinned_ip:
        raise ValueError(f"Hostname '{host}' resolved to no addresses.")
    return clean, host, port, pinned_ip


def _build_pinned_ip_url(url: str, pinned_ip: str, port: int) -> str:
    parsed = urlparse(url)
    ip_host = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
    default_port = 443 if parsed.scheme == "https" else 80
    netloc = f"{ip_host}:{port}" if port != default_port else ip_host
    return urlunparse((parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, ""))


class _JobHTMLParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "form", "nav", "footer", "button", "select", "option"}
    _BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "br", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self._in_title: bool = False
        self._skip_depth: int = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t == "title":
            self._in_title = True
        elif t in ("h1", "h2", "h3"):
            self._chunks.append("\n\n## ")
        elif t == "li":
            self._chunks.append("\n- ")
        elif t in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if t == "title":
            self._in_title = False
        elif t in self._BLOCK_TAGS and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
            return
        cleaned = re.sub(r"\s+", " ", data)
        if cleaned.strip():
            self._chunks.append(cleaned)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = [ln.strip() for ln in raw.splitlines()]
        out: list[str] = []
        for ln in lines:
            if not ln:
                if out and out[-1] != "":
                    out.append("")
                continue
            if ln in ("##", "-"):
                continue
            out.append(ln)
        return "\n".join(out).strip()


_ATS_FOOTER_MARKERS = (
    "create a job alert",
    "agreement to arbitrate",
    "for government reporting purposes",
    "equal employment opportunity policy",
    "equal opportunity employer",
    "voluntary self-identification",
    "apply for this job",
    "powered by greenhouse",
)


def clean_job_listing_text(raw_text: str) -> str:
    """Trim noisy ATS application forms, arbitration clauses, and EEO survey boilerplate."""
    text = _normalize_extracted_whitespace(html.unescape(raw_text or ""))
    lines = text.splitlines()
    kept: list[str] = []
    for ln in lines:
        low = ln.strip().lower()
        if any(marker in low for marker in _ATS_FOOTER_MARKERS):
            break
        if low in ("back to jobs", "apply", "apply now"):
            continue
        kept.append(ln)
    cleaned = "\n".join(kept).strip()
    return cleaned if len(cleaned) >= 120 else text


def parse_job_title_and_company(title_raw: str, url: str = "", text: str = "") -> tuple[str, str]:
    title = re.sub(r"\s+", " ", (title_raw or "")).strip()
    company = ""

    # Common Greenhouse pattern: "Job Application for <Role> at <Company>"
    m = re.match(r"^Job Application for\s+(.+?)\s+at\s+(.+)$", title, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Common "<Role> at <Company>" or "<Role> - <Company>" or "<Role> | <Company>"
    m = re.match(r"^(.+?)\s+(?:at|@)\s+(.+?)(?:\s*[|\-–].*)?$", title, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    for sep in (" | ", " - ", " — ", " – "):
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

    if url:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path_parts = [p for p in parsed.path.split("/") if p]
        if ("greenhouse.io" in host or "lever.co" in host or "ashbyhq.com" in host) and path_parts:
            company = path_parts[0].replace("-", " ").title()

    if not title and text:
        first_line = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        title = first_line[:90] or "Target Role"
    return title or "Target Role", company


async def fetch_job_listing(url: str, *, timeout: float = 12.0) -> dict[str, Any]:
    """Fetch and clean a public job listing URL with pinned-IP SSRF protection on initial URL and redirects."""
    current_url, host, port, pinned_ip = await asyncio.to_thread(_resolve_public_http_url, url)
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Mockery/0.2",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }
    body_bytes = bytearray()
    content_type = ""
    encoding = "utf-8"

    async with asyncio.timeout(timeout + 3.0):
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for hop in range(5):
                pinned_url = _build_pinned_ip_url(current_url, pinned_ip, port)
                parsed = urlparse(current_url)
                default_port = 443 if parsed.scheme == "https" else 80
                host_header = f"{host}:{port}" if port != default_port else host
                req_headers = {**base_headers, "Host": host_header}
                req = client.build_request(
                    "GET",
                    pinned_url,
                    headers=req_headers,
                    extensions={"sni_hostname": host} if parsed.scheme == "https" else None,
                )
                async with client.stream(req.method, req.url, headers=req.headers, extensions=req.extensions) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("location")
                        if not loc:
                            raise RuntimeError(f"Job page returned redirect HTTP {resp.status_code} without Location")
                        if hop == 4:
                            raise RuntimeError("Job listing URL exceeded maximum redirect limit.")
                        next_url = urljoin(current_url, loc)
                        current_url, host, port, pinned_ip = await asyncio.to_thread(
                            _resolve_public_http_url, next_url
                        )
                        continue
                    if resp.status_code >= 400:
                        raise RuntimeError(f"Job page returned HTTP {resp.status_code}")
                    content_type = (resp.headers.get("content-type") or "").lower()
                    encoding = resp.encoding or "utf-8"
                    async for chunk in resp.aiter_bytes(chunk_size=16_384):
                        remaining = MAX_JD_FETCH_BYTES - len(body_bytes)
                        if remaining <= 0:
                            break
                        body_bytes.extend(chunk[:remaining])
                    break

    body = body_bytes.decode(encoding, errors="replace")[:400_000]

    if "html" in content_type or "<html" in body[:500].lower():
        parser = _JobHTMLParser()
        parser.feed(body)
        title_raw = parser.title
        cleaned_text = clean_job_listing_text(parser.get_text())
    else:
        title_raw = ""
        cleaned_text = clean_job_listing_text(body)

    role_title, company = parse_job_title_and_company(title_raw, url=current_url, text=cleaned_text)
    return {
        "url": current_url,
        "title": role_title,
        "company": company,
        "text": cleaned_text[:20_000],
    }


# ---------------- Rampart Local ONNX PII Redactor ----------------


@dataclass
class DetectedSpan:
    start: int
    end: int
    label: str
    source: str  # "rules" | "rampart" | "org" | "custom"
    original: str = ""
    placeholder: str = ""
    active: bool = True
    span_id: str = ""


@dataclass
class RedactionResult:
    redacted_text: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    engine: str = "rules_only"
    rampart_installed: bool = False
    entity_counts: dict[str, int] = field(default_factory=dict)
    redacted_count: int = 0
    total_detected: int = 0


def _normalize_for_rampart(text: str) -> tuple[str, list[int]]:
    """
    NFKD decompose, strip combining marks, lowercase, while maintaining a character-index map
    back into the original string so Rampart offsets align cleanly with original text.
    """
    norm_chars: list[str] = []
    idx_map: list[int] = []
    for orig_idx, ch in enumerate(text):
        decomp = unicodedata.normalize("NFKD", ch)
        for d in decomp:
            if unicodedata.category(d) == "Mn":
                continue
            norm_chars.append(d.lower())
            idx_map.append(orig_idx)
    return "".join(norm_chars), idx_map


def _hf_hub_cache_dir() -> Path:
    hub_env = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_env:
        return Path(hub_env)
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub"


class RampartRedactor:
    """
    America.gov-style hybrid local PII redactor:
      1. Deterministic regex/Luhn rules (SSN, credit card, email, phone, URLs, IPv4, street address)
      2. Optional 14.7 MB Rampart 4-bit ONNX MiniLM token classifier (`nationaldesignstudio/rampart`)
      3. Optional resume employer & school anonymizer + custom scrub terms + interactive un-redaction
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Any | None = None
        self._input_names: set[str] = set()
        self._tokenizer: Any | None = None
        self._id2label: dict[int, str] | None = None
        self._last_error: str | None = None
        self._cached_paths_key: str | None = None
        self._cached_paths: dict[str, Path] | None = None
        self._span_cache: OrderedDict[str, tuple[tuple[int, int, str, str], ...]] = OrderedDict()

    def _fast_snapshot_paths(self, hub_dir: Path) -> dict[str, Path] | None:
        repo_dir = hub_dir / f"models--{RAMPART_REPO_ID.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None

        candidate_dirs: list[Path] = []
        ref_main = repo_dir / "refs" / "main"
        if ref_main.is_file():
            try:
                rev = ref_main.read_text(encoding="utf-8").strip()
                if rev:
                    candidate_dirs.append(snapshots_dir / rev)
            except OSError:
                pass
        try:
            with os.scandir(snapshots_dir) as it:
                for entry in it:
                    if entry.is_dir():
                        p = Path(entry.path)
                        if p not in candidate_dirs:
                            candidate_dirs.append(p)
        except OSError:
            pass

        for snap in candidate_dirs:
            resolved: dict[str, Path] = {}
            ok = True
            for rel in RAMPART_FILES:
                fp = snap / rel
                if fp.is_file():
                    resolved[rel] = fp
                else:
                    ok = False
                    break
            if ok:
                return resolved
        return None

    def _local_files_paths(self) -> dict[str, Path] | None:
        hub_dir = _hf_hub_cache_dir()
        cache_key = str(hub_dir)
        cached = self._cached_paths
        if cached is not None and self._cached_paths_key == cache_key:
            if all(p.is_file() for p in cached.values()):
                return cached
            self._cached_paths = None
            self._cached_paths_key = None

        fast = self._fast_snapshot_paths(hub_dir)
        if fast is not None:
            self._cached_paths = fast
            self._cached_paths_key = cache_key
            return fast

        repo_dir = hub_dir / f"models--{RAMPART_REPO_ID.replace('/', '--')}"
        if not repo_dir.is_dir():
            return None

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None
        paths: dict[str, Path] = {}
        for rel in RAMPART_FILES:
            try:
                p = hf_hub_download(RAMPART_REPO_ID, rel, local_files_only=True)
                paths[rel] = Path(p)
            except Exception:
                return None
        self._cached_paths = paths
        self._cached_paths_key = cache_key
        return paths

    def is_downloaded(self) -> bool:
        return self._local_files_paths() is not None

    def status_dict(self) -> dict[str, Any]:
        downloaded = self.is_downloaded()
        return {
            "repo_id": RAMPART_REPO_ID,
            "installed": downloaded,
            "loaded": self._session is not None,
            "ready": downloaded and self._last_error is None,
            "size_mb": RAMPART_APPROX_MB,
            "error": self._last_error,
        }

    def ensure_downloaded(
        self,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download

        tqdm_cls = (
            make_hf_progress_tqdm_class(
                on_progress,
                approx_total_bytes=RAMPART_APPROX_MB * 1024 * 1024,
                cancel_event=cancel_event,
            )
            if (on_progress is not None or cancel_event is not None)
            else None
        )
        supports_tqdm = False
        if tqdm_cls is not None:
            try:
                params = inspect.signature(hf_hub_download).parameters
                supports_tqdm = "tqdm_class" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                supports_tqdm = False
        for rel in RAMPART_FILES:
            if cancel_event is not None and cancel_event.is_set():
                from .downloads import DownloadCancelledError

                raise DownloadCancelledError("Download cancelled.")
            if tqdm_cls is not None and supports_tqdm:
                hf_hub_download(RAMPART_REPO_ID, rel, tqdm_class=tqdm_cls)
            else:
                hf_hub_download(RAMPART_REPO_ID, rel)
        if cancel_event is not None and cancel_event.is_set():
            from .downloads import DownloadCancelledError

            raise DownloadCancelledError("Download cancelled.")
        if on_progress is not None:
            on_progress(DownloadProgress(phase="loading", percent=100.0, cancellable=False).to_dict())
        with self._lock:
            self._cached_paths = None
            self._cached_paths_key = None
        if not self._ensure_loaded():
            raise RuntimeError(self._last_error or "Rampart ONNX model downloaded but failed to initialize.")
        return self.status_dict()

    def delete_cache(self) -> bool:
        with self._lock:
            self._session = None
            self._input_names = set()
            self._tokenizer = None
            self._id2label = None
            self._cached_paths = None
            self._cached_paths_key = None
            self._span_cache.clear()
        hub_dir = _hf_hub_cache_dir()
        try:
            from huggingface_hub import scan_cache_dir

            cache = scan_cache_dir(cache_dir=hub_dir)
            revs = []
            for repo in cache.repos:
                if repo.repo_id == RAMPART_REPO_ID:
                    revs.extend(r.commit_hash for r in repo.revisions)
            if revs:
                strategy = cache.delete_revisions(*revs)
                strategy.execute()
                return True
        except Exception as exc:
            logger.warning("Could not delete Rampart cache cleanly: %s", exc)
        # Fallback: remove repo folder directly under HF hub cache if present
        repo_dir = hub_dir / f"models--{RAMPART_REPO_ID.replace('/', '--')}"
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
            return True
        return False

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        paths = self._local_files_paths()
        if paths is None:
            return False
        with self._lock:
            if self._session is not None:
                return True
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(str(paths["tokenizer.json"]))
                tok.enable_truncation(max_length=512, stride=64)
                cfg = json.loads(paths["config.json"].read_text(encoding="utf-8"))
                id2label = {int(k): str(v) for k, v in cfg["id2label"].items()}
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = optimal_ort_threads()
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    str(paths["onnx/model_q4.onnx"]),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._tokenizer = tok
                self._id2label = id2label
                self._input_names = {inp.name for inp in sess.get_inputs()}
                self._session = sess
                self._last_error = None
                return True
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Failed to load Rampart ONNX model: %s", exc)
                return False

    def _run_rampart_onnx(self, text: str) -> list[DetectedSpan]:
        if not text.strip() or not self._ensure_loaded():
            return []

        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            sess = self._session
            tok = self._tokenizer
            id2label = self._id2label
            input_names = set(self._input_names) if self._input_names else None
            if sess is None or tok is None or id2label is None:
                return []
            cached = self._span_cache.get(cache_key)
            if cached is not None:
                self._span_cache.move_to_end(cache_key)
                return [DetectedSpan(s, e, lbl, src) for s, e, lbl, src in cached]

        norm_text, idx_map = _normalize_for_rampart(text)
        if not norm_text:
            return []

        if not input_names:
            input_names = {inp.name for inp in sess.get_inputs()}
        enc = tok.encode(norm_text)
        encodings = [enc] + list(getattr(enc, "overflowing", []) or [])

        raw_spans: list[DetectedSpan] = []
        for sub_enc in encodings:
            ids = sub_enc.ids
            if not ids:
                continue
            feeds: dict[str, Any] = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([sub_enc.attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.array([sub_enc.type_ids], dtype=np.int64)

            logits = sess.run(None, feeds)[0][0]  # [seq_len, num_labels]
            pred_ids = np.argmax(logits, axis=-1)

            cur_label: str | None = None
            cur_start = -1
            cur_end = -1

            def _flush() -> None:
                nonlocal cur_label, cur_start, cur_end
                if cur_label is not None and 0 <= cur_start < cur_end <= len(idx_map):
                    o_start = idx_map[cur_start]
                    o_end = idx_map[cur_end - 1] + 1
                    # Expand slightly to include trailing combining marks if any
                    while o_end < len(text) and unicodedata.category(text[o_end]) == "Mn":
                        o_end += 1
                    raw_spans.append(
                        DetectedSpan(start=o_start, end=o_end, label=cur_label, source="rampart")
                    )
                cur_label = None
                cur_start = -1
                cur_end = -1

            for (s, e), lid in zip(sub_enc.offsets, pred_ids):
                if s >= e:
                    _flush()
                    continue
                tag = id2label.get(int(lid), "O")
                if tag == "O":
                    _flush()
                    continue
                if "-" in tag:
                    prefix, ent = tag.split("-", 1)
                else:
                    prefix, ent = "B", tag

                crossed_newline = (
                    cur_label is not None
                    and 0 <= cur_end <= s <= len(norm_text)
                    and "\n" in norm_text[cur_end:s]
                )
                if prefix == "B" or cur_label != ent or s > cur_end + 1 or crossed_newline:
                    _flush()
                    cur_label = ent
                    cur_start = s
                    cur_end = e
                else:
                    cur_end = max(cur_end, e)
            _flush()

        # Span repair: merge adjacent same-group spans separated by short whitespace/hyphens/apostrophes
        raw_spans.sort(key=lambda sp: (sp.start, -sp.end))
        repaired: list[DetectedSpan] = []
        for sp in raw_spans:
            frag = text[sp.start : sp.end].strip()
            if not frag or frag.lower() in _TECH_STOPWORDS:
                continue
            if repaired:
                prev = repaired[-1]
                same_group = (
                    prev.label == sp.label
                    or (prev.label in NAME_LABELS and sp.label in NAME_LABELS)
                    or (prev.label in ADDRESS_LABELS and sp.label in ADDRESS_LABELS)
                )
                gap = text[prev.end : sp.start] if sp.start >= prev.end else ""
                if sp.start <= prev.end and same_group:
                    prev.end = max(prev.end, sp.end)
                    if prev.label in NAME_LABELS and sp.label in NAME_LABELS:
                        prev.label = "PERSON_NAME"
                    elif prev.label in ADDRESS_LABELS and sp.label in ADDRESS_LABELS:
                        prev.label = "STREET_ADDRESS"
                    continue
                if same_group and gap and len(gap) <= 2 and all(c in " -'." for c in gap):
                    prev.end = sp.end
                    if prev.label in NAME_LABELS and sp.label in NAME_LABELS:
                        prev.label = "PERSON_NAME"
                    elif prev.label in ADDRESS_LABELS and sp.label in ADDRESS_LABELS:
                        prev.label = "STREET_ADDRESS"
                    continue
            repaired.append(sp)

        # Normalize single-token GIVEN_NAME/SURNAME and filter obvious uncased-BERT false positives
        cleaned: list[DetectedSpan] = []
        for sp in repaired:
            frag = text[sp.start : sp.end].strip()
            if len(frag) <= 1:
                continue
            if sp.label in NAME_LABELS:
                sp.label = "PERSON_NAME"
            elif sp.label in ADDRESS_LABELS:
                sp.label = "STREET_ADDRESS"
            if sp.label == "PERSON_NAME":
                if frag.lower() in _TECH_STOPWORDS:
                    continue
                words = frag.split()
                if (
                    len(words) >= 2
                    and words[0][0].islower()
                    and words[0].lower() not in _NAME_PARTICLES
                    and any(w[0].isupper() for w in words[1:] if w)
                ):
                    continue
            elif sp.label == "ZIP_CODE":
                trailing = text[sp.end : sp.end + 1]
                if "." in frag or trailing == "%" or not _ZIP_CLEAN_RE.match(frag):
                    continue
                if sum(ch.isdigit() for ch in frag) < 3:
                    continue
            cleaned.append(sp)

        frozen = tuple((sp.start, sp.end, sp.label, sp.source) for sp in cleaned)
        with self._lock:
            self._span_cache[cache_key] = frozen
            self._span_cache.move_to_end(cache_key)
            while len(self._span_cache) > MAX_SPAN_CACHE_ENTRIES:
                self._span_cache.popitem(last=False)
        return cleaned

    def _run_deterministic_rules(self, text: str) -> list[DetectedSpan]:
        spans: list[DetectedSpan] = []
        for m in _EMAIL_RE.finditer(text):
            spans.append(DetectedSpan(m.start(), m.end(), "EMAIL", "rules"))
        for m in _URL_RE.finditer(text):
            end = m.end()
            while end > m.start() and text[end - 1] in ".,;:)>]}":
                end -= 1
            spans.append(DetectedSpan(m.start(), end, "URL", "rules"))
        for m in _SSN_RE.finditer(text):
            spans.append(DetectedSpan(m.start(), m.end(), "SSN", "rules"))
        for m in _PHONE_RE.finditer(text):
            spans.append(DetectedSpan(m.start(), m.end(), "PHONE", "rules"))
        for m in _IPV4_RE.finditer(text):
            spans.append(DetectedSpan(m.start(), m.end(), "IP_ADDRESS", "rules"))
        for m in _CC_CANDIDATE_RE.finditer(text):
            digits = _NON_DIGIT_RE.sub("", m.group(0))
            if _luhn_valid(digits):
                spans.append(DetectedSpan(m.start(), m.end(), "CREDIT_CARD", "rules"))
        for m in _STREET_RE.finditer(text):
            spans.append(DetectedSpan(m.start(), m.end(), "STREET_ADDRESS", "rules"))
        for m in _ZIP_RE.finditer(text):
            spans.append(DetectedSpan(m.start(1), m.end(1), "ZIP_CODE", "rules"))
        for m in _CITY_STATE_RE.finditer(text):
            spans.append(DetectedSpan(m.start(), m.end(), "CITY", "rules"))
        return spans

    def strip_contact_details(self, text: str) -> tuple[str, list[str]]:
        """
        Light redaction for text the candidate wrote themselves: replace email addresses, phone
        numbers, Social Security numbers and card numbers with a plain note, and leave every other
        word as written. Returns the cleaned text and the kinds of detail removed, in reading order.
        """
        spans = sorted(
            (sp for sp in self._run_deterministic_rules(text) if sp.label in CONTACT_DETAIL_NAMES),
            key=lambda sp: (sp.start, -(sp.end - sp.start)),
        )
        parts: list[str] = []
        removed: list[str] = []
        cursor = 0
        for sp in spans:
            if sp.start < cursor:
                continue
            name = CONTACT_DETAIL_NAMES[sp.label]
            parts.append(text[cursor : sp.start])
            parts.append(f"[{name} removed]")
            if name not in removed:
                removed.append(name)
            cursor = sp.end
        parts.append(text[cursor:])
        return "".join(parts), removed

    def _detect_header_candidate_name(self, text: str) -> list[DetectedSpan]:
        """
        Deterministic resume-header name recognizer (always run alongside Rampart):
        the first non-empty line of a resume (2–4 capitalized name tokens without digits/role words)
        is reliably the candidate's name, and any exact repeated occurrence of that full name
        later in the document (e.g. summary or page-2 header) is redacted as well.
        """
        offset = 0
        for line in text.splitlines(keepends=True)[:4]:
            stripped = line.strip()
            if not stripped:
                offset += len(line)
                continue
            cleaned = _MD_HEADING_PREFIX_RE.sub("", stripped)
            # Take prefix before '|' or '•' or '—' if contact info shares line 1
            head = _HEADER_SPLIT_RE.split(cleaned)[0].strip()
            words = head.split()
            if (
                2 <= len(words) <= 4
                and not _ROLE_WORDS_RE.search(head)
                and head.lower() not in _TECH_STOPWORDS
                and all(_is_name_token(w) for w in words)
            ):
                pos = line.find(head)
                if pos >= 0:
                    header_start = offset + pos
                    header_end = header_start + len(head)
                    spans = [DetectedSpan(header_start, header_end, "PERSON_NAME", "rules")]
                    # Propagate exact full-name occurrences elsewhere in the resume (NFC & NFD agnostic)
                    name_pat = re.compile(
                        r"(?<!\w)" + r"\s+".join(_unicode_equiv_word_pattern(w) for w in words) + r"(?!\w)",
                        re.IGNORECASE,
                    )
                    for m in name_pat.finditer(text):
                        if m.start() == header_start and m.end() == header_end:
                            continue
                        spans.append(DetectedSpan(m.start(), m.end(), "PERSON_NAME", "rules"))
                    return spans
            break
        return []

    def _detect_employers_and_schools(
        self,
        text: str,
        *,
        location_spans: list[DetectedSpan] | None = None,
    ) -> list[DetectedSpan]:
        """Optional resume pass to anonymize university names and employer names."""
        spans: list[DetectedSpan] = []
        loc_ranges = [(sp.start, sp.end) for sp in (location_spans or [])]
        loc_strings = {
            text[sp.start : sp.end].strip().lower()
            for sp in (location_spans or [])
            if sp.end > sp.start
        }
        for cm in _CITY_STATE_CAPTURE_RE.finditer(text):
            loc_strings.add(cm.group(1).strip().lower())

        # 1. Universities / schools
        for m in _UNIVERSITY_RE.finditer(text):
            frag = m.group(0).strip()
            if frag.lower() not in _TECH_STOPWORDS:
                spans.append(DetectedSpan(m.start(), m.end(), "SCHOOL", "org"))

        # 2. Employer names on resume experience header lines (lines containing date ranges or `|`/`—` separators)
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or len(stripped) > 140 or stripped.startswith(("-", "•", "*", "–")):
                offset += len(line)
                continue
            has_date = bool(_DATE_RANGE_RE.search(stripped))
            if has_date or (" | " in stripped) or (" — " in stripped) or (" – " in stripped):
                # Strip date range and split into segments
                without_dates = _EMPTY_PARENS_RE.sub("", _DATE_RANGE_RE.sub("", stripped))
                parts = [p.strip(" ,|-–—•·()") for p in _SEP_SPLIT_RE.split(without_dates)]
                for part in parts:
                    if not part or len(part) < 2 or len(part) > 48:
                        continue
                    low = part.lower()
                    # Skip ALL-CAPS multi-word section headers (e.g. "ENGINEERING PROJECTS")
                    if part.isupper() and " " in part and len(part) >= 8:
                        continue
                    if (
                        low in _TECH_STOPWORDS
                        or low in loc_strings
                        or _ROLE_WORDS_RE.search(part)
                        or _EMAIL_RE.search(part)
                        or _PHONE_RE.search(part)
                        or _URL_RE.search(part)
                        or _CITY_STATE_EXACT_RE.match(part)  # City, ST
                    ):
                        continue
                    # Likely a company name if Title Case / brand token (1-4 words)
                    words = part.split()
                    if 1 <= len(words) <= 4 and any(w[0].isupper() for w in words if w):
                        pos = line.find(part)
                        if pos >= 0:
                            s_abs = offset + pos
                            e_abs = s_abs + len(part)
                            if any(not (e_abs <= ls or s_abs >= le) for ls, le in loc_ranges):
                                continue
                            spans.append(DetectedSpan(s_abs, e_abs, "COMPANY", "org"))
            offset += len(line)

        return spans

    def _detect_custom_terms(self, text: str, custom_terms: list[str]) -> list[DetectedSpan]:
        spans: list[DetectedSpan] = []
        for term in custom_terms:
            clean = term.strip()
            if len(clean) < 2:
                continue
            pat = re.compile(re.escape(clean), re.IGNORECASE)
            for m in pat.finditer(text):
                spans.append(DetectedSpan(m.start(), m.end(), "CUSTOM", "custom"))
        return spans

    def redact(
        self,
        text: str,
        *,
        keep_location: bool = True,
        hide_employers_and_schools: bool = False,
        custom_terms: list[str] | None = None,
        unredacted_values: list[str] | None = None,
        use_rampart: bool = True,
    ) -> RedactionResult:
        downloaded = self.is_downloaded()
        if not text:
            return RedactionResult(
                redacted_text="",
                engine="rampart+rules" if downloaded else "rules_only",
                rampart_installed=downloaded,
            )

        rampart_spans: list[DetectedSpan] = []
        used_rampart = False
        if use_rampart and downloaded:
            try:
                rampart_spans = self._run_rampart_onnx(text)
                used_rampart = self._session is not None or bool(rampart_spans)
            except Exception as exc:
                logger.warning("Rampart ONNX pass failed (%s); falling back to deterministic rules.", exc)
                rampart_spans = []
                used_rampart = False

        rule_spans = self._run_deterministic_rules(text)
        # Always run deterministic header-name detection alongside Rampart so a false-positive
        # or partial Rampart PERSON_NAME early in the resume (e.g. "Francisco" in "San Francisco, CA"
        # or a surname-only hit) can never suppress redaction of the candidate's full name.
        rule_spans.extend(self._detect_header_candidate_name(text))

        # Suppress Rampart sub-spans contained inside deterministic city/ZIP locations
        # (e.g. Rampart tagging "Francisco" inside "San Francisco, CA" as PERSON_NAME).
        deterministic_loc_ranges = [
            (sp.start, sp.end) for sp in rule_spans if sp.label in LOCATION_LABELS
        ]
        if deterministic_loc_ranges and rampart_spans:
            rampart_spans = [
                sp
                for sp in rampart_spans
                if sp.label in LOCATION_LABELS
                or not any(ls <= sp.start and sp.end <= le for ls, le in deterministic_loc_ranges)
            ]

        location_spans = [sp for sp in rampart_spans if sp.label in LOCATION_LABELS]

        all_spans: list[DetectedSpan] = []
        all_spans.extend(rule_spans)
        all_spans.extend(rampart_spans)
        if hide_employers_and_schools:
            all_spans.extend(self._detect_employers_and_schools(text, location_spans=location_spans))
        if custom_terms:
            all_spans.extend(self._detect_custom_terms(text, custom_terms))

        # Respect keep_location policy (America.gov Rampart keeps CITY/STATE/ZIP_CODE by default)
        if keep_location:
            all_spans = [sp for sp in all_spans if sp.label not in LOCATION_LABELS]

        # Priority order when resolving overlapping spans: custom > rules > rampart > org
        source_priority = {"custom": 4, "rules": 3, "rampart": 2, "org": 1}
        all_spans.sort(key=lambda sp: (sp.start, -source_priority.get(sp.source, 0), -(sp.end - sp.start)))

        merged: list[DetectedSpan] = []
        for sp in all_spans:
            if not (0 <= sp.start < sp.end <= len(text)):
                continue
            frag = text[sp.start : sp.end].strip()
            if not frag:
                continue
            if merged and sp.start < merged[-1].end:
                prev = merged[-1]
                prev_len = prev.end - prev.start
                sp_len = sp.end - sp.start
                union_end = max(prev.end, sp.end)
                # Preserve full union coverage; only override label/source if incoming span
                # has higher priority AND covers at least as much as prev (never shrink PII).
                if source_priority.get(sp.source, 0) > source_priority.get(prev.source, 0) and sp_len >= prev_len:
                    prev.label = sp.label
                    prev.source = sp.source
                prev.end = union_end
                continue
            merged.append(sp)

        unredacted_norm = {
            _WS_RE.sub(" ", v.strip()).lower()
            for v in (unredacted_values or [])
            if v and v.strip()
        }

        # Assign stable deterministic placeholders by (label, normalized_original)
        label_counters: dict[str, int] = {}
        value_to_placeholder: dict[tuple[str, str], str] = {}
        entity_counts: dict[str, int] = {}

        out_parts: list[str] = []
        cursor = 0
        serialized_spans: list[dict[str, Any]] = []
        redacted_count = 0

        for idx, sp in enumerate(merged):
            orig = text[sp.start : sp.end]
            norm_val = _WS_RE.sub(" ", orig.strip()).lower()
            is_active = norm_val not in unredacted_norm

            key = (sp.label, norm_val)
            if key not in value_to_placeholder:
                label_counters[sp.label] = label_counters.get(sp.label, 0) + 1
                num = label_counters[sp.label]
                value_to_placeholder[key] = f"[{sp.label}_{num}]"
            placeholder = value_to_placeholder[key]

            out_parts.append(text[cursor : sp.start])
            if is_active:
                out_parts.append(placeholder)
                redacted_count += 1
                entity_counts[sp.label] = entity_counts.get(sp.label, 0) + 1
            else:
                out_parts.append(orig)
            cursor = sp.end

            serialized_spans.append(
                {
                    "id": f"pii_{idx}",
                    "start": sp.start,
                    "end": sp.end,
                    "original": orig,
                    "placeholder": placeholder,
                    "label": sp.label,
                    "source": sp.source,
                    "active": is_active,
                }
            )

        out_parts.append(text[cursor:])
        return RedactionResult(
            redacted_text="".join(out_parts),
            spans=serialized_spans,
            engine="rampart+rules" if used_rampart else "rules_only",
            rampart_installed=downloaded,
            entity_counts=entity_counts,
            redacted_count=redacted_count,
            total_detected=len(serialized_spans),
        )
