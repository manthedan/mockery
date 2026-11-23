"""Shared verified asset downloader with retries, HTTP Range resume, progress tracking, and cancellation."""

from __future__ import annotations

import email.utils
import hashlib
import logging
import math
import os
import random
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("mockery.downloads")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_S = 1.0
MAX_RETRY_AFTER_S = 30.0

_QUERY_URL_RE = re.compile(r"(https?://[^\s'\"<>?#]+)\?[^\s'\"<>]+")
_MDN_SUFFIX_RE = re.compile(r"\s*For more information check:\s*https?://\S+\s*", re.IGNORECASE)
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
_CONTENT_RANGE_UNSAT_RE = re.compile(r"^bytes\s+\*/(\d+)$", re.IGNORECASE)


class DownloadCancelledError(Exception):
    """Raised when an in-flight download is cancelled by the user."""


@dataclass
class DownloadProgress:
    phase: str = "downloading"  # "downloading" | "retrying" | "verifying" | "loading"
    bytes_done: int = 0
    bytes_total: int | None = None
    percent: float | None = None
    speed_bps: float | None = None
    eta_seconds: float | None = None
    file_index: int = 1
    file_count: int = 1
    current_file: str = ""
    attempt: int = 1
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    retry_note: str | None = None
    cancellable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "bytes_done": int(max(0, self.bytes_done)),
            "bytes_total": int(self.bytes_total) if self.bytes_total and self.bytes_total > 0 else None,
            "percent": round(max(0.0, min(100.0, float(self.percent))), 1) if self.percent is not None else None,
            "speed_bps": int(round(self.speed_bps)) if self.speed_bps and self.speed_bps > 0 else None,
            "eta_seconds": int(round(self.eta_seconds)) if self.eta_seconds is not None and self.eta_seconds >= 0 else None,
            "file_index": self.file_index,
            "file_count": self.file_count,
            "current_file": self.current_file,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "retry_note": self.retry_note,
            "cancellable": bool(self.cancellable),
        }


def sanitize_error_text(text: str) -> str:
    """Strip signed CDN query strings and httpx MDN links from error text."""
    cleaned = _MDN_SUFFIX_RE.sub(" ", str(text or ""))
    cleaned = _QUERY_URL_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _host_label(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return host or "remote server"


def format_download_error(exc: BaseException, *, label: str = "") -> str:
    """Produce a concise, user-friendly download error without raw signed CDN URLs."""
    target = f" for {label}" if label else ""
    if isinstance(exc, DownloadCancelledError):
        return "Download cancelled."
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        reason = (exc.response.reason_phrase or "").strip()
        host = _host_label(str(exc.request.url))
        status_text = f"HTTP {code} {reason}".strip()
        return f"Download failed{target} ({status_text} from {host}) — please retry."
    if isinstance(exc, httpx.TimeoutException):
        return f"Download timed out{target} — check your connection and retry."
    if isinstance(exc, httpx.TransportError):
        what = f" {label}" if label else ""
        return f"Network error while downloading{what} — check your connection and retry."
    msg = sanitize_error_text(str(exc))
    if not msg:
        return f"Download failed{target} — please retry."
    return msg


def _origin_tuple(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if not scheme or not host:
        return None
    try:
        port = parsed.port or (443 if scheme == "https" else 80 if scheme == "http" else 0)
    except ValueError:
        return None
    if port <= 0:
        return None
    return (scheme, host, port)


def _is_trusted_hf_origin(url: str) -> bool:
    """Return True only for explicit HTTPS Hugging Face Hub origins (or explicit HTTPS HF_ENDPOINT)."""
    origin = _origin_tuple(url)
    if origin is None or origin[0] != "https":
        return False
    if origin in (("https", "huggingface.co", 443), ("https", "www.huggingface.co", 443), ("https", "hf.co", 443)):
        return True
    endpoint = (os.getenv("HF_ENDPOINT") or "").strip()
    if endpoint:
        ep_origin = _origin_tuple(endpoint)
        if ep_origin is not None and ep_origin[0] == "https" and origin == ep_origin:
            return True
    return False


def _implicit_hf_token_disabled() -> bool:
    raw = (os.getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        from huggingface_hub import constants as hf_constants

        return bool(getattr(hf_constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False))
    except Exception:
        return False


def _hf_auth_headers_for_url(url: str) -> dict[str, str]:
    """Attach the user's local Hugging Face token only to explicit HTTPS Hugging Face Hub origins."""
    if not _is_trusted_hf_origin(url):
        return {}
    if _implicit_hf_token_disabled():
        return {}
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        try:
            from huggingface_hub import get_token

            token = (get_token() or "").strip()
        except Exception:
            token = ""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            total += len(chunk)
            h.update(chunk)
    return h.hexdigest(), total


def _parse_content_range(header_val: str | None) -> tuple[int, int, int | None] | None:
    if not header_val:
        return None
    m = _CONTENT_RANGE_RE.match(header_val.strip())
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    total_raw = m.group(3)
    total = int(total_raw) if total_raw != "*" else None
    if end < start:
        return None
    if total is not None and end >= total:
        return None
    return start, end, total


def _parse_unsatisfied_range_total(header_val: str | None) -> int | None:
    if not header_val:
        return None
    m = _CONTENT_RANGE_UNSAT_RE.match(header_val.strip())
    if not m:
        return None
    return int(m.group(1))


def _parse_retry_after(header_val: str | None, *, now_epoch: float | None = None) -> float | None:
    """Parse RFC 9110 Retry-After (delay-seconds or HTTP-date). Returns seconds >= 0 or None."""
    if not header_val:
        return None
    val = header_val.strip()
    if not val:
        return None
    try:
        secs = float(val)
        if math.isfinite(secs) and secs >= 0:
            return secs
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now_s = time.time() if now_epoch is None else now_epoch
        delta = (dt - datetime.fromtimestamp(now_s, tz=timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


def _sleep_with_cancel(
    seconds: float,
    cancel_event: threading.Event | None,
    sleep_fn: Callable[[float], None],
) -> None:
    if seconds <= 0:
        return
    if cancel_event is not None:
        if cancel_event.wait(seconds):
            raise DownloadCancelledError("Download cancelled.")
    else:
        sleep_fn(seconds)


def _create_exclusive_staging_file(parent: Path, dest_name: str) -> Path:
    """Create a unique regular staging file under `parent` without following symlinks."""
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(16):
        cand = parent / f".{dest_name}.{secrets.token_hex(8)}.part"
        try:
            fd = os.open(str(cand), flags, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return cand
    raise RuntimeError(f"Could not allocate staging file for {dest_name}.")


def _open_staging_nofollow(path: Path, *, append: bool) -> Any:
    """Open an owned staging regular file without following symlinks."""
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_APPEND if append else os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise RuntimeError(f"Refusing to write to non-regular staging path: {path.name}")
        return os.fdopen(fd, "ab" if append else "wb")
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def download_verified_file(
    url: str,
    dest: Path,
    expected_sha256: str,
    *,
    client: httpx.Client | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    max_bytes: int | None = None,
    progress_cb: Callable[[int, int | None], None] | None = None,
    retry_cb: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Path, int]:
    """
    Stream an HTTPS asset to an exclusively owned staging file with retries, exponential backoff,
    strict HTTP Range resume, cooperative cancellation, and SHA-256 verification before atomic commit.
    Returns `(dest_path, size_bytes)`.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelledError("Download cancelled.")

    if dest.is_symlink():
        raise RuntimeError(f"Refusing to download into symlinked path: {dest.name}")

    if dest.is_file():
        existing_sha, existing_size = _sha256_file(dest)
        if existing_sha == expected_sha256:
            if max_bytes is not None and existing_size > max_bytes:
                raise RuntimeError(f"Download exceeded safety size limit on {dest.name}.")
            if progress_cb is not None:
                progress_cb(existing_size, existing_size)
            return dest, existing_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=20.0),
    )
    tmp_path: Path | None = None
    attempts = max(1, int(max_attempts))
    base_headers = {"Accept-Encoding": "identity", **_hf_auth_headers_for_url(url)}

    def _cleanup_tmp() -> None:
        if tmp_path is not None and (tmp_path.exists() or tmp_path.is_symlink()):
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _reset_tmp() -> None:
        assert tmp_path is not None
        with _open_staging_nofollow(tmp_path, append=False):
            pass

    def _commit_tmp(size_bytes: int) -> tuple[Path, int]:
        assert tmp_path is not None
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelledError("Download cancelled.")
        if dest.is_symlink():
            raise RuntimeError(f"Refusing to overwrite symlinked path: {dest.name}")
        if max_bytes is not None and size_bytes > max_bytes:
            raise RuntimeError(f"Download exceeded safety size limit on {dest.name}.")
        os.replace(tmp_path, dest)
        if progress_cb is not None:
            progress_cb(size_bytes, size_bytes)
        return dest, size_bytes

    try:
        tmp_path = _create_exclusive_staging_file(dest.parent, dest.name)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelledError("Download cancelled.")

            resume_offset = 0
            try:
                st = tmp_path.lstat()
                if stat.S_ISREG(st.st_mode):
                    resume_offset = st.st_size
                else:
                    raise RuntimeError(f"Staging file changed to non-regular file for {dest.name}.")
            except OSError:
                resume_offset = 0

            if max_bytes is not None and resume_offset > max_bytes:
                _reset_tmp()
                resume_offset = 0

            req_headers = dict(base_headers)
            if resume_offset > 0:
                req_headers["Range"] = f"bytes={resume_offset}-"

            retry_delay: float | None = None
            retry_reason = ""
            try:
                with http.stream("GET", url, headers=req_headers) as resp:
                    status = resp.status_code

                    if status == 416 and resume_offset > 0:
                        advertised_total = _parse_unsatisfied_range_total(resp.headers.get("content-range"))
                        if advertised_total is not None and resume_offset == advertised_total:
                            digest, verified_size = _sha256_file(tmp_path)
                            if digest == expected_sha256:
                                return _commit_tmp(verified_size)
                        _reset_tmp()
                        if attempt < attempts:
                            retry_reason = "Restarting partial download"
                            retry_delay = 0.05
                        else:
                            resp.raise_for_status()
                    elif status in RETRYABLE_STATUS_CODES:
                        retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                        if retry_after is not None and retry_after > MAX_RETRY_AFTER_S:
                            raise RuntimeError(
                                f"Server asked to wait {int(round(retry_after))}s before retrying {dest.name} (HTTP {status}) — please retry shortly."
                            )
                        if attempt < attempts:
                            reason_phrase = (resp.reason_phrase or "").strip()
                            retry_reason = f"CDN hiccup (HTTP {status}{f' {reason_phrase}' if reason_phrase else ''})"
                            if retry_after is not None:
                                max_jitter = max(0.0, min(0.5, MAX_RETRY_AFTER_S - retry_after))
                                jitter = random.uniform(0.0, max_jitter) if (backoff_base_s > 0 and max_jitter > 0) else 0.0
                                retry_delay = retry_after + jitter
                            else:
                                jitter = (
                                    random.uniform(0.0, max(0.05, backoff_base_s * 0.25))
                                    if backoff_base_s > 0
                                    else 0.0
                                )
                                retry_delay = backoff_base_s * (2 ** (attempt - 1)) + jitter
                        else:
                            resp.raise_for_status()
                    else:
                        resp.raise_for_status()

                        content_encoding = (resp.headers.get("content-encoding") or "").strip().lower()
                        is_encoded = bool(content_encoding and content_encoding != "identity")
                        is_resumed = False
                        expected_resp_len: int | None = None
                        if status == 206 and resume_offset > 0:
                            parsed_range = _parse_content_range(resp.headers.get("content-range"))
                            range_invalid = (
                                is_encoded
                                or parsed_range is None
                                or parsed_range[0] != resume_offset
                                or (parsed_range[2] is not None and parsed_range[1] + 1 != parsed_range[2])
                            )
                            if range_invalid:
                                _reset_tmp()
                                if attempt < attempts:
                                    retry_reason = "Invalid resume range from CDN — restarting from 0"
                                    retry_delay = 0.05
                                else:
                                    raise RuntimeError(f"Server returned invalid byte range for {dest.name}.")
                            else:
                                assert parsed_range is not None
                                r_start, r_end, r_total = parsed_range
                                expected_resp_len = (r_end - r_start) + 1
                                total_size = r_total if r_total is not None else (r_end + 1)
                                cl = resp.headers.get("content-length")
                                if cl and cl.isdigit() and int(cl) != expected_resp_len:
                                    _reset_tmp()
                                    if attempt < attempts:
                                        retry_reason = "Mismatched partial length — restarting from 0"
                                        retry_delay = 0.05
                                    else:
                                        raise RuntimeError(f"Server returned inconsistent partial length for {dest.name}.")
                                else:
                                    is_resumed = True
                                    bytes_done = resume_offset
                        elif status == 206 and resume_offset == 0:
                            # Unsolicited 206 without Range: accept only if complete from byte 0
                            parsed_range = _parse_content_range(resp.headers.get("content-range"))
                            unsolicited_invalid = (
                                is_encoded
                                or parsed_range is None
                                or parsed_range[0] != 0
                                or (parsed_range[2] is not None and parsed_range[1] + 1 != parsed_range[2])
                            )
                            if unsolicited_invalid:
                                _reset_tmp()
                                if attempt < attempts:
                                    retry_reason = "Unexpected partial response — retrying"
                                    retry_delay = 0.05
                                else:
                                    raise RuntimeError(f"Server returned unexpected partial response for {dest.name}.")
                            else:
                                assert parsed_range is not None
                                expected_resp_len = parsed_range[1] + 1
                                total_size = parsed_range[2] if parsed_range[2] is not None else expected_resp_len
                                cl = resp.headers.get("content-length")
                                if cl and cl.isdigit() and int(cl) != expected_resp_len:
                                    _reset_tmp()
                                    if attempt < attempts:
                                        retry_reason = "Mismatched partial length — retrying"
                                        retry_delay = 0.05
                                    else:
                                        raise RuntimeError(f"Server returned inconsistent partial length for {dest.name}.")
                                bytes_done = 0
                        else:
                            cl = resp.headers.get("content-length")
                            total_size = int(cl) if (cl and cl.isdigit()) else None
                            bytes_done = 0

                        if retry_delay is None:
                            if max_bytes is not None and total_size is not None and total_size > max_bytes:
                                raise RuntimeError(f"Download exceeded safety size limit on {dest.name}.")
                            if max_bytes is not None and bytes_done > max_bytes:
                                raise RuntimeError(f"Download exceeded safety size limit on {dest.name}.")

                            hasher = None if is_resumed else hashlib.sha256()
                            if progress_cb is not None:
                                progress_cb(bytes_done, total_size)

                            received_this_resp = 0
                            with _open_staging_nofollow(tmp_path, append=is_resumed) as out:
                                for chunk in resp.iter_bytes():
                                    if cancel_event is not None and cancel_event.is_set():
                                        raise DownloadCancelledError("Download cancelled.")
                                    if not chunk:
                                        continue
                                    received_this_resp += len(chunk)
                                    bytes_done += len(chunk)
                                    if max_bytes is not None and bytes_done > max_bytes:
                                        raise RuntimeError(f"Download exceeded safety size limit on {dest.name}.")
                                    out.write(chunk)
                                    if hasher is not None:
                                        hasher.update(chunk)
                                    if progress_cb is not None:
                                        progress_cb(bytes_done, total_size)

                            if expected_resp_len is not None and received_this_resp != expected_resp_len:
                                _reset_tmp()
                                if attempt < attempts:
                                    retry_reason = "Truncated partial response — restarting from 0"
                                    retry_delay = 0.05
                                else:
                                    raise RuntimeError(f"Truncated response while resuming {dest.name}.")
                            elif bytes_done <= 0:
                                raise RuntimeError(f"Downloaded file {dest.name} was empty.")
                            else:
                                if hasher is not None:
                                    digest = hasher.hexdigest()
                                    verified_size = bytes_done
                                else:
                                    digest, verified_size = _sha256_file(tmp_path)

                                if digest != expected_sha256:
                                    _reset_tmp()
                                    if is_resumed and attempt < attempts:
                                        retry_reason = "Resumed partial checksum mismatch — restarting from 0"
                                        retry_delay = 0.05
                                    else:
                                        raise RuntimeError(
                                            f"Checksum mismatch for {dest.name}: expected {expected_sha256[:12]}…, got {digest[:12]}…"
                                        )
                                else:
                                    return _commit_tmp(verified_size)

            except DownloadCancelledError:
                raise
            except (httpx.TransportError, httpx.TimeoutException, httpx.StreamError) as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise RuntimeError(format_download_error(exc, label=dest.name)) from exc
                retry_reason = "Connection interrupted"
                jitter = random.uniform(0.0, max(0.05, backoff_base_s * 0.25)) if backoff_base_s > 0 else 0.0
                retry_delay = backoff_base_s * (2 ** (attempt - 1)) + jitter
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(format_download_error(exc, label=dest.name)) from exc

            if retry_delay is not None:
                note = f"{retry_reason} — retrying ({attempt + 1}/{attempts})…"
                logger.info("Download retry for %s: %s (sleep %.2fs)", dest.name, note, retry_delay)
                if retry_cb is not None:
                    retry_cb(attempt + 1, attempts, note)
                _sleep_with_cancel(retry_delay, cancel_event, sleep_fn)

        if last_exc is not None:
            raise RuntimeError(format_download_error(last_exc, label=dest.name)) from last_exc
        raise RuntimeError(f"Failed to download {dest.name} after {attempts} attempts.")
    except BaseException:
        _cleanup_tmp()
        raise
    finally:
        if own_client:
            http.close()


class MultiFileProgressTracker:
    """Aggregates byte progress, smoothed transfer rate, and ETA across multi-file pack installs."""

    def __init__(
        self,
        *,
        file_count: int,
        approx_total_bytes: int | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        throttle_s: float = 0.2,
    ) -> None:
        self.file_count = max(1, int(file_count))
        self.approx_total_bytes = approx_total_bytes if (approx_total_bytes and approx_total_bytes > 0) else None
        self._on_update = on_update
        self._throttle_s = throttle_s
        self._lock = threading.Lock()
        self._completed_bytes = 0
        self._completed_files = 0
        self._current_file = ""
        self._current_done = 0
        self._current_total: int | None = None
        self._phase = "downloading"
        self._attempt = 1
        self._max_attempts = DEFAULT_MAX_ATTEMPTS
        self._retry_note: str | None = None
        self._cancellable = True
        self._started_at = time.monotonic()
        self._last_emit = 0.0
        self._speed_bps: float | None = None
        self._last_sample_t = self._started_at
        self._last_sample_bytes = 0

    def _build_snapshot_locked(self, now: float) -> dict[str, Any]:
        bytes_done = self._completed_bytes + self._current_done
        if self._completed_files + 1 >= self.file_count and self._current_total:
            bytes_total: int | None = self._completed_bytes + self._current_total
        else:
            known_floor = self._completed_bytes + (self._current_total or self._current_done)
            if self.approx_total_bytes:
                bytes_total = max(self.approx_total_bytes, known_floor)
            elif self.file_count == 1 and self._current_total:
                bytes_total = self._current_total
            else:
                bytes_total = None

        percent: float | None = None
        if bytes_total and bytes_total > 0:
            percent = min(99.5 if self._phase == "downloading" else 100.0, (bytes_done / bytes_total) * 100.0)

        dt = now - self._last_sample_t
        if dt >= 0.4 and bytes_done >= self._last_sample_bytes:
            inst_speed = (bytes_done - self._last_sample_bytes) / dt
            if self._speed_bps is None:
                self._speed_bps = inst_speed
            else:
                self._speed_bps = 0.65 * self._speed_bps + 0.35 * inst_speed
            self._last_sample_t = now
            self._last_sample_bytes = bytes_done

        eta: float | None = None
        if bytes_total and self._speed_bps and self._speed_bps > 1024 and bytes_done < bytes_total:
            eta = (bytes_total - bytes_done) / self._speed_bps

        prog = DownloadProgress(
            phase=self._phase,
            bytes_done=bytes_done,
            bytes_total=bytes_total,
            percent=percent,
            speed_bps=self._speed_bps,
            eta_seconds=eta,
            file_index=min(self.file_count, self._completed_files + 1),
            file_count=self.file_count,
            current_file=self._current_file,
            attempt=self._attempt,
            max_attempts=self._max_attempts,
            retry_note=self._retry_note,
            cancellable=self._cancellable,
        )
        return prog.to_dict()

    def _emit(self, *, force: bool = False) -> None:
        if self._on_update is None:
            return
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_emit) < self._throttle_s:
                return
            self._last_emit = now
            snap = self._build_snapshot_locked(now)
        try:
            self._on_update(snap)
        except Exception:
            pass

    def start_file(self, filename: str) -> None:
        with self._lock:
            self._current_file = filename
            self._current_done = 0
            self._current_total = None
            self._phase = "downloading"
            self._attempt = 1
            self._retry_note = None
            self._cancellable = True
        self._emit(force=True)

    def on_file_progress(self, done: int, total: int | None) -> None:
        with self._lock:
            self._current_done = max(0, int(done))
            if total is not None and total > 0:
                self._current_total = int(total)
            if self._phase == "retrying":
                self._phase = "downloading"
                self._retry_note = None
        self._emit(force=False)

    def on_file_retry(self, attempt: int, max_attempts: int, note: str) -> None:
        with self._lock:
            self._phase = "retrying"
            self._attempt = attempt
            self._max_attempts = max_attempts
            self._retry_note = note
        self._emit(force=True)

    def finish_file(self, size_bytes: int) -> None:
        with self._lock:
            self._completed_bytes += max(0, int(size_bytes))
            self._completed_files += 1
            self._current_done = 0
            self._current_total = None
            self._retry_note = None
        self._emit(force=True)

    def set_phase(self, phase: str, *, note: str | None = None, cancellable: bool = False) -> None:
        with self._lock:
            self._phase = phase
            self._retry_note = note
            self._cancellable = cancellable
        self._emit(force=True)


def make_hf_progress_tqdm_class(
    on_update: Callable[[dict[str, Any]], None] | None,
    *,
    approx_total_bytes: int | None = None,
    cancel_event: threading.Event | None = None,
    throttle_s: float = 0.25,
) -> type:
    """
    Create a subclass of `huggingface_hub.utils.tqdm` that preserves the full tqdm API
    (`set_description_str`, `set_postfix_str`, `format_dict`, etc.) while aggregating byte
    progress into Mockery's unified DownloadProgress format and supporting cooperative cancellation.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm_base

    lock = threading.Lock()
    state: dict[str, Any] = {
        "bars": {},  # bar_id -> [n, total]
        "next_id": 0,
        "last_emit": 0.0,
        "last_sample_t": time.monotonic(),
        "last_sample_bytes": 0,
        "speed_bps": None,
        "current_file": "",
    }

    def _emit(force: bool = False) -> None:
        if on_update is None:
            return
        now = time.monotonic()
        with lock:
            if not force and (now - state["last_emit"]) < throttle_s:
                return
            state["last_emit"] = now
            bars = state["bars"]
            bytes_done = sum(v[0] for v in bars.values())
            known_total = sum(v[1] for v in bars.values() if v[1] and v[1] > 0)
            bytes_total = (
                max(approx_total_bytes or 0, known_total)
                if (approx_total_bytes or known_total)
                else None
            )
            percent = (
                min(99.5, (bytes_done / bytes_total) * 100.0)
                if (bytes_total and bytes_total > 0 and bytes_done > 0)
                else None
            )
            dt = now - state["last_sample_t"]
            if dt >= 0.4 and bytes_done >= state["last_sample_bytes"]:
                inst = (bytes_done - state["last_sample_bytes"]) / dt
                prev = state["speed_bps"]
                state["speed_bps"] = inst if prev is None else (0.65 * prev + 0.35 * inst)
                state["last_sample_t"] = now
                state["last_sample_bytes"] = bytes_done
            speed = state["speed_bps"]
            eta = (
                (bytes_total - bytes_done) / speed
                if (bytes_total and speed and speed > 1024 and bytes_done < bytes_total)
                else None
            )
            snap = DownloadProgress(
                phase="downloading",
                bytes_done=bytes_done,
                bytes_total=bytes_total,
                percent=percent,
                speed_bps=speed,
                eta_seconds=eta,
                current_file=state["current_file"],
            ).to_dict()
        try:
            on_update(snap)
        except Exception:
            pass

    class _HFProgressTqdm(hf_tqdm_base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._tracked_total = kwargs.get("total")
            self._tracked_n = int(kwargs.get("initial") or 0)
            self._tracked_unit = str(kwargs.get("unit") or "")
            self._tracked_desc = str(kwargs.get("desc") or "")
            bar_name = str(kwargs.get("name") or "")
            self._bar_id: int | None = None
            # Keep terminal output quiet while preserving tqdm's full API & attributes
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            # Hub 2.4+ opens both a network transfer bar and a disk reconstruction bar for the
            # same snapshot; skip the reconstruction bar so bytes and throughput are never doubled.
            is_reconstruct_bar = (
                bar_name == "huggingface_hub.snapshot_download"
                or self._tracked_desc.lower().startswith("reconstructing")
            )
            is_byte_bar = (not is_reconstruct_bar) and (
                (self._tracked_unit == "B")
                or (isinstance(self._tracked_total, (int, float)) and self._tracked_total > 10_000)
            )
            if is_byte_bar:
                with lock:
                    state["next_id"] += 1
                    self._bar_id = state["next_id"]
                    state["bars"][self._bar_id] = [self._tracked_n, int(self._tracked_total or 0)]
                    if self._tracked_desc:
                        state["current_file"] = self._tracked_desc
                _emit(force=True)

        def __iter__(self):
            if self.iterable is None:
                return
            for item in self.iterable:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelledError("Download cancelled.")
                yield item
                self.update(1)

        def update(self, n: int | float | None = 1) -> bool | None:
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelledError("Download cancelled.")
            delta = int(n or 0)
            self._tracked_n = max(0, self._tracked_n + delta)
            self.n = self._tracked_n
            if self._bar_id is not None:
                with lock:
                    entry = state["bars"].get(self._bar_id)
                    if entry is not None:
                        entry[0] = self._tracked_n
                        cur_total = int(self.total or self._tracked_total or 0)
                        if cur_total > entry[1]:
                            entry[1] = cur_total
                _emit(force=False)
            return super().update(n)

        def reset(self, total: int | float | None = None) -> None:
            self._tracked_n = 0
            if total is not None:
                self._tracked_total = total
            if self._bar_id is not None:
                with lock:
                    entry = state["bars"].get(self._bar_id)
                    if entry is not None:
                        entry[0] = 0
                        if total is not None and int(total) > 0:
                            entry[1] = int(total)
            super().reset(total=total)

        def set_description(self, desc: str | None = None, refresh: bool | None = True) -> None:
            if desc:
                self._tracked_desc = str(desc)
                if self._bar_id is not None:
                    with lock:
                        state["current_file"] = self._tracked_desc
            super().set_description(desc=desc, refresh=refresh)

        def set_description_str(self, desc: str | None = None, refresh: bool | None = True) -> None:
            if desc:
                self._tracked_desc = str(desc)
                if self._bar_id is not None:
                    with lock:
                        state["current_file"] = self._tracked_desc
            super().set_description_str(desc=desc, refresh=refresh)

        def close(self) -> None:
            if self._bar_id is not None:
                _emit(force=True)
            super().close()

    return _HFProgressTqdm
