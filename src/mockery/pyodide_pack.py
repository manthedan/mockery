from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import stat
import threading
from pathlib import Path
from typing import BinaryIO

import httpx

logger = logging.getLogger("mockery.pyodide")

PYODIDE_VERSION = "0.27.5"
PYODIDE_BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full"

PYODIDE_FILE_SHA256: dict[str, str] = {
    "pyodide.js": "7fdbe66e53f68f6a4e93c295a667371759be093d2bd402bb44545514584039b6",
    "pyodide.asm.js": "3a889f073e628c2196c705b42fa0e955ba2e25c034b1e3dd589c35be675bc01b",
    "pyodide.asm.wasm": "f7fefe563134714a17abd65516d94960e8dbd96fe6778a7a842947fc9686b3a1",
    "python_stdlib.zip": "6030964967e447c887abc46c5f0967c55688644d759496de82a3ef09f49f5cba",
    "pyodide-lock.json": "be1807745da93daa09d360b109c17a0e526e74d664d1f1b9870aafcce98ce426",
}

PYODIDE_CORE_FILES: tuple[str, ...] = tuple(PYODIDE_FILE_SHA256.keys())
MANIFEST_FILENAME = "manifest.json"

MAX_SINGLE_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_PACK_BYTES = 100 * 1024 * 1024

MIME_BY_EXT: dict[str, str] = {
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".wasm": "application/wasm",
    ".zip": "application/zip",
    ".json": "application/json; charset=utf-8",
    ".data": "application/octet-stream",
}


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
    return h.hexdigest(), total


class PyodidePackManager:
    """
    Manages the opt-in offline Pyodide WASM runtime under `<data_root>/models/pyodide`.
    Never fetches from any external CDN unless the user explicitly clicks Install.
    Verifies every core asset against pinned SHA-256 digests and serves opened file
    descriptors relative to a verified non-symlink directory handle.
    """

    def __init__(
        self,
        pack_dir: Path,
        base_url: str = PYODIDE_BASE_URL,
        *,
        expected_hashes: dict[str, str] | None = None,
    ) -> None:
        self.pack_dir = pack_dir
        self.base_url = base_url.rstrip("/")
        self.expected_hashes: dict[str, str] = dict(expected_hashes or PYODIDE_FILE_SHA256)
        self._lock = threading.RLock()

    def _open_trusted_pack_dir_fd(self) -> int | None:
        """Open `self.pack_dir` without following symlinks on the root directory."""
        if self.pack_dir.is_symlink() or not self.pack_dir.exists():
            return None
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            dir_fd = os.open(str(self.pack_dir), flags)
        except OSError:
            return None
        try:
            st = os.fstat(dir_fd)
            if not stat.S_ISDIR(st.st_mode):
                os.close(dir_fd)
                return None
            return dir_fd
        except OSError:
            os.close(dir_fd)
            return None

    def _read_manifest_from_fd(self, dir_fd: int) -> dict | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(MANIFEST_FILENAME, flags, dir_fd=dir_fd)
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size <= 0 or st.st_size > 65536:
                os.close(fd)
                return None
            with os.fdopen(fd, "rb") as fh:
                raw = json.loads(fh.read().decode("utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict) or raw.get("version") != PYODIDE_VERSION:
            return None
        files = raw.get("files")
        if not isinstance(files, dict):
            return None
        return raw

    def _verify_dir_fd(self, dir_fd: int) -> dict | None:
        manifest = self._read_manifest_from_fd(dir_fd)
        if manifest is None:
            return None
        files = manifest["files"]
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        for name, expected_sha in self.expected_hashes.items():
            entry = files.get(name)
            if not isinstance(entry, dict):
                return None
            if entry.get("sha256") != expected_sha:
                return None
            expected_size = entry.get("size")
            if not isinstance(expected_size, int) or expected_size <= 0:
                return None
            try:
                fd = os.open(name, flags, dir_fd=dir_fd)
            except OSError:
                return None
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size != expected_size:
                    return None
            finally:
                os.close(fd)
        return manifest

    def is_installed(self) -> bool:
        with self._lock:
            dir_fd = self._open_trusted_pack_dir_fd()
            if dir_fd is None:
                return False
            try:
                return self._verify_dir_fd(dir_fd) is not None
            finally:
                os.close(dir_fd)

    def _validate_asset_name(self, rel_path: str) -> str:
        raw = str(rel_path or "")
        if not raw or "\x00" in raw or "/" in raw or "\\" in raw:
            raise ValueError("Invalid Pyodide asset path.")
        if raw not in self.expected_hashes:
            raise ValueError(f"Unsupported or unknown Pyodide asset '{raw}'.")
        return raw

    def open_asset(self, rel_path: str) -> tuple[BinaryIO, int, str, str] | None:
        """
        Validate `rel_path` against the exact verified core-file allowlist and return an
        already-opened binary file handle `(file_obj, size_bytes, media_type, etag)`.
        Raises ValueError on traversal/disallowed paths; returns None if pack is not installed.
        """
        name = self._validate_asset_name(rel_path)
        with self._lock:
            dir_fd = self._open_trusted_pack_dir_fd()
            if dir_fd is None:
                return None
            try:
                manifest = self._verify_dir_fd(dir_fd)
                if manifest is None:
                    return None
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                try:
                    file_fd = os.open(name, flags, dir_fd=dir_fd)
                except OSError:
                    return None
                try:
                    st = os.fstat(file_fd)
                    if not stat.S_ISREG(st.st_mode):
                        os.close(file_fd)
                        return None
                    sha = str(manifest["files"][name]["sha256"])
                    etag = f'"{PYODIDE_VERSION}-{sha[:16]}"'
                    media_type = MIME_BY_EXT.get(Path(name).suffix.lower(), "application/octet-stream")
                    return os.fdopen(file_fd, "rb"), int(st.st_size), media_type, etag
                except Exception:
                    os.close(file_fd)
                    raise
            finally:
                os.close(dir_fd)

    def write_verified_manifest(self, target_dir: Path | None = None) -> dict:
        """Compute and write `manifest.json` after verifying every core file matches `expected_hashes`."""
        dest_dir = target_dir or self.pack_dir
        files_meta: dict[str, dict[str, object]] = {}
        for name, expected_sha in self.expected_hashes.items():
            fp = dest_dir / name
            if fp.is_symlink() or not fp.is_file():
                raise RuntimeError(f"Missing required Pyodide core file: {name}")
            actual_sha, size = _sha256_file(fp)
            if size <= 0 or actual_sha != expected_sha:
                raise RuntimeError(f"Checksum mismatch for Pyodide core file {name}.")
            files_meta[name] = {"sha256": actual_sha, "size": size}
        manifest = {"version": PYODIDE_VERSION, "files": files_meta}
        (dest_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    def install(self) -> Path:
        """Download pinned Pyodide core files, verify SHA-256 digests, and publish atomically."""
        if self.pack_dir.is_symlink():
            raise RuntimeError("Refusing to install Pyodide into a symlinked directory.")
        self.pack_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.pack_dir.parent / f".pyodide_staging_{secrets.token_hex(6)}"
        backup = self.pack_dir.parent / f".pyodide_backup_{secrets.token_hex(6)}"
        staging.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        files_meta: dict[str, dict[str, object]] = {}
        try:
            timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                for filename, expected_sha in self.expected_hashes.items():
                    url = f"{self.base_url}/{filename}"
                    dest = staging / filename
                    file_bytes = 0
                    digest = hashlib.sha256()
                    with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            raise RuntimeError(
                                f"Failed to download {filename} (HTTP {resp.status_code})."
                            )
                        with dest.open("wb") as fh:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                if not chunk:
                                    continue
                                file_bytes += len(chunk)
                                total_bytes += len(chunk)
                                if file_bytes > MAX_SINGLE_FILE_BYTES or total_bytes > MAX_TOTAL_PACK_BYTES:
                                    raise RuntimeError(
                                        f"Pyodide download exceeded safety size limit on {filename}."
                                    )
                                digest.update(chunk)
                                fh.write(chunk)
                    actual_sha = digest.hexdigest()
                    if file_bytes <= 0:
                        raise RuntimeError(f"Downloaded Pyodide file {filename} was empty.")
                    if actual_sha != expected_sha:
                        raise RuntimeError(
                            f"Integrity check failed for {filename} "
                            f"(expected {expected_sha[:12]}…, got {actual_sha[:12]}…)."
                        )
                    files_meta[filename] = {"sha256": actual_sha, "size": file_bytes}

            manifest = {"version": PYODIDE_VERSION, "files": files_meta}
            (staging / MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            # Atomic directory swap serialized with open_asset/uninstall
            with self._lock:
                if self.pack_dir.is_symlink():
                    raise RuntimeError("Refusing to overwrite symlinked Pyodide pack directory.")
                had_existing = self.pack_dir.exists()
                if had_existing:
                    self.pack_dir.rename(backup)
                try:
                    staging.rename(self.pack_dir)
                except Exception:
                    if had_existing and backup.exists() and not self.pack_dir.exists():
                        backup.rename(self.pack_dir)
                    raise
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            logger.info("Installed verified offline Pyodide %s (%d bytes)", PYODIDE_VERSION, total_bytes)
            return self.pack_dir
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def uninstall(self) -> bool:
        with self._lock:
            if self.pack_dir.is_symlink():
                self.pack_dir.unlink()
                return True
            if not self.pack_dir.exists():
                return False
            shutil.rmtree(self.pack_dir)
            return True
