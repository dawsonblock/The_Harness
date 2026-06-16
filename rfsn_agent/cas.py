"""Content-addressed filesystem store for large binary/text objects.

All objects are addressed by a fixed SHA-256 lowercase hex digest. The store
validates every read and write against the requested digest, and atomic rename
is used so readers never see a partially written object.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CASCorruptionError(ValueError):
    """Raised when bytes in the CAS do not match their content hash."""


def _sha256_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_path(base_dir: Path, content_hash: str) -> Path:
    """Return the storage path for ``content_hash``.

    Validates that ``content_hash`` is a 64-character lowercase SHA-256 digest,
    resolves the final path, and asserts containment under ``base_dir``.
    """
    if not isinstance(content_hash, str):
        raise ValueError(f"content_hash must be a string, got {type(content_hash).__name__}")
    if not _HASH_PATTERN.match(content_hash):
        raise ValueError(
            "content_hash must be a 64-character lowercase hex SHA-256 digest, "
            f"got {content_hash!r}"
        )
    prefix = content_hash[:2]
    target = (base_dir / prefix / content_hash).resolve()
    base_resolved = base_dir.resolve()
    if base_resolved not in target.parents and target != base_resolved:
        raise ValueError(
            f"CAS path escapes base directory: {target} not under {base_resolved}"
        )
    return target


class ContentAddressedStore:
    """Atomic, content-addressed filesystem store using fixed SHA-256.

    Writes are performed to a temporary file in the target directory and then
    atomically renamed. Reads verify that the stored bytes hash to the
    requested digest. Objects are immutable and never overwritten.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes | str) -> str:
        """Store ``data`` and return its SHA-256 content hash."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, bytes):
            raise TypeError(f"CAS data must be bytes or str, got {type(data).__name__}")
        content_hash = _sha256_hash(data)
        target = _content_path(self.base_dir, content_hash)

        if target.exists():
            existing = target.read_bytes()
            if _sha256_hash(existing) != content_hash:
                raise CASCorruptionError(
                    f"Existing CAS object is corrupt: {content_hash}"
                )
            return content_hash

        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=target.parent, prefix=".tmp-", suffix=".cas"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
            self._fsync_dir(target.parent)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return content_hash

    def get(self, content_hash: str) -> bytes:
        """Return the object bytes for ``content_hash`` if it exists and is valid."""
        target = _content_path(self.base_dir, content_hash)
        if not target.exists():
            raise KeyError(f"CAS object not found: {content_hash}")
        data = target.read_bytes()
        if _sha256_hash(data) != content_hash:
            raise CASCorruptionError(
                f"CAS object {content_hash} content does not match its digest"
            )
        return data

    def get_text(self, content_hash: str) -> str:
        """Return the object decoded as UTF-8 text."""
        return self.get(content_hash).decode("utf-8")

    def exists(self, content_hash: str) -> bool:
        """Return True if the object is already stored and valid."""
        try:
            target = _content_path(self.base_dir, content_hash)
        except ValueError:
            return False
        if not target.exists():
            return False
        try:
            data = target.read_bytes()
        except OSError:
            return False
        return _sha256_hash(data) == content_hash

    def delete(self, content_hash: str) -> None:
        """Remove an object. Use sparingly; CAS objects are normally immutable."""
        target = _content_path(self.base_dir, content_hash)
        if target.exists():
            target.unlink()

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """Flush the directory entry so the rename is durable."""
        dir_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
