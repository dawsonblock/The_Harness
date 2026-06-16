"""Content-addressed filesystem store for large binary/text objects."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol


class HashFunction(Protocol):
    """Protocol for a hash algorithm callable."""

    def __call__(self, data: bytes) -> str:
        ...


def _sha256_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_path(base_dir: Path, content_hash: str) -> Path:
    """Spread objects across prefix directories (ab/abcdef...)."""
    if len(content_hash) < 4:
        raise ValueError("content_hash must be at least 4 characters")
    prefix = content_hash[:2]
    return base_dir / prefix / content_hash


class ContentAddressedStore:
    """Atomic, content-addressed filesystem store.

    Writes are performed to a temporary file in the target directory and then
    atomically renamed. Reads are by content hash. Objects are immutable and
    never overwritten.
    """

    def __init__(
        self,
        base_dir: str | Path,
        hash_fn: HashFunction = _sha256_hash,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.hash_fn = hash_fn
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes | str) -> str:
        """Store ``data`` and return its content hash."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        content_hash = self.hash_fn(data)
        target = _content_path(self.base_dir, content_hash)
        if target.exists():
            # Object already stored; idempotent no-op.
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
        except Exception:
            # Clean up the temporary file on failure; the target must not exist
            # or must be a valid, fully-written object.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return content_hash

    def get(self, content_hash: str) -> bytes:
        """Return the object bytes for ``content_hash`` if it exists."""
        target = _content_path(self.base_dir, content_hash)
        if not target.exists():
            raise KeyError(f"CAS object not found: {content_hash}")
        return target.read_bytes()

    def get_text(self, content_hash: str) -> str:
        """Return the object decoded as UTF-8 text."""
        return self.get(content_hash).decode("utf-8")

    def exists(self, content_hash: str) -> bool:
        """Return True if the object is already stored."""
        return _content_path(self.base_dir, content_hash).exists()

    def delete(self, content_hash: str) -> None:
        """Remove an object. Use sparingly; CAS objects are normally immutable."""
        target = _content_path(self.base_dir, content_hash)
        if target.exists():
            target.unlink()
