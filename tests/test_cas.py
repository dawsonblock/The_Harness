"""Tests for the content-addressed filesystem store."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from rfsn_agent.cas import CASCorruptionError, ContentAddressedStore


def test_put_and_get_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        data = b"hello world"
        h = store.put(data)
        assert store.get(h) == data


def test_put_and_get_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        text = "unicode: αβγ"
        h = store.put(text)
        assert store.get_text(h) == text


def test_put_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        data = b"same"
        h1 = store.put(data)
        h2 = store.put(data)
        assert h1 == h2
        files = [p for p in Path(tmp).rglob("*") if p.is_file()]
        assert len(files) == 1


def test_exists_and_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        h = store.put(b"exists")
        assert store.exists(h) is True
        assert store.exists("0" * 64) is False


def test_get_missing_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        with pytest.raises(KeyError):
            store.get("0" * 64)


def test_content_hash_changes_with_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        h1 = store.put(b"a")
        h2 = store.put(b"b")
        assert h1 != h2


def test_rejects_malformed_content_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        with pytest.raises(ValueError):
            store.get("short")
        with pytest.raises(ValueError):
            store.get("g" * 64)
        with pytest.raises(ValueError):
            store.get("A" * 64)
        with pytest.raises(ValueError):
            store.get("0" * 63)


def test_detects_corrupt_object_on_get() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        data = b"hello world"
        h = store.put(data)
        prefix = h[:2]
        target = Path(tmp) / prefix / h
        target.write_bytes(b"corrupted")
        with pytest.raises(CASCorruptionError):
            store.get(h)
        assert store.exists(h) is False


def test_detects_corrupt_preexisting_object_on_put() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        data = b"hello world"
        h = hashlib.sha256(data).hexdigest()
        prefix = h[:2]
        target = Path(tmp) / prefix / h
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"corrupted")
        with pytest.raises(CASCorruptionError):
            store.put(data)


def test_path_escapes_base_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        # A digest that would decode to a relative path is still contained,
        # but absolute digests are rejected by the regex.
        with pytest.raises(ValueError):
            store.get("../" + "0" * 61)


def test_successful_put_leaves_no_temp_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        data = b"partial"
        h = hashlib.sha256(data).hexdigest()

        store.put(data)
        files = [p for p in Path(tmp).rglob("*") if p.is_file()]
        assert len(files) == 1
        assert files[0].name == h
        assert store.get(h) == data


def test_failed_put_cleans_up_its_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ContentAddressedStore(tmp)
        data = b"partial"

        def _failing_replace(src: str, dst: str) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("rfsn_agent.cas.os.replace", _failing_replace)

        with pytest.raises(OSError):
            store.put(data)

        # Neither target nor temp file should remain.
        files = [p for p in Path(tmp).rglob("*") if p.is_file()]
        assert len(files) == 0
