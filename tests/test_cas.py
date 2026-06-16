"""Tests for the content-addressed filesystem store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rfsn_agent.cas import ContentAddressedStore


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
