"""Deterministic hashing and serialization utilities for the KV layer.

This module is self-contained: it does NOT import from rfsn_agent so that
rfsn_kv remains a clean boundary crate with no upward dependencies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types as pytypes
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, TypeVar, cast, get_args, get_origin, get_type_hints

from rfsn_kv.types import ContentHash

T = TypeVar("T")


class CASStore(Protocol):
    """Protocol for content-addressed storage.

    Satisfied by ``rfsn_agent.cas.ContentAddressedStore`` via duck typing.
    """

    def put(self, data: bytes | str) -> str:
        """Store ``data`` and return its SHA-256 content hash."""
        ...

    def put_text(self, text: str) -> str:
        """Store text and return its SHA-256 content hash."""
        ...

    def get_text(self, content_hash: str) -> str:
        """Retrieve text by its content hash."""
        ...

    def get_bytes(self, content_hash: str) -> bytes:
        """Retrieve raw bytes by its content hash."""
        ...


def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def canonical_json(obj: Any) -> str:
    """Serialize an object to a deterministic, canonical JSON string."""
    return json.dumps(
        _serialize(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_hash(data: str | bytes) -> str:
    """Return the hex SHA-256 digest of a string or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_content(content: str) -> ContentHash:
    """Hash arbitrary textual content and return a ContentHash."""
    return ContentHash(sha256_hash(content))


def hash_bytes(data: bytes) -> ContentHash:
    """Hash raw bytes and return a ContentHash."""
    return ContentHash(sha256_hash(data))


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Serialize a dataclass instance into a dictionary."""
    if not hasattr(obj, "__dataclass_fields__"):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")
    return cast(dict[str, Any], _serialize(obj))


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Deserialize a dictionary into a dataclass instance."""
    cls_any: Any = cls
    if not hasattr(cls_any, "__dataclass_fields__"):
        raise TypeError(f"Expected a dataclass type, got {cls.__name__}")

    hints = get_type_hints(cls_any)
    kwargs: dict[str, Any] = {}
    for field_name, field_info in cls_any.__dataclass_fields__.items():
        if field_name in data:
            raw = data[field_name]
            field_type = hints[field_name]
            kwargs[field_name] = _deserialize_value(raw, field_type)
        else:
            if field_info.default is not dataclasses.MISSING:
                kwargs[field_name] = field_info.default
            elif field_info.default_factory is not dataclasses.MISSING:
                kwargs[field_name] = field_info.default_factory()
            else:
                raise KeyError(
                    f"Missing required field {field_name!r} for {cls.__name__}"
                )
    return cls(**kwargs)


def _serialize(value: Any) -> Any:
    """Recursively convert values into JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, frozenset | set):
        return sorted(_serialize(v) for v in value)
    if isinstance(value, tuple | list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _serialize(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return str(value)


def _deserialize_value(raw: Any, typ: Any) -> Any:
    """Convert a JSON primitive into the target field type."""
    if raw is None:
        return None

    origin = get_origin(typ)
    args = get_args(typ)

    # Union types (typing.Union or X | Y types.UnionType)
    if origin is not None and (
        getattr(origin, "__name__", None) == "Union" or origin is pytypes.UnionType
    ):
        # Try each variant in order
        for arg in args:
            if arg is type(None) and raw is None:
                return None
            if arg is type(None):
                continue
            try:
                return _deserialize_value(raw, arg)
            except (ValueError, TypeError, KeyError):
                continue
        raise ValueError(f"Cannot deserialize {raw!r} into {typ}")

    # NewType: origin is the underlying type
    if origin is None and callable(typ) and not isinstance(typ, type):
        # It's a NewType — get the supertype
        supertype = getattr(typ, "__supertype__", None)
        if supertype is not None:
            return typ(_deserialize_value(raw, supertype))

    if isinstance(raw, str) and typ is datetime:
        return datetime.fromisoformat(raw)

    if isinstance(raw, str) and isinstance(typ, type) and issubclass(typ, Enum):
        return typ(raw)

    if isinstance(raw, str) and typ is str:
        return raw

    if isinstance(raw, int | float) and typ is int:
        return int(raw)

    if isinstance(raw, int | float) and typ is float:
        return float(raw)

    if isinstance(raw, bool) and typ is bool:
        return raw

    if isinstance(raw, str) and typ is bytes:
        return raw.encode("utf-8")

    if isinstance(raw, bytes) and typ is bytes:
        return raw

    # Tuples
    if origin is tuple or (isinstance(typ, type) and issubclass(typ, tuple)):
        if isinstance(raw, list):
            if args:
                return tuple(_deserialize_value(item, args[0]) for item in raw)
            return tuple(raw)

    # Lists
    if origin is list or (isinstance(typ, type) and issubclass(typ, list)):
        if isinstance(raw, list):
            if args:
                return [_deserialize_value(item, args[0]) for item in raw]
            return list(raw)

    # Frozensets
    if origin is frozenset or (isinstance(typ, type) and issubclass(typ, frozenset)):
        if isinstance(raw, list):
            if args:
                return frozenset(_deserialize_value(item, args[0]) for item in raw)
            return frozenset(raw)

    # Optional (already handled via Union above, but direct None check)
    if raw is None:
        return None

    # Nested dataclasses
    if isinstance(typ, type) and hasattr(typ, "__dataclass_fields__"):
        if isinstance(raw, dict):
            return dataclass_from_dict(typ, raw)

    return raw
