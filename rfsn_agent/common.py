"""Deterministic hashing and serialization utilities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

from rfsn_agent.types import ContentHash

T = TypeVar("T")


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
    """Hash arbitrary textual content."""
    return ContentHash(sha256_hash(content))


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Serialize a dataclass instance into a dictionary.

    This is the inverse of :func:`dataclass_from_dict` for the immutable
    harness domain objects.
    """
    if not hasattr(obj, "__dataclass_fields__"):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")
    return cast(dict[str, Any], _serialize(obj))


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Deserialize a dictionary into a dataclass instance.

    Supports datetime, enums, tuples, frozensets, optionals, nested
    dataclasses, and NewType-wrapped primitives.
    """
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
                raise KeyError(f"Missing required field {field_name!r} for {cls.__name__}")
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
    # Treat dataclasses as dictionaries of their fields.
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

    # Optional / Union (typing.Union or PEP 604 X | Y types.UnionType)
    if origin is not None and (
        getattr(origin, "__name__", None) == "Union" or origin is types.UnionType
    ):
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _deserialize_value(raw, arg)
            except Exception:
                continue
        raise ValueError(f"Could not deserialize {raw!r} into {typ}")

    # NewType
    if hasattr(typ, "__supertype__"):
        return typ(_deserialize_value(raw, typ.__supertype__))

    # datetime
    if typ is datetime:
        return datetime.fromisoformat(raw)

    # Enum
    if isinstance(typ, type) and issubclass(typ, Enum):
        return typ(raw)

    # tuple (homogeneous or fixed-length)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_deserialize_value(item, args[0]) for item in raw)
        return tuple(
            _deserialize_value(item, arg) for item, arg in zip(raw, args)
        )

    # frozenset / set
    if origin in (frozenset, set):
        return frozenset(_deserialize_value(item, args[0]) for item in raw)

    # Nested dataclass
    if hasattr(typ, "__dataclass_fields__"):
        return dataclass_from_dict(typ, raw)

    return raw
