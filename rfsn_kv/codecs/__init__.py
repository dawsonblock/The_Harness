"""KV compression codec registry.

All available codecs are registered here. Use ``get_codec(codec_id)`` to
retrieve a codec instance by its identifier.
"""

from __future__ import annotations

from rfsn_kv.codecs.base import KVCodec
from rfsn_kv.codecs.identity import IdentityCodec
from rfsn_kv.codecs.mlx_tensor import MLXTensorCodec, MLXUnavailableError
from rfsn_kv.codecs.quantize import QuantizeCodec

# Global codec registry — maps codec_id → codec instance.
CODEC_REGISTRY: dict[str, KVCodec] = {
    "identity": IdentityCodec(),
    "quantize": QuantizeCodec(bit_width=8, group_size=64),
}

try:
    CODEC_REGISTRY["mlx_tensor"] = MLXTensorCodec()
except MLXUnavailableError:
    # MLX is optional on non-Metal hosts; callers can import MLXTensorCodec
    # directly and handle MLXUnavailableError when they call compress/decompress.
    pass


def get_codec(codec_id: str) -> KVCodec:
    """Return the codec instance for ``codec_id``.

    Raises:
        KeyError: If no codec is registered with that identifier.
    """
    try:
        return CODEC_REGISTRY[codec_id]
    except KeyError:
        available = ", ".join(sorted(CODEC_REGISTRY))
        raise KeyError(
            f"Unknown codec {codec_id!r}. Available: {available}"
        ) from None


__all__: list[str] = [
    "CODEC_REGISTRY",
    "get_codec",
    "IdentityCodec",
    "QuantizeCodec",
    "MLXTensorCodec",
    "MLXUnavailableError",
]
