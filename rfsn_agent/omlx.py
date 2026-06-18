"""OMLX inference adapter protocols.

This module bridges semantic harness context packets to an external OMLX runtime.
It intentionally does not import ``rfsn_kv``; OMLX implementations may connect a
``ContextPacket.cache_branch_id`` to a KV provider through a protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rfsn_agent.context import ContextPacket


@dataclass(frozen=True, slots=True)
class OMLXInferenceRequest:
    """Request sent to an OMLX inference runtime."""

    trajectory_id: str
    epoch_id: str
    sequence: int
    rendered_context: str
    cache_branch_id: str
    max_new_tokens: int
    temperature: float = 0.0
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class OMLXInferenceResponse:
    """Response returned by an OMLX inference runtime."""

    text: str
    finish_reason: str
    token_count: int
    latency_ms: float
    cache_branch_id: str
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)


class OMLXInferencePolicy(Protocol):
    """Policy controlling OMLX inference parameters."""

    def inference_parameters(self, *, max_new_tokens: int) -> dict[str, Any]:
        """Return runtime-specific inference parameters."""
        ...


class OMLXKVCacheProvider(Protocol):
    """Protocol for an external KV cache provider used by OMLX."""

    def ensure_prefix(self, *, trajectory_id: str, epoch_id: str, cache_branch_id: str) -> None:
        """Ensure the KV prefix for a context epoch is resident."""
        ...


class OMLXInferenceAdapter:
    """Adapter from harness ``ContextPacket`` values to an OMLX runtime."""

    def __init__(
        self,
        *,
        policy: OMLXInferencePolicy,
        cache_provider: OMLXKVCacheProvider | None = None,
    ) -> None:
        self.policy = policy
        self.cache_provider = cache_provider

    def build_request(
        self,
        *,
        trajectory_id: str,
        epoch_id: str,
        sequence: int,
        rendered_context: str,
        cache_branch_id: str,
        max_new_tokens: int,
    ) -> OMLXInferenceRequest:
        """Build an OMLX request from a compiled context packet."""
        params = self.policy.inference_parameters(max_new_tokens=max_new_tokens)
        params.pop("max_new_tokens", None)
        return OMLXInferenceRequest(
            trajectory_id=trajectory_id,
            epoch_id=epoch_id,
            sequence=sequence,
            rendered_context=rendered_context,
            cache_branch_id=cache_branch_id,
            max_new_tokens=max_new_tokens,
            **params,
        )

    def build_request_from_packet(
        self,
        packet: ContextPacket,
        *,
        rendered_context: str,
        max_new_tokens: int,
    ) -> OMLXInferenceRequest:
        """Build an OMLX request from a compiled ``ContextPacket``."""
        return self.build_request(
            trajectory_id=packet.trajectory_id,
            epoch_id=packet.epoch_id,
            sequence=packet.state_sequence,
            rendered_context=rendered_context,
            cache_branch_id=packet.packet_hash,
            max_new_tokens=max_new_tokens,
        )

    def infer(
        self,
        *,
        trajectory_id: str,
        epoch_id: str,
        sequence: int,
        rendered_context: str,
        cache_branch_id: str,
        max_new_tokens: int,
    ) -> OMLXInferenceResponse:
        """Run OMLX inference for a rendered context packet.

        The default implementation is deterministic and suitable for tests.
        Production integrations should override this method.
        """
        request = self.build_request(
            trajectory_id=trajectory_id,
            epoch_id=epoch_id,
            sequence=sequence,
            rendered_context=rendered_context,
            cache_branch_id=cache_branch_id,
            max_new_tokens=max_new_tokens,
        )
        if self.cache_provider is not None:
            self.cache_provider.ensure_prefix(
                trajectory_id=request.trajectory_id,
                epoch_id=request.epoch_id,
                cache_branch_id=request.cache_branch_id,
            )
        text = request.rendered_context[-min(max_new_tokens, len(request.rendered_context)) :]
        return OMLXInferenceResponse(
            text=text,
            finish_reason="length" if max_new_tokens > 0 else "stop",
            token_count=max_new_tokens,
            latency_ms=0.0,
            cache_branch_id=request.cache_branch_id,
        )


__all__: list[str] = [
    "OMLXInferenceAdapter",
    "OMLXInferencePolicy",
    "OMLXInferenceRequest",
    "OMLXInferenceResponse",
    "OMLXKVCacheProvider",
]
