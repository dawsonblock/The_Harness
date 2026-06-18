"""Tests for OMLX inference adapter protocols."""

from __future__ import annotations

from rfsn_agent.omlx import (
    OMLXInferenceAdapter,
    OMLXInferenceRequest,
)
from rfsn_agent.types import ContentHash


class _Policy:
    def inference_parameters(self, *, max_new_tokens: int) -> dict[str, object]:
        return {"temperature": 0.2, "max_new_tokens": max_new_tokens}


class _CacheProvider:
    def __init__(self) -> None:
        self.calls: list[OMLXInferenceRequest] = []

    def ensure_prefix(
        self, *, trajectory_id: str, epoch_id: str, cache_branch_id: str
    ) -> None:
        self.calls.append(
            OMLXInferenceRequest(
                trajectory_id=trajectory_id,
                epoch_id=epoch_id,
                sequence=0,
                rendered_context="",
                cache_branch_id=cache_branch_id,
                max_new_tokens=0,
            )
        )


def test_omlx_adapter_builds_request_from_context_packet() -> None:
    adapter = OMLXInferenceAdapter(policy=_Policy())
    request = adapter.build_request(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=2,
        rendered_context="rendered text",
        cache_branch_id=ContentHash("branch-1"),
        max_new_tokens=5,
    )
    assert request.trajectory_id == "traj-1"
    assert request.epoch_id == "epoch-1"
    assert request.sequence == 2
    assert request.rendered_context == "rendered text"
    assert request.cache_branch_id == "branch-1"
    assert request.temperature == 0.2
    assert request.max_new_tokens == 5


def test_omlx_adapter_uses_cache_provider() -> None:
    cache_provider = _CacheProvider()
    adapter = OMLXInferenceAdapter(policy=_Policy(), cache_provider=cache_provider)
    response = adapter.infer(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=1,
        rendered_context="abc",
        cache_branch_id=ContentHash("branch-1"),
        max_new_tokens=2,
    )
    assert response.text == "bc"
    assert response.finish_reason == "length"
    assert response.token_count == 2
    assert len(cache_provider.calls) == 1
    assert cache_provider.calls[0].cache_branch_id == "branch-1"
