"""Tests for rfsn_kv.prefix_index."""

from __future__ import annotations

import pytest

from rfsn_kv.common import hash_content
from rfsn_kv.pages import PageRange
from rfsn_kv.prefix_index import (
    PrefixBranch,
    PrefixIndex,
    PrefixNode,
    _create_branch,
)
from rfsn_kv.types import BranchId, ContentHash, NodeId, PageId


def _node(
    node_id: str,
    start: int,
    end: int,
    page_ids: tuple[str, ...] = (),
    ref_count: int = 1,
    branch_id: str = "",
) -> PrefixNode:
    return PrefixNode.create(
        node_id=NodeId(node_id),
        page_range=PageRange(
            start_token=start,
            end_token=end,
            page_ids=tuple(PageId(p) for p in page_ids),
        ),
        ref_count=ref_count,
        branch_id=BranchId(branch_id),
    )


def _branch(
    branch_id: str,
    root_node_id: str = "n-0",
    node_ids: tuple[str, ...] = (),
    page_table_hash: str = "",
) -> PrefixBranch:
    return _create_branch(
        branch_id=BranchId(branch_id),
        root_node_id=NodeId(root_node_id),
        node_ids=tuple(NodeId(n) for n in node_ids),
        page_table_hash=(
            ContentHash(page_table_hash)
            if page_table_hash
            else hash_content("default")
        ),
    )


class TestPrefixNode:
    def test_create_and_immutable(self) -> None:
        node = _node("n-0", 0, 100, ("p-0",))
        assert node.node_id == "n-0"
        assert node.ref_count == 1
        assert node.node_hash != ""
        with pytest.raises(AttributeError):
            node.ref_count = 2  # type: ignore[misc]

    def test_with_incremented_ref(self) -> None:
        node = _node("n-0", 0, 100)
        incremented = node.with_incremented_ref()
        assert incremented.ref_count == 2
        assert node.ref_count == 1  # original unchanged

    def test_with_decremented_ref(self) -> None:
        node = _node("n-0", 0, 100, ref_count=3)
        decremented = node.with_decremented_ref()
        assert decremented.ref_count == 2

    def test_decrement_does_not_go_below_zero(self) -> None:
        node = _node("n-0", 0, 100, ref_count=1)
        decremented = node.with_decremented_ref()
        assert decremented.ref_count == 0

    def test_node_hash_deterministic(self) -> None:
        a = _node("n-0", 0, 100, ("p-0",))
        b = _node("n-0", 0, 100, ("p-0",))
        assert a.node_hash == b.node_hash

    def test_node_hash_changes_with_id(self) -> None:
        a = _node("n-0", 0, 100)
        b = _node("n-1", 0, 100)
        assert a.node_hash != b.node_hash

    def test_node_hash_excludes_ref_count(self) -> None:
        a = _node("n-0", 0, 100, ref_count=1)
        b = _node("n-0", 0, 100, ref_count=5)
        assert a.node_hash == b.node_hash


class TestPrefixBranch:
    def test_create(self) -> None:
        branch = _branch("branch-1", root_node_id="n-0")
        assert branch.branch_id == "branch-1"
        assert branch.root_node_id == "n-0"
        assert branch.branch_hash != ""

    def test_branch_hash_deterministic(self) -> None:
        a = _branch("b-1", node_ids=("n-0", "n-1"))
        b = _branch("b-1", node_ids=("n-0", "n-1"))
        assert a.branch_hash == b.branch_hash

    def test_branch_hash_changes_with_id(self) -> None:
        a = _branch("b-1", node_ids=("n-0",))
        b = _branch("b-2", node_ids=("n-0",))
        assert a.branch_hash != b.branch_hash

    def test_branch_hash_excludes_provenance(self) -> None:
        a = _branch("b-1")
        b = _create_branch(
            branch_id=BranchId("b-1"),
            root_node_id=NodeId("n-0"),
            node_ids=(),
            page_table_hash=hash_content("default"),
            actor="other",
        )
        assert a.branch_hash == b.branch_hash


class TestPrefixIndex:
    def test_empty_index(self) -> None:
        idx = PrefixIndex()
        assert len(idx) == 0
        assert idx.branches == ()
        assert idx.nodes == ()

    def test_register_branch(self) -> None:
        idx = PrefixIndex()
        n0 = _node("n-0", 0, 100, ("p-0",))
        idx2 = idx.register_branch(
            branch_id=BranchId("b-1"),
            page_table_hash=hash_content("test"),
            nodes=(n0,),
        )
        assert len(idx2) == 1
        assert len(idx2.nodes) == 1
        assert idx2.get_branch(BranchId("b-1")) is not None

    def test_preserves_immutability(self) -> None:
        idx = PrefixIndex()
        n0 = _node("n-0", 0, 100, ("p-0",))
        idx2 = idx.register_branch(
            branch_id=BranchId("b-1"),
            page_table_hash=hash_content("test"),
            nodes=(n0,),
        )
        assert len(idx) == 0
        assert len(idx2) == 1

    def test_get_branch_not_found(self) -> None:
        idx = PrefixIndex()
        assert idx.get_branch(BranchId("missing")) is None

    def test_get_node_not_found(self) -> None:
        idx = PrefixIndex()
        assert idx.get_node(NodeId("missing")) is None

    def test_get_node_found(self) -> None:
        n0 = _node("n-0", 0, 100)
        idx = PrefixIndex(nodes=(n0,))
        found = idx.get_node(NodeId("n-0"))
        assert found is not None
        assert found.node_id == "n-0"

    def test_get_common_prefix_same_branch(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",))
        idx = PrefixIndex()
        idx2 = idx.register_branch(BranchId("a"), hash_content("a"), (n0,))
        idx3 = idx2.register_branch(BranchId("b"), hash_content("b"), (n0,))
        # Same node → common prefix is the full range.
        common = idx3.get_common_prefix(BranchId("a"), BranchId("b"))
        assert common is not None
        assert common.start_token == 0
        assert common.end_token == 100

    def test_get_common_prefix_disjoint(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",))
        n1 = _node("n-1", 100, 200, ("p-1",))
        idx = PrefixIndex()
        idx2 = idx.register_branch(BranchId("a"), hash_content("a"), (n0,))
        idx3 = idx2.register_branch(BranchId("b"), hash_content("b"), (n1,))
        common = idx3.get_common_prefix(BranchId("a"), BranchId("b"))
        assert common is None

    def test_get_common_prefix_partial(self) -> None:
        """Branch a has nodes [n0, n1, n2]; branch b has [n0, n1, n3]."""
        n0 = _node("n-0", 0, 50, ("p-0",))
        n1 = _node("n-1", 50, 100, ("p-1",))
        n2 = _node("n-2", 100, 150, ("p-2",))
        n3 = _node("n-3", 100, 150, ("p-3",))
        idx = PrefixIndex()
        idx2 = idx.register_branch(BranchId("a"), hash_content("a"), (n0, n1, n2))
        idx3 = idx2.register_branch(BranchId("b"), hash_content("b"), (n0, n1, n3))
        common = idx3.get_common_prefix(BranchId("a"), BranchId("b"))
        assert common is not None
        assert common.start_token == 0
        assert common.end_token == 100
        assert PageId("p-0") in common.page_ids
        assert PageId("p-1") in common.page_ids
        assert PageId("p-2") not in common.page_ids
        assert PageId("p-3") not in common.page_ids

    def test_get_common_prefix_missing_branch(self) -> None:
        idx = PrefixIndex()
        assert idx.get_common_prefix(BranchId("a"), BranchId("b")) is None

    def test_fork_branch(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",), branch_id="a")
        n1 = _node("n-1", 100, 150, ("p-1",), branch_id="a")
        idx = PrefixIndex()
        idx2 = idx.register_branch(BranchId("a"), hash_content("a"), (n0, n1))

        # Fork: reuse n0 (shared), new n2 for suffix
        n0_shared = _node("n-0", 0, 100, ("p-0",), branch_id="a")
        n2_new = _node("n-2", 100, 200, ("p-2",), branch_id="b")
        idx3 = idx2.fork_branch(
            source_branch_id=BranchId("a"),
            new_branch_id=BranchId("b"),
            shared_nodes=(n0_shared,),
            new_nodes=(n2_new,),
        )
        assert len(idx3) == 2
        assert idx3.get_branch(BranchId("b")) is not None
        # Shared node should have ref_count incremented.
        shared = idx3.get_node(NodeId("n-0"))
        assert shared is not None
        assert shared.ref_count == 2

    def test_fork_branch_source_not_found(self) -> None:
        idx = PrefixIndex()
        n0 = _node("n-0", 0, 100, ("p-0",))
        with pytest.raises(KeyError, match="not found"):
            idx.fork_branch(
                source_branch_id=BranchId("missing"),
                new_branch_id=BranchId("b"),
                shared_nodes=(),
                new_nodes=(n0,),
            )

    def test_gc_removes_zero_ref_nodes(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",), ref_count=0)
        n1 = _node("n-1", 100, 200, ("p-1",), ref_count=1)
        idx = PrefixIndex(nodes=(n0, n1))
        idx2 = idx.gc()
        assert len(idx2.nodes) == 1
        assert idx2.get_node(NodeId("n-1")) is not None

    def test_gc_keeps_referenced_nodes(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",), ref_count=0)
        idx = PrefixIndex()
        idx2 = idx.register_branch(BranchId("a"), hash_content("a"), (n0,))
        # n0 is referenced by branch a, so gc should keep it.
        idx3 = idx2.gc()
        assert len(idx3.nodes) == 1

    def test_index_hash_deterministic(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",))
        a = PrefixIndex()
        a2 = a.register_branch(BranchId("b"), hash_content("t"), (n0,))
        b = PrefixIndex()
        b2 = b.register_branch(BranchId("b"), hash_content("t"), (n0,))
        assert a2.index_hash == b2.index_hash

    def test_index_hash_changes(self) -> None:
        n0 = _node("n-0", 0, 100, ("p-0",))
        idx = PrefixIndex()
        idx2 = idx.register_branch(BranchId("b"), hash_content("t"), (n0,))
        assert idx.index_hash != idx2.index_hash
