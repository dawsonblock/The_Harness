"""Prefix sharing and copy-on-write indexing for KV cache epochs.

When the harness computes a ``ContextEpoch`` and identifies
``common_prefix_tokens``, the inference adapter uses this module to:

1. Register the current page table as a branch.
2. Fork a new branch for the invalidated suffix.
3. Reuse the common prefix pages across branches via copy-on-write
   reference counting.

The ``branch_id`` field on each ``PrefixBranch`` corresponds to
``ContextEpoch.cache_branch_id`` (the first 16 characters of the
packet hash). This is a documented contract — no code import from
``rfsn_agent`` is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rfsn_kv.common import canonical_json, hash_content, utc_now
from rfsn_kv.pages import PageRange
from rfsn_kv.types import BranchId, ContentHash, NodeId


@dataclass(frozen=True, slots=True)
class PrefixNode:
    """A shared prefix node in the copy-on-write tree.

    Attributes:
        node_id: Unique identifier for this node.
        page_range: The token range this node covers.
        node_hash: SHA-256 content hash of this node's structural fields.
        ref_count: Number of branches referencing this node.
        branch_id: The branch that created this node.
        actor: Entity that created this node.
        action_id: Action identifier for provenance tracking.
        event_id: Optional event identifier for provenance tracking.
    """

    node_id: NodeId
    page_range: PageRange
    node_hash: ContentHash
    ref_count: int = 1
    branch_id: BranchId = BranchId("")
    actor: str = "system"
    action_id: str = "init"
    event_id: str | None = None

    def __post_init__(self) -> None:
        expected = self._compute_hash()
        if expected != self.node_hash:
            raise ValueError(
                f"PrefixNode {self.node_id}: node_hash mismatch: "
                f"expected {expected}, got {self.node_hash}"
            )

    def _compute_hash(self) -> ContentHash:
        payload = {
            "node_id": self.node_id,
            "page_range_start": self.page_range.start_token,
            "page_range_end": self.page_range.end_token,
            "branch_id": self.branch_id,
        }
        return hash_content(canonical_json(payload))

    @classmethod
    def create(
        cls,
        *,
        node_id: NodeId,
        page_range: PageRange,
        ref_count: int = 1,
        branch_id: BranchId = BranchId(""),
        actor: str = "system",
        action_id: str = "init",
        event_id: str | None = None,
    ) -> PrefixNode:
        """Create a node with a correctly computed node hash."""
        node = cls.__new__(cls)
        object.__setattr__(node, "node_id", node_id)
        object.__setattr__(node, "page_range", page_range)
        object.__setattr__(node, "ref_count", ref_count)
        object.__setattr__(node, "branch_id", branch_id)
        object.__setattr__(node, "actor", actor)
        object.__setattr__(node, "action_id", action_id)
        object.__setattr__(node, "event_id", event_id)
        object.__setattr__(node, "node_hash", node._compute_hash())
        return node

    def with_incremented_ref(self) -> PrefixNode:
        """Return a new node with ref_count + 1."""
        return PrefixNode(
            node_id=self.node_id,
            page_range=self.page_range,
            node_hash=self.node_hash,
            ref_count=self.ref_count + 1,
            branch_id=self.branch_id,
            actor=self.actor,
            action_id=self.action_id,
            event_id=self.event_id,
        )

    def with_decremented_ref(self) -> PrefixNode:
        """Return a new node with ref_count - 1."""
        return PrefixNode(
            node_id=self.node_id,
            page_range=self.page_range,
            node_hash=self.node_hash,
            ref_count=max(0, self.ref_count - 1),
            branch_id=self.branch_id,
            actor=self.actor,
            action_id=self.action_id,
            event_id=self.event_id,
        )


@dataclass(frozen=True, slots=True)
class PrefixBranch:
    """A branch in the prefix index, analogous to a git branch.

    Attributes:
        branch_id: Unique identifier (typically first 16 chars of packet hash).
        root_node_id: The root node of this branch's prefix tree.
        branch_hash: SHA-256 content hash of this branch's structural fields.
        node_ids: Ordered tuple of node IDs that make up this branch's chain.
        page_table_hash: Hash of the page table this branch was created from.
        created_at: UTC creation timestamp.
        actor: Entity that created this branch.
        action_id: Action identifier for provenance tracking.
        event_id: Optional event identifier for provenance tracking.
    """

    branch_id: BranchId
    root_node_id: NodeId
    branch_hash: ContentHash
    node_ids: tuple[NodeId, ...] = field(default_factory=tuple)
    page_table_hash: ContentHash = field(default=ContentHash(""))
    created_at: datetime = field(default_factory=utc_now)
    actor: str = "system"
    action_id: str = "init"
    event_id: str | None = None

    def __post_init__(self) -> None:
        expected = self._compute_hash()
        if expected != self.branch_hash:
            raise ValueError(
                f"PrefixBranch {self.branch_id}: branch_hash mismatch: "
                f"expected {expected}, got {self.branch_hash}"
            )

    def _compute_hash(self) -> ContentHash:
        payload = {
            "branch_id": self.branch_id,
            "root_node_id": self.root_node_id,
            "node_ids": list(self.node_ids),
            "page_table_hash": self.page_table_hash,
        }
        return hash_content(canonical_json(payload))


def _create_branch(
    *,
    branch_id: BranchId,
    root_node_id: NodeId,
    node_ids: tuple[NodeId, ...],
    page_table_hash: ContentHash,
    actor: str = "system",
    action_id: str = "init",
    event_id: str | None = None,
) -> PrefixBranch:
    """Create a PrefixBranch with a correctly computed branch_hash."""
    branch = PrefixBranch.__new__(PrefixBranch)
    object.__setattr__(branch, "branch_id", branch_id)
    object.__setattr__(branch, "root_node_id", root_node_id)
    object.__setattr__(branch, "node_ids", node_ids)
    object.__setattr__(branch, "page_table_hash", page_table_hash)
    object.__setattr__(branch, "created_at", utc_now())
    object.__setattr__(branch, "actor", actor)
    object.__setattr__(branch, "action_id", action_id)
    object.__setattr__(branch, "event_id", event_id)
    object.__setattr__(branch, "branch_hash", branch._compute_hash())
    return branch


@dataclass(frozen=True, slots=True)
class PrefixIndex:
    """An immutable copy-on-write prefix index with reference counting.

    The index tracks branches and their shared prefix nodes. Branches that
    share a common prefix reference the same nodes (ref_count > 1). When a
    branch is forked, only the invalidated suffix nodes are duplicated; the
    shared prefix nodes have their ref_count incremented.

    Every mutation returns a new ``PrefixIndex`` instance.

    Attributes:
        branches: All registered branches.
        nodes: All prefix nodes, keyed by node_id.
    """

    branches: tuple[PrefixBranch, ...] = field(default_factory=tuple)
    nodes: tuple[PrefixNode, ...] = field(default_factory=tuple)
    index_hash: ContentHash = field(default=ContentHash(""))

    def __post_init__(self) -> None:
        expected = self._compute_hash()
        if self.index_hash != expected:
            object.__setattr__(self, "index_hash", expected)

    def _compute_hash(self) -> ContentHash:
        payload = {
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "root_node_id": b.root_node_id,
                    "page_table_hash": b.page_table_hash,
                }
                for b in self.branches
            ],
            "nodes": [
                {
                    "node_id": n.node_id,
                    "page_range_start": n.page_range.start_token,
                    "page_range_end": n.page_range.end_token,
                    "page_ids": list(n.page_range.page_ids),
                    "ref_count": n.ref_count,
                }
                for n in self.nodes
            ],
        }
        return hash_content(canonical_json(payload))

    def register_branch(
        self,
        branch_id: BranchId,
        page_table_hash: ContentHash,
        nodes: tuple[PrefixNode, ...],
    ) -> PrefixIndex:
        """Register a new branch with the given prefix nodes.

        Returns a new PrefixIndex with the branch and nodes added.
        """
        node_ids = tuple(n.node_id for n in nodes)
        root_id = nodes[0].node_id if nodes else NodeId("empty")
        branch = _create_branch(
            branch_id=branch_id,
            root_node_id=root_id,
            node_ids=node_ids,
            page_table_hash=page_table_hash,
        )
        return PrefixIndex(
            branches=self.branches + (branch,),
            nodes=self.nodes + nodes,
        )

    def get_branch(self, branch_id: BranchId) -> PrefixBranch | None:
        """Look up a branch by its ID."""
        for b in self.branches:
            if b.branch_id == branch_id:
                return b
        return None

    def get_node(self, node_id: NodeId) -> PrefixNode | None:
        """Look up a node by its ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_common_prefix(
        self, branch_a: BranchId, branch_b: BranchId
    ) -> PageRange | None:
        """Find the longest shared prefix between two branches.

        Walks from the root of each branch and finds the longest common
        sequence of nodes. Returns the PageRange of the shared prefix,
        or None if the branches have no common nodes.
        """
        ba = self.get_branch(branch_a)
        bb = self.get_branch(branch_b)
        if ba is None or bb is None:
            return None

        node_map = {n.node_id: n for n in self.nodes}

        # Get the node chains from both branches.
        chain_a = [node_map[nid] for nid in ba.node_ids if nid in node_map]
        chain_b = [node_map[nid] for nid in bb.node_ids if nid in node_map]

        # Find common prefix length.
        common_len = 0
        for na, nb in zip(chain_a, chain_b):
            if na.node_id != nb.node_id:
                break
            common_len += 1

        if common_len == 0:
            return None

        # The shared prefix range is from root to the last common node.
        last_common = chain_a[common_len - 1]
        return PageRange(
            start_token=chain_a[0].page_range.start_token,
            end_token=last_common.page_range.end_token,
            page_ids=tuple(
                nid for n in chain_a[:common_len] for nid in n.page_range.page_ids
            ),
        )

    def fork_branch(
        self,
        source_branch_id: BranchId,
        new_branch_id: BranchId,
        shared_nodes: tuple[PrefixNode, ...],
        new_nodes: tuple[PrefixNode, ...],
    ) -> PrefixIndex:
        """Fork a new branch from ``source_branch_id``.

        ``shared_nodes`` are the prefix nodes reused from the source
        (their ref_counts are incremented). ``new_nodes`` are the
        invalidated suffix nodes for the new branch.
        """
        source = self.get_branch(source_branch_id)
        if source is None:
            raise KeyError(f"Source branch {source_branch_id!r} not found")

        # Build updated node list: increment shared, keep existing, add new.
        shared_ids = {n.node_id for n in shared_nodes}
        updated_nodes: list[PrefixNode] = []
        for existing in self.nodes:
            if existing.node_id in shared_ids:
                updated_nodes.append(existing.with_incremented_ref())
            else:
                updated_nodes.append(existing)
        updated_nodes.extend(new_nodes)

        new_branch = _create_branch(
            branch_id=new_branch_id,
            root_node_id=new_nodes[0].node_id if new_nodes else source.root_node_id,
            node_ids=tuple(n.node_id for n in shared_nodes) + tuple(n.node_id for n in new_nodes),
            page_table_hash=source.page_table_hash,
        )

        return PrefixIndex(
            branches=self.branches + (new_branch,),
            nodes=tuple(updated_nodes),
        )

    def gc(self) -> PrefixIndex:
        """Remove nodes with ref_count == 0 and no branch references."""
        referenced_node_ids: set[NodeId] = set()
        for b in self.branches:
            referenced_node_ids.update(b.node_ids)

        surviving = tuple(
            n
            for n in self.nodes
            if n.ref_count > 0 or n.node_id in referenced_node_ids
        )
        return PrefixIndex(branches=self.branches, nodes=surviving)

    def __len__(self) -> int:
        return len(self.branches)
