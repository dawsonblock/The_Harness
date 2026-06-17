"""Physical KV-cache storage and compression codecs.

This module provides the immutable, content-hashed physical layer for KV-cache
pages. It is self-contained: it does NOT import from ``rfsn_agent``. The
inference adapter bridges these types with the harness via the
``ContextEpoch.cache_branch_id`` contract.
"""

from rfsn_kv.codecs import CODEC_REGISTRY, get_codec
from rfsn_kv.codecs.identity import IdentityCodec
from rfsn_kv.codecs.quantize import QuantizeCodec
from rfsn_kv.common import canonical_json, hash_bytes, hash_content, sha256_hash
from rfsn_kv.integrity import IntegrityChecker, IntegrityReport
from rfsn_kv.page_table import PageTable, PageTableEntry
from rfsn_kv.pages import KVPage, PageRange
from rfsn_kv.persistence import KVPersistence
from rfsn_kv.prefix_index import PrefixBranch, PrefixIndex, PrefixNode
from rfsn_kv.residency import LRUEvictionPolicy, ResidencyManager, ResidentPageEntry
from rfsn_kv.types import (
    BranchId,
    ContentHash,
    EvictionState,
    KVPageStatus,
    LayerIndex,
    NodeId,
    PageId,
)

__all__: list[str] = [
    "BranchId",
    "CODEC_REGISTRY",
    "ContentHash",
    "EvictionState",
    "get_codec",
    "IdentityCodec",
    "IntegrityChecker",
    "IntegrityReport",
    "canonical_json",
    "KVPage",
    "KVPageStatus",
    "KVPersistence",
    "LayerIndex",
    "LRUEvictionPolicy",
    "NodeId",
    "PageId",
    "PageRange",
    "PageTable",
    "PageTableEntry",
    "PrefixBranch",
    "PrefixIndex",
    "PrefixNode",
    "QuantizeCodec",
    "ResidentPageEntry",
    "ResidencyManager",
    "hash_bytes",
    "hash_content",
    "sha256_hash",
]
