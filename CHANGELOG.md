# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-16

### Added

#### Harness Control Plane (`rfsn_agent/`)
- **Types**: Strong identifiers (`EventId`, `TrajectoryId`, `ItemId`, etc.) and enumerations (`ClaimStatus`, `TaskStatus`, `VerificationResult`, etc.)
- **Common**: Deterministic serialization, SHA-256 hashing, canonical JSON, and dataclass serialization/deserialization utilities
- **Domain**: Immutable domain schemas (`CandidateItem`, `CuratedItem`, `Claim`, `EvidenceLink`, `VerificationRecord`, `TaskNode`, `BudgetLedger`, `ToolInvocation`, `ToolResult`, `SubmissionRecord`, `HarnessSnapshot`) with content hash validation and provenance tracking
- **Events**: 14 typed event payloads with `EventHeader` envelope, HMAC-SHA256 signature chains, and idempotency key support
- **Reducer**: Pure, deterministic event reducer with invariant checks for trajectory continuity, hash chains, and signature chains
- **Store**: SQLite WAL persistence with schema migrations, idempotent commits, CAS offloading for large payloads, and automatic checkpointing
- **Context**: Deterministic context compiler with token budget allocation, priority-based eviction, epoch/prefix reuse, and multi-model rendering (LLaMA3, GPT-4, Claude)
- **Actions**: 13 typed semantic actions (`DecomposeAction`, `SearchAction`, `ReadAction`, `CurateAction`, `DiscardAction`, `VerifyAction`, `ReviseClaimAction`, `PruneSemanticAction`, `RequestContextAction`, `SubmitAction`, `AddCandidateAction`, `CreateClaimAction`, `LinkEvidenceAction`) with safety, budget, and precondition validation
- **Runtime**: Agent runtime loop with optimistic concurrency, stale context retry, and terminal state detection
- **CAS**: Content-addressed filesystem store with atomic writes, corruption detection, and path traversal protection
- **Security**: Safety profiles with tool allow-lists and forbidden path patterns

#### KV Physical Layer (`rfsn_kv/`)
- **Types**: KV identifiers (`PageId`, `NodeId`, `BranchId`, `LayerIndex`, `ContentHash`), `ContentReference` dataclass, and enumerations (`KVPageStatus`, `EvictionState`)
- **Common**: Self-contained hashing and serialization utilities (no `rfsn_agent` dependencies), `CASStore` protocol for content-addressed storage
- **Pages**: Immutable `KVPage` with content hash validation, `create_with_cas()` for offloading, `resolve_content()` for retrieval, and `PageRange` for contiguous token ranges
- **Page Table**: Immutable `PageTable` with binary search lookup, range queries, and copy-on-write mutations (`with_entry()`, `without_entries()`)
- **Codecs**: `KVCodec` protocol with `IdentityCodec` (passthrough) and `QuantizeCodec` (4/8-bit byte-level quantization), global codec registry
- **Persistence**: SQLite WAL-backed page storage with schema migrations, idempotent puts, conflict detection, and layer-based queries
- **Integrity**: SHA-256 verification, corruption detection, and hash repair for stored pages
- **Prefix Index**: Copy-on-write prefix sharing with reference counting, branch registration/forking, common prefix discovery, and garbage collection
- **Residency**: LRU eviction policy, pin-guarded residency management, and immutable cache state tracking
- **Kernels**: `FusedAttentionKernel` protocol stub for Phase 10 Metal/MLX integration

### Testing
- 310 tests across 17 test files
- mypy strict mode: 0 issues
- ruff linting: all checks passed
- Zero cross-boundary imports (`rfsn_agent` ↔ `rfsn_kv`)

### Documentation
- `AGENTS.md`: Build conventions and P0 plan
- `README.md`: Architecture diagram, module inventory, and usage instructions
- `CHANGELOG.md`: This file

## [Unreleased]

### Planned
- Phase 7: OMLX inference adapter
- Phase 8: Multi-file coding benchmark
- Phase 9: Objective evaluator and receipts
- Phase 10: Compressed-page experiments with real float-tensor codecs and Metal/MLX kernels
