# RFSN Agent Harness — Agent Notes

## Project scope

This repository implements the **harness control plane** for RFSN/OMLX agents:
external, deterministic, event-sourced semantic state that lives above the
inference runtime. It does **not** contain model weights, MLX kernels, or
KV-cache compression research code in its initial phases.

## Architecture boundaries

- `rfsn_agent/` owns semantic task state: events, snapshots, actions,
  verification, budgets, context compilation, and trajectory auditing.
- `rfsn_kv/` owns physical KV-cache storage and compression codecs.
- The harness communicates with RFSN/OMLX through the narrow
  `Policy` protocol defined in `rfsn_agent/runtime.py`.
- The harness never manipulates MLX tensors, KV pages, or model weights.

## Coding conventions

- Python 3.11+.
- All domain objects are immutable frozen dataclasses (`frozen=True, slots=True`).
- Every content-bearing object carries a SHA-256 content hash computed at
  construction time; mismatches raise `ValueError`.
- Every domain object carries provenance: `actor`, `action_id`, and
  optionally `event_id`.
- Events are append-only; snapshots are derived by pure reducers.
- Use `tuple` for collections that must be hashable and deterministic.
- Prefer SQLite/CAS filesystem for persistence; avoid Redis for local-first MVP.

## Build/test

```bash
python -m pytest
python -m mypy rfsn_agent rfsn_kv
python -m ruff check rfsn_agent rfsn_kv tests
```

## P0 build order

1. `HarnessEvent` and immutable domain schemas (this commit).
2. Pure event reducer with invariant checks.
3. SQLite/CAS persistence and trajectory isolation.
4. Deterministic `ContextPacket` compiler.
5. Typed semantic actions with validated preconditions.
6. Context epochs and prefix/suffix reuse rules.
7. OMLX inference adapter.
8. One multi-file coding benchmark.
9. Objective evaluator and receipts.
10. Compressed-page experiments.
