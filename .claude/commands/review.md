---
description: Review recent code changes for correctness and performance
---

Review the recent code changes in the repository focusing on:

1. **Correctness**: Check for bugs, edge cases, and logic errors

2. **Performance**: Identify optimization opportunities (following CLAUDE.md guidelines)
   - Flag any performance changes without explaining benefit/trade-offs first
   - Look for unnecessary allocations, copies, or computations

3. **Rust Best Practices**:
   - GIL released around kernel calls (`py.detach`), no Python objects touched inside
   - Every `unsafe` block justified by an invariant the Python prep or `CsrView`/`validate` guarantees
   - Pre-allocated per-thread buffers instead of repeated allocations
   - No shared mutable state across rayon workers
   - Float32 vs Float64 usage; codegen-sensitive loops (`tests/benchmarks/README.md`) left intact
   - `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` clean

4. **Docs**: public functions have a docstring (ruff `D103`), updated when a signature changes

5. **Testing**: Suggest additional test cases for edge cases if needed

Provide specific, actionable feedback with file and line references.
