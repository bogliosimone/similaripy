---
description: Run quick checks before committing code
---

Before committing, run the following checks:

1. **Compile & Test**: Run `make test-dev` to verify compilation and unit tests
   - `uv run` rebuilds the Rust extension after `.rs` edits
   - Watch for cargo/maturin build errors
   - Rust changes: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`
   - Run correctness tests (exclude performance benchmarks)
   - Report any test failures with details

2. **Common Issues Check**:
   - If adding new public functions, verify docstrings are present

If all checks pass, **ask for user approval** before proceeding with the commit. If any check fails, report the issue and suggest fixes.
