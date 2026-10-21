---
description: Run correctness tests with make test-dev
---

Run the correctness tests using `make test-dev` and report the results. Before a commit, run `make check` instead (ruff, ty, cargo fmt/clippy/test, then the tests: what CI runs).

**Note:** `uv run` rebuilds the editable install when Rust sources change (`[tool.uv] cache-keys`).

After running the tests:
1. Check if the code compiled successfully (watch for cargo/maturin errors)
2. If any tests failed, analyze the error messages and identify the root cause
3. Suggest fixes for any failing tests
