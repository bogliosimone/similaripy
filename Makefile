.PHONY: install-dev
install-dev:
	uv sync --all-groups

.PHONY: test-dev
test-dev:
	uv run pytest

# What CI's lint, docs, Rust and test jobs run (Miri and wheels stay in CI)
.PHONY: check
check:
	uv run --only-group lint ruff check
	uv check --no-install-project --group lint --preview-features check-command
	cargo fmt --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace
	uv run pytest
	rm -rf .cache  # as make docs: zensical caches API pages by their .md, not the docstrings
	uv run --only-group docs zensical build --strict

# Undefined-behaviour check of the Rust core unsafe code. One-time setup:
#   rustup toolchain install nightly --component miri rust-src && cargo +nightly miri setup
#   cargo install --locked cargo-nextest
# nextest runs each test in its own Miri process, in parallel (no shared state between tests).
# Tree Borrows: rayon's crossbeam-epoch fails the default Stacked Borrows model
# (int-to-ptr casts, not our code). ignore-leaks: rayon's global pool outlives main.
# CI pins the nightly: make test-miri MIRI_TOOLCHAIN=nightly-YYYY-MM-DD
MIRI_TOOLCHAIN ?= nightly
.PHONY: test-miri
test-miri:
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +$(MIRI_TOOLCHAIN) miri nextest run -p similaripy-core

.PHONY: build
build:
	uv build

.PHONY: benchmark-similarity-small
benchmark-similarity-small:
	uv run --group bench python benchmarks/run.py similarity --dataset movielens --version 32m

.PHONY: benchmark-similarity-medium
benchmark-similarity-medium:
	uv run --group bench python benchmarks/run.py similarity --dataset yambda --version 50m

.PHONY: benchmark-similarity-large
benchmark-similarity-large:
	uv run --group bench python benchmarks/run.py similarity --dataset yambda --version 500m

.PHONY: clean
clean:
	rm -rf dist/
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -f similaripy/_core*.so similaripy/_core*.pyd

.PHONY: docs
docs:
	rm -rf .cache  # zensical caches API pages by their .md, not the docstrings
	uv run --only-group docs zensical serve
