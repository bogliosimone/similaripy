.PHONY: install
install:
	uv pip install '.'

.PHONY: install-dev
install-dev:
	uv pip install -e '.[dev,bench,notebook]'

.PHONY: test
test:
	uv run tox -e py311

.PHONY: test-dev
test-dev:
	uv run pytest

# Undefined-behaviour check of the Rust core unsafe code. One-time setup:
#   rustup toolchain install nightly --component miri rust-src && cargo +nightly miri setup
# Tree Borrows: rayon's crossbeam-epoch fails the default Stacked Borrows model
# (int-to-ptr casts, not our code). ignore-leaks: rayon's global pool outlives main.
# CI pins the nightly: make test-miri MIRI_TOOLCHAIN=nightly-YYYY-MM-DD
MIRI_TOOLCHAIN ?= nightly
.PHONY: test-miri
test-miri:
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +$(MIRI_TOOLCHAIN) miri test -p similaripy-core

.PHONY: build
build:
	uv run tox -e build

.PHONY: wheelcheck
wheelcheck:
	uv run tox -e package-test

.PHONY: benchmark-similarity-small
benchmark-similarity-small:
	uv run tests/benchmarks/run_benchmarks.py --dataset movielens --version 32m --rounds 1

.PHONY: benchmark-similarity-medium
benchmark-similarity-medium:
	uv run tests/benchmarks/run_benchmarks.py --dataset yambda --version 50m --rounds 1

.PHONY: benchmark-similarity-large
benchmark-similarity-large:
	uv run tests/benchmarks/run_benchmarks.py --dataset yambda --version 500m --rounds 1

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f similaripy/_core*.so similaripy/_core*.pyd

.PHONY: mkdocs
mkdocs:
	uv run mkdocs serve
