.PHONY: install
install:
	uv pip install '.'

.PHONY: install-dev
install-dev:
	uv pip install -e '.[dev,bench,notebook]'

.PHONY: install-dev-editable
# The first time, you must run install-dev before, to have dependencies installed
install-dev-editable:
	SKBUILD_EDITABLE_REBUILD=true uv pip install -e '.[dev,bench,notebook]' --no-build-isolation -v

.PHONY: install-dev-rust
# Rust kernel module (`_similaripy_rs`) next to the Cython package, Step 2a only.
# Rerun after Rust edits; use with _backend='rust' or SIMILARIPY_BACKEND=rust.
install-dev-rust:
	uv run maturin develop --release -m crates/similaripy-py/Cargo.toml

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
.PHONY: test-miri
test-miri:
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p similaripy-core

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
	find . -name '*.so' -path '*/cython_code/*' -delete 2>/dev/null || true

.PHONY: mkdocs
mkdocs:
	uv run mkdocs serve
