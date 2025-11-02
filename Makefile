.PHONY: install test build wheelcheck benchmark all clean mkdocs test-dev benchmark-local benchmark-normalization benchmark-sim install-dev install-dev-editable benchmark-similarity-rnd-seed benchmark-similarity-medium

# Install all dev dependencies using uv
install:
	uv pip install '.'

# Install dev dependencies using uv
install-dev:
	uv pip install -e '.[dev,bench]'

# Install in editable mode with skbuild redirect
# The first time, you must run install-dev before, to have dependencies installed
install-dev-editable:
	SKBUILD_EDITABLE_REBUILD=true uv pip install -e '.[dev,bench]' --no-build-isolation -v

# Run unit tests using tox (py311)
test:
	uv run tox -e py311

# Run unit test locally (excluding performance tests)
test-dev:
	uv run pytest

# Build wheel + sdist
build:
	uv run tox -e build

# Build, install from wheel, and run tests (package integrity)
wheelcheck:
	uv run tox -e package-test

# Run benchmarks (tests marked with @pytest.mark.perf)
benchmark:
	uv run tox -e benchmark

# Run benchmarks locally
benchmark-similarity:
	uv run tests/benchmarks/run_benchmarks.py

# Run benchmarks on yambda 50m dataset
benchmark-similarity-medium:
	uv run tests/benchmarks/run_benchmarks.py --dataset yambda --version 50m --rounds 3

# Run only normalization benchmarks locally
benchmark-normalization:
	uv run pytest tests/benchmarks/benchmarks_rnd_seed.py::TestNormalizationPerformance --benchmark-only

# Run only similarity benchmarks locally
benchmark-similarity-rnd-seed:
	uv run pytest tests/benchmarks/benchmarks_rnd_seed.py::TestSPlusPerformance --benchmark-only

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info

# Run
mkdocs:
	uv run mkdocs serve
