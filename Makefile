.PHONY: install test build wheelcheck benchmark all clean mkdocs test-dev benchmark-local benchmark-normalization benchmark-sim install-dev install-dev-editable

# Install all dev dependencies using uv
install:
	uv pip install '.'

# Install dev dependencies using uv
install-dev:
	uv pip install -e '.[dev]'

# Install in editable mode with skbuild redirect
# The first time, you must run install-dev before, to have dependencies installed
install-dev-editable:
	SKBUILD_EDITABLE_REBUILD=true uv pip install -e '.[dev]' --no-build-isolation

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
benchmark-local:
	uv run pytest tests/benchmarks.py --benchmark-only

# Run only normalization benchmarks locally
benchmark-normalization:
	uv run pytest tests/benchmarks.py::TestNormalizationPerformance --benchmark-only

# Run only similarity benchmarks locally
benchmark-similarity:
	uv run pytest tests/benchmarks.py::TestSPlusPerformance --benchmark-only

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info

# Run
mkdocs:
	uv run mkdocs serve
