.PHONY: install test build wheelcheck benchmark all clean mkdocs test-local

# Install all dev dependencies using uv
install:
	uv pip install '.[dev]'

# Run unit tests using tox (py311)
test:
	uv run tox -e py311

# Run unit test locally (excluding performance tests)
test-local:
	uv run pytest -m "not perf"

# Build wheel + sdist
build:
	uv run tox -e build

# Build, install from wheel, and run tests (package integrity)
wheelcheck:
	uv run tox -e package-test

# Run performance benchmarks (tests marked with @pytest.mark.perf)
benchmark:
	uv run tox -e benchmark

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info

# Run
mkdocs:
	uv run mkdocs serve
