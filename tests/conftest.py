
def pytest_configure(config):
    config.addinivalue_line("markers", "perf: mark test as performance-only (used by pytest-benchmark)")
