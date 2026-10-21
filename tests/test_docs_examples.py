import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["README.md", "docs/guide.md"]


@pytest.mark.parametrize("page", PAGES)
def test_python_blocks_run(page):
    """The page's ```python blocks run in order, sharing one namespace."""
    path = ROOT / page
    if not path.exists():  # the sdist ships tests/ without the docs
        pytest.skip(f"{page} not found")
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), re.DOTALL | re.MULTILINE)
    assert blocks, f"no python blocks in {page}"
    namespace = {}
    for block in blocks:
        exec(compile(block, str(path), "exec"), namespace)  # noqa: S102 (the repo's own docs)
