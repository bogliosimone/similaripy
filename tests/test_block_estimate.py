"""block_size='auto' is the core's estimate from the width, the L2 cache and the threads (4c); results don't
change with the block size (the block tests check bits across sizes)."""

import sys

from similaripy import _core  # ty: ignore[unresolved-import]
from similaripy._s_plus import DEFAULT_BLOCK_SIZE


def test_auto_block_size_is_the_estimate():
    blocks = [_core.auto_block_size(n, t) for n, t in ((10_000, 1), (50_000_000, 1), (50_000_000, 64))]
    if sys.platform in ("darwin", "linux"):  # L2 readable
        assert blocks[0] == 0  # 40 KB fits any L2: one block
        assert blocks[2] >= DEFAULT_BLOCK_SIZE  # 200 MB never fits
        assert all(b == 0 or b >= DEFAULT_BLOCK_SIZE for b in blocks)  # never below the floor
    else:  # unknown cache: the fixed size
        assert blocks == [DEFAULT_BLOCK_SIZE] * 3


def test_auto_block_size_counts_hot_bytes_per_column():
    # more bytes per column (the drain's norm arrays) never gives a larger block
    n = 50_000_000
    b4, b8, b12 = (_core.auto_block_size(n, 1, c) for c in (4, 8, 12))
    assert _core.auto_block_size(n, 1) == b4  # default: the accumulator alone
    if sys.platform in ("darwin", "linux"):
        assert b4 >= b8 >= b12 >= DEFAULT_BLOCK_SIZE
