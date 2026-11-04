"""Private switch SIMILARIPY_AUTO_BLOCK=estimate (4c step 1b): 'auto' picks the block size from the width, the L2
cache and the threads; results don't change (block size never changes the addition order). Read once per process,
so the estimate runs in a child process (cwd=tmp_path: the child imports the installed package, not the source)."""

import os
import pickle
import subprocess
import sys

import similaripy as sim
from similaripy import _core  # ty: ignore[unresolved-import]

CHILD = """
import pickle, sys
import numpy as np, scipy.sparse as sp
import similaripy as sim
from similaripy import _core
from similaripy.serving import ItemKNN
URM = sp.random(300, 5000, density=0.02, format="csr", random_state=0, dtype=np.float32)
S = sim.cosine(URM.T, k=20, verbose=False, format_output="csr", num_threads=2)
rec = ItemKNN(S)
recs = rec.recommend_batch([URM[u].indices for u in range(50)], k=10)
blocks = [_core.auto_block_size(n, t) for n, t in ((10_000, 1), (50_000_000, 1), (50_000_000, 64))]
sys.stdout.buffer.write(pickle.dumps((S, recs, blocks)))
"""


def _child(tmp_path, env):
    out = subprocess.run([sys.executable, "-c", CHILD], cwd=tmp_path, env=env, capture_output=True, check=True)
    return pickle.loads(out.stdout)


def test_default_is_unchanged():
    assert _core.auto_block_size(10**8, 1) == sim._s_plus.DEFAULT_BLOCK_SIZE


def test_estimate_same_results(tmp_path):
    # plumbing check (both layers read the switch); block size vs bits is covered by the block tests
    base = {k: v for k, v in os.environ.items() if k != "SIMILARIPY_AUTO_BLOCK"}
    S0, recs0, blocks0 = _child(tmp_path, base)
    S1, recs1, blocks1 = _child(tmp_path, {**base, "SIMILARIPY_AUTO_BLOCK": "estimate"})
    assert blocks0 == [sim._s_plus.DEFAULT_BLOCK_SIZE] * 3
    if sys.platform in ("darwin", "linux"):  # cache readable
        assert blocks1[0] == 0  # 40 KB fits any L2
        assert all(b == 0 or b >= sim._s_plus.DEFAULT_BLOCK_SIZE for b in blocks1)
        assert blocks1[2] >= sim._s_plus.DEFAULT_BLOCK_SIZE  # 200 MB never fits
    assert (S0 != S1).nnz == 0 and (S0.data.view("i4") == S1.data.view("i4")).all()
    for a, b in zip(recs0, recs1):
        for x, y in zip(a, b):
            assert (x.view("i4") == y.view("i4")).all() if x.dtype.kind == "f" else (x == y).all()
