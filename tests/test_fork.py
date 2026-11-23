"""Load-then-fork: a child reuses no thread pool its parent started (their
workers don't survive fork()). Every child runs under a timeout, so a hang
fails the test instead of CI."""

import multiprocessing as mp
import os
import signal
import sys
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim
from similaripy.normalization import bm25
from similaripy.serving import SparseKNN

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no fork() on Windows")

HIST = [[1, 2, 3], [4, 5], [6]]


def _m():
    return sp.random_array((300, 80), density=0.05, format="csr", dtype=np.float32, rng=0)


def _calls(m, model):
    """Every pool kind: the global pool, a cached pool (num_threads=2), a
    normalization, a serving batch."""
    return [
        (sim.cosine, (m.T,), {"k": 10}),
        (sim.cosine, (m.T,), {"k": 10, "num_threads": 2}),
        (bm25, (m,), {}),
        (SparseKNN.recommend_batch, (model, HIST), {}),
    ]


def _same(a, b):
    """Same matrix bits, or the same (ids, scores) lists of arrays."""
    if sp.issparse(a):
        return a.dtype == b.dtype and (a != b).nnz == 0
    return all(np.array_equal(x, y) for xs, ys in zip(a, b, strict=True) for x, y in zip(xs, ys, strict=True))


@pytest.mark.parametrize("method", [m for m in ("fork", "spawn", "forkserver") if m in mp.get_all_start_methods()])
def test_child_after_parallel_calls(method):
    m = _m()
    model = SparseKNN(sim.cosine(m.T, k=10))
    calls = _calls(m, model)
    expected = [f(*a, **kw) for f, a, kw in calls]  # the parent starts every pool first
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # 3.12+: fork() of a multi-threaded process
        with mp.get_context(method).Pool(1) as pool:
            got = [r.get(timeout=60) for r in [pool.apply_async(f, a, kw) for f, a, kw in calls]]
    for g, e in zip(got, expected, strict=True):
        assert _same(g, e)


def _forked(fn):
    """fn() in an os.fork() child killed by SIGALRM after 60 s; its exit code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        signal.alarm(60)
        code = 2  # an exception: the child never returns into pytest
        try:
            code = 0 if fn() else 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)  # -14: SIGALRM, a hang


def test_mmap_model_then_fork(tmp_path):
    # gunicorn --preload: the parent loads (mmap) and serves; the forked child
    # serves from the same mapping, single and batch
    SparseKNN(sim.cosine(_m().T, k=10)).save(tmp_path / "m")
    model = SparseKNN.load(tmp_path / "m")
    ids, scores = model.recommend_batch(HIST)
    one = model.recommend(HIST[0])

    def child():
        i, s = model.recommend_batch(HIST)
        o = model.recommend(HIST[0])
        return _same((i, s), (ids, scores)) and _same(([o[0]], [o[1]]), ([one[0]], [one[1]]))

    assert _forked(child) == 0


def test_fork_of_a_fork():
    # a grandchild rebuilds again (its pid is neither the parent's nor the child's)
    m = _m()
    expected = sim.cosine(m.T, k=10, num_threads=2)
    sim.cosine(m.T, k=10)

    def grandchild():
        return _same(sim.cosine(m.T, k=10, num_threads=2), expected) and _same(sim.cosine(m.T, k=10), expected)

    def child():
        sim.cosine(m.T, k=10, num_threads=2)  # the child's own cached pool
        return _forked(grandchild) == 0 and grandchild()

    assert _forked(child) == 0
