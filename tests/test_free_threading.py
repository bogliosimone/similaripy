import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim

N_THREADS = 8
ROUNDS = 4


def _matrix():
    return sp.random(2000, 500, density=0.02, format="csr", dtype=np.float32, random_state=42)


def _calls(m):
    return {
        "dot_product": lambda: sim.dot_product(m, k=50, verbose=False, num_threads=2),
        "cosine": lambda: sim.cosine(m, k=50, verbose=False, num_threads=2),
        # default threads: the rayon global pool, shared by all callers
        "cosine_all_threads": lambda: sim.cosine(m, k=50, verbose=False),
        "rp3beta": lambda: sim.rp3beta(m, k=50, verbose=False, num_threads=2),
        "s_plus": lambda: sim.s_plus(m, k=50, verbose=False, num_threads=2),
        "bm25": lambda: sim.bm25(m),
        "normalize": lambda: sim.normalize(m),
    }


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="free-threaded build only")
def test_gil_stays_disabled():
    # importing a module not marked freethreading_compatible re-enables the GIL
    assert not sys._is_gil_enabled()


def test_concurrent_calls_match_serial():
    m = _matrix()
    calls = _calls(m)
    expected = {name: fn().toarray() for name, fn in calls.items()}

    def run(name):
        return name, calls[name]().toarray()

    jobs = [name for name in calls for _ in range(ROUNDS)] * 2
    with ThreadPoolExecutor(N_THREADS) as pool:
        for name, got in pool.map(run, jobs):
            np.testing.assert_array_equal(got, expected[name], err_msg=name)  # same machine: same bits
    # inputs are never modified
    np.testing.assert_array_equal(m.toarray(), _matrix().toarray())


def _non_canonical_coo():
    """Shuffled COO with duplicates, past the 2^16-entry leaf size: canonicalized in parallel."""
    rng = np.random.default_rng(7)
    n = 150_000
    rows, cols = rng.integers(0, 2000, n), rng.integers(0, 500, n)
    return sp.coo_array((rng.random(n).astype(np.float32), (rows, cols)), shape=(2000, 500))


def _bits(x):
    x = x.tocsr()
    return x.data.tobytes(), x.indices.tobytes(), x.indptr.tobytes()


@pytest.mark.parametrize("name", ["dot_product", "cosine", "rp3beta", "s_plus"])
def test_similarity_identical_across_threads_and_blocking(name):
    # Same version, same machine: the result never depends on thread count or blocking
    m = _non_canonical_coo()
    f = getattr(sim, name)
    runs = [_bits(f(m, k=50, verbose=False, num_threads=n, block_size=bs)) for n in (1, 4, 0) for bs in (None, 64)]
    assert all(r == runs[0] for r in runs[1:])


_NORMALIZE_SCRIPT = """
import hashlib, numpy as np, scipy.sparse as sp, similaripy as sim
rng = np.random.default_rng(7)
n = 150_000
m = sp.coo_array((rng.random(n).astype(np.float32),
                  (rng.integers(0, 2000, n), rng.integers(0, 500, n))), shape=(2000, 500)).tocsr()
for x in (sim.normalize(m, norm='l2'), sim.tfidf(m), sim.bm25(m), sim.bm25plus(m)):
    print(hashlib.sha256(x.data.tobytes() + x.indices.tobytes() + x.indptr.tobytes()).hexdigest())
"""


def test_normalization_identical_across_threads(tmp_path):
    # Normalizations run on the global pool, sized once per process: one child per size.
    # cwd outside the repo: `python -c` puts the cwd first on sys.path, and the source
    # tree's `similaripy/` (no compiled _core) would shadow the installed package.
    import os
    import subprocess

    def run(n):
        env = {**os.environ, "RAYON_NUM_THREADS": str(n)}
        r = subprocess.run(
            [sys.executable, "-c", _NORMALIZE_SCRIPT],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        return r.stdout

    assert run(1) == run(4)
