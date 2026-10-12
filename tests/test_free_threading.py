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
    return sp.random(2000, 500, density=0.02, format='csr', dtype=np.float32, random_state=42)


def _calls(m):
    return {
        'dot_product': lambda: sim.dot_product(m, k=50, verbose=False, num_threads=2),
        'cosine': lambda: sim.cosine(m, k=50, verbose=False, num_threads=2),
        # default threads: OpenMP team per call / Rust global pool shared by all callers
        'cosine_all_threads': lambda: sim.cosine(m, k=50, verbose=False),
        'rp3beta': lambda: sim.rp3beta(m, k=50, verbose=False, num_threads=2),
        's_plus': lambda: sim.s_plus(m, k=50, verbose=False, num_threads=2),
        'bm25': lambda: sim.bm25(m),
        'normalize': lambda: sim.normalize(m),
    }


@pytest.mark.skipif(not sysconfig.get_config_var('Py_GIL_DISABLED'), reason='free-threaded build only')
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
            np.testing.assert_allclose(got, expected[name], rtol=1e-5, err_msg=name)
    # inputs are not modified by non-inplace calls
    np.testing.assert_array_equal(m.toarray(), _matrix().toarray())
