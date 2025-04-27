# tests/test_perf_s_plus.py

import numpy as np
import scipy.sparse as sp
import pytest

from similaripy import s_plus


def generate_sparse_matrix(n_rows, n_cols, density=0.01, random_state=42):
    rng = np.random.default_rng(random_state)
    nnz = int(n_rows * n_cols * density)
    row = rng.integers(0, n_rows, nnz)
    col = rng.integers(0, n_cols, nnz)
    data = rng.random(nnz, dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n_rows, n_cols)).tocsr()


@pytest.mark.perf
@pytest.mark.parametrize("n_rows,n_cols,density,k", [
    (6_000, 4_000, 0.04, 50),      # small test case (MovieLens1M)
    (30_000, 140_000, 0.005, 50),  # medium-scale test (MovieLens20M)
])
def test_s_plus_perf_basic(benchmark, n_rows, n_cols, density, k):
    mat = generate_sparse_matrix(n_rows, n_cols, density)
    result = benchmark(
        s_plus,
        mat,
        k=k,
        verbose=False,
        binary=False,
        format_output='csr',
        num_threads=0  # use all available cores
    )
    assert result.nnz > 0
    assert result.shape[0] == n_rows
