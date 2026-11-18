"""Helpers shared by the test files: plain functions, `from _helpers import ...`."""

import numpy as np
import scipy.sparse as sp


def random_csr(n_rows=100, n_cols=50, density=0.05, seed=42):
    """A random float32 CSR array, values in [0, 1)."""
    rng = np.random.default_rng(seed)
    return sp.random_array((n_rows, n_cols), density=density, format="csr", dtype=np.float32, random_state=rng)


def raw_csr(rows, cols, vals, shape):
    """CSR of the (row, col, val) entries as given, grouped by row (stable): duplicates,
    column order and stored zeros kept, where scipy's (vals, (rows, cols)) form sums and sorts."""
    rows = np.asarray(rows)
    order = np.argsort(rows, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=shape[0]))])
    return sp.csr_array((np.asarray(vals)[order], np.asarray(cols)[order], indptr), shape=shape)


def assert_same_csr(a, b):
    """Same indptr, indices and data values, entry for entry."""
    for name in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
