"""Property test of the s_plus prep: random raw inputs (duplicates, explicit zeros,
unsorted rows, selectors) through every similarity, against the dense reference.
Skipped without hypothesis (packagers running the tests with pytest only)."""

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim
from test_similarity import _CASES, _SIGNED_CASES, _check_scores, _dense_scores, _drop_cancellation_noise

hypothesis = pytest.importorskip("hypothesis")
given, settings, st = hypothesis.given, hypothesis.settings, hypothesis.strategies


@st.composite
def _prep_cases(draw):
    signed = draw(st.booleans())
    name, kw = draw(st.sampled_from(_SIGNED_CASES if signed else _CASES))
    n_rows, n_cols = draw(st.integers(1, 25)), draw(st.integers(1, 15))
    entries = draw(
        st.lists(
            st.tuples(st.integers(0, n_rows - 1), st.integers(0, n_cols - 1), st.integers(-3 if signed else 0, 3)),
            max_size=80,
        )
    )
    filter_cols = draw(st.none() | st.lists(st.integers(0, n_rows - 1), max_size=5))
    target_rows = draw(st.none() | st.lists(st.integers(0, n_rows - 1), min_size=1, max_size=5))
    return (
        name,
        kw,
        (n_rows, n_cols),
        entries,
        filter_cols,
        target_rows,
        draw(st.sampled_from(["coo", "csr"])),
        draw(st.integers(1, 25)),
    )


@settings(max_examples=300, deadline=None)
@given(_prep_cases())
def test_random_raw_input_matches_dense_reference(case):
    # The bug class v1 found in 0.7.0's prep: input as given (duplicates summed, explicit
    # zeros and cancelled pairs dropped, rows unsorted), selectors, against the dense float64
    # reference on the summed matrix. Small integers: exact float32 dot products.
    name, kw, shape, entries, filter_cols, target_rows, fmt, k = case
    r, c, v = (np.array(x, dtype=np.int64) for x in zip(*entries)) if entries else (np.zeros(0, np.int64),) * 3
    m = sp.coo_array((v.astype(np.float32), (r, c)), shape=shape)
    if fmt == "csr":  # raw CSR: rows grouped, columns unsorted, duplicates and zeros kept
        order = np.argsort(r, kind="stable")
        m = sp.csr_array(
            (v[order].astype(np.float32), c[order], np.searchsorted(r[order], np.arange(shape[0] + 1))), shape=shape
        )
    scores = _dense_scores(m, name, kw)
    if filter_cols:
        scores[:, filter_cols] = 0
    if target_rows is not None:
        scores[np.setdiff1d(np.arange(shape[0]), target_rows)] = 0
    out = getattr(sim, name)(m, k=k, filter_cols=filter_cols, target_rows=target_rows, **kw)
    _check_scores(_drop_cancellation_noise(out, scores), scores, k)
