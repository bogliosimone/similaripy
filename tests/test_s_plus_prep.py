"""Prep regressions: 0.7.0 (Cython) bugs fixed by the numpy prep (v1 Step 2a slice 3)."""
import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim


def _m(rows=60, cols=40, seed=0):
    return sp.random_array((rows, cols), density=0.2, format='csr', dtype=np.float32, rng=seed)


def _same(a, b):
    np.testing.assert_allclose(a.toarray(), b.toarray(), rtol=1e-6)


@pytest.mark.parametrize('selector', ['filter_cols', 'target_cols'])
def test_int64_indices_with_array_selector(selector):
    m = _m()
    m64 = m.copy()
    m64.indices, m64.indptr = m64.indices.astype(np.int64), m64.indptr.astype(np.int64)
    kw = {selector: [1, 2, 7], 'k': 10, 'verbose': False}
    _same(sim.cosine(m64, **kw), sim.cosine(m, **kw))


def test_unsorted_m2_blocked_identity_permutation():
    # Column j is in the first 40 - j rows: popularity already descending, so
    # the popularity permutation is the identity and 0.7.0 skipped the sort.
    cnt = np.arange(40, 0, -1)
    rows = np.concatenate([np.arange(c) for c in cnt])
    cols = np.concatenate([np.full(c, j) for j, c in enumerate(cnt)])
    m2 = sp.csr_array((np.ones(len(rows), np.float32), (rows, cols)), shape=(50, 40))
    m2u = m2.copy()
    for r in range(m2u.shape[0]):
        s, e = m2u.indptr[r], m2u.indptr[r + 1]
        m2u.indices[s:e] = m2u.indices[s:e][::-1].copy()
    m2u.has_sorted_indices = False
    m1 = _m(30, 50, seed=1)
    ref = sim.dot_product(m1, m2, k=40, block_size=None, verbose=False)
    out = sim.dot_product(m1, m2u, k=40, block_size=8, verbose=False)
    _same(out, ref)


def test_inputs_not_mutated():
    m = _m()
    m.data[::7] = 0  # explicit zeros
    f = sp.csr_array((np.array([1, 0, 1], np.float32), np.array([5, 3, 1]), np.array([0, 3] + [3] * 59)),
                     shape=(60, 60))
    m_before, f_before = (m.data.copy(), m.indices.copy()), (f.data.copy(), f.indices.copy())
    sim.cosine(m, k=5, binary=True, filter_cols=f, target_cols=f.copy(), block_size=8,
               verbose=False)
    for arr, before in zip((m.data, m.indices, f.data, f.indices), (*m_before, *f_before)):
        np.testing.assert_array_equal(arr, before)


def test_binary_ignores_explicit_zeros():
    m = _m()
    clean = m.copy()
    m.data[::7] = 0
    clean.data[::7] = 0
    clean.eliminate_zeros()
    kw = {'k': 10, 'binary': True, 'verbose': False}
    _same(sim.jaccard(m, **kw), sim.jaccard(clean, **kw))


@pytest.mark.parametrize('format_output', ['csr', 'coo'])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_output_dtypes(format_output, index_dtype):
    m = _m()
    m.indices, m.indptr = m.indices.astype(index_dtype), m.indptr.astype(index_dtype)
    out = sim.cosine(m, k=10, verbose=False, format_output=format_output, filter_cols=[1, 2])
    assert out.format == format_output and out.data.dtype == np.float32
    idx = (out.indices,) if format_output == 'csr' else out.coords
    assert all(i.dtype == np.int32 for i in idx)


def test_trailing_empty_rows():
    m = _m().tolil()
    m[-3:, :] = 0
    m = m.tocsr()
    m.eliminate_zeros()
    out = sim.cosine(m, k=5, verbose=False)
    assert out.shape == (60, 60) and out.tocsr()[-3:].nnz == 0


def test_progress_bar_closed_on_prep_error(capfd, monkeypatch):
    from similaripy import _s_plus

    def boom(cols):
        raise RuntimeError('prep failed')
    monkeypatch.setattr(_s_plus, '_column_selector', boom)
    with pytest.raises(RuntimeError, match='prep failed'):
        sim.cosine(_m(), k=5, verbose=True)
    assert capfd.readouterr().err.split('\r')[-1].startswith('Failed:')


def test_get_num_threads():
    from similaripy import _s_plus
    assert _s_plus.get_num_threads() >= 1


def test_int32_output_guard():
    # 50000 rows * k=50000 > int32 max: must fail before allocating the output
    m = sp.csr_array((50000, 50000), dtype=np.float32)
    with pytest.raises(ValueError, match='output too large'):
        sim.dot_product(m, k=50000, verbose=False)


def test_negative_block_size():
    with pytest.raises(ValueError, match='block_size'):
        sim.cosine(_m(), k=5, block_size=-1, verbose=False)


def test_no_blocking_matches_blocked():
    m = _m(60, 300)
    kw = {'k': 20, 'verbose': False}
    _same(sim.cosine(m.T, block_size=None, **kw), sim.cosine(m.T, block_size=16, **kw))
