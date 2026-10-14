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


@pytest.mark.parametrize('limit, dtype', [(None, np.int32), (100, np.int64)])
def test_csr_index_dtype_past_int32_nnz(limit, dtype, monkeypatch):
    # D42: past 2^31 results the CSR gets int64 indptr and indices; tested at a
    # lowered limit (a real run needs ~26 GB). Same matrix either way.
    from similaripy import _s_plus
    m = _m(60, 40)
    ref = sim.cosine(m, k=10, format_output='csr', verbose=False)
    if limit is not None:
        monkeypatch.setattr(_s_plus, '_MAX_INT32_NNZ', limit)
    out = sim.cosine(m, k=10, format_output='csr', verbose=False)
    assert out.nnz > 100
    assert out.indptr.dtype == out.indices.dtype == dtype
    assert out.has_canonical_format
    for a in ('indptr', 'indices', 'data'):
        np.testing.assert_array_equal(getattr(out, a), getattr(ref, a))


def test_negative_block_size():
    with pytest.raises(ValueError, match='block_size'):
        sim.cosine(_m(), k=5, block_size=-1, verbose=False)


def test_no_blocking_matches_blocked():
    m = _m(60, 300)
    kw = {'k': 20, 'verbose': False}
    _same(sim.cosine(m.T, block_size=None, **kw), sim.cosine(m.T, block_size=16, **kw))


@pytest.mark.parametrize('order', ['popularity', 'original'])
@pytest.mark.parametrize('block_size', [None, 8, 64])
def test_tie_rule_independent_of_blocking(order, block_size, monkeypatch):
    # F4: binary dot scores are small integers, so ties at the k-th place are
    # everywhere; the kept columns must not depend on block size or push order.
    # Ties go to the more popular column (m2 column count), then the lower id;
    # with the 'original' column order, to the lower id.
    from similaripy import _s_plus
    monkeypatch.setattr(_s_plus, '_COLUMN_ORDER', order)
    m = _m(60, 300)
    k = 10
    b = sim.normalization.binarize(m)
    xy = (b.T @ b).toarray()
    pop = np.diff(b.tocsc().indptr) if order == 'popularity' else np.zeros(m.shape[1])
    out = sim.dot_product(m.T, k=k, binary=True, block_size=block_size, verbose=False).tocsr()
    for row in range(xy.shape[0]):
        cols = np.flatnonzero(xy[row])
        best = cols[np.lexsort((cols, -pop[cols], -xy[row, cols]))][:k]
        got = out.indices[out.indptr[row]:out.indptr[row + 1]]
        got = got[out.data[out.indptr[row]:out.indptr[row + 1]] != 0]
        assert sorted(got.tolist()) == sorted(best.tolist()), row


def test_coo_blocked_keeps_zero_score_column():
    # Column 5 is the most popular (permuted to 0) and scores exactly 0.0 for row 0.
    # The un-permute treated (col 0, val 0.0) as padding and left it at column 0.
    m2 = np.zeros((3, 8), dtype=np.float32)
    m2[0, 5], m2[1, 5], m2[2, 5], m2[0, 3] = 1, -1, 1, 1
    m1 = sp.csr_array(np.array([[1, 1, 0]], dtype=np.float32))
    kw = {'k': 4, 'format_output': 'coo', 'verbose': False}
    ref = sim.dot_product(m1, sp.csr_array(m2), block_size=None, **kw)
    out = sim.dot_product(m1, sp.csr_array(m2), block_size=2, **kw)
    assert (0, 5, 0.0) in zip(out.row.tolist(), out.col.tolist(), out.data.tolist())
    assert sorted(zip(out.row, out.col, out.data)) == sorted(zip(ref.row, ref.col, ref.data))


def _with_duplicates(m, seed=0):
    """m's entries plus split copies (v = a + b) and cancelling pairs (+1, -1), shuffled within rows."""
    rng = np.random.default_rng(seed)
    coo = m.tocoo()
    r, c, v = coo.row, coo.col, coo.data
    split = rng.random(len(v)) < 0.3
    a = np.where(split, v * 0.25, v)
    rows = np.concatenate([r, r[split], [0, 0, 3, 3]])
    cols = np.concatenate([c, c[split], [m.shape[1] - 1] * 2 + [1] * 2])
    vals = np.concatenate([a, v[split] * 0.75, [1, -1, -2, 2]]).astype(np.float32)
    order = np.lexsort((rng.random(len(rows)), rows))  # rows grouped, columns shuffled
    return rows[order], cols[order], vals[order]


@pytest.mark.parametrize('fmt', ['csr', 'coo'])
@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('block_size', [None, 8])
def test_duplicate_entries_are_summed(fmt, binary, block_size):
    # 0.7.0 passed CSR duplicates to the kernel unsummed: wrong norms and binary counts.
    m = _m(60, 40)
    rows, cols, vals = _with_duplicates(m)
    if fmt == 'coo':
        dup = sp.coo_array((vals, (rows, cols)), shape=m.shape)
    else:
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=m.shape[0]))])
        dup = sp.csr_array((vals, cols, indptr), shape=m.shape)
        assert not dup.has_canonical_format
    clean = sp.csr_array((vals, (rows, cols)), shape=m.shape)  # scipy sums duplicates
    clean.eliminate_zeros()
    before = (dup.data.copy(),)
    kw = {'k': 10, 'binary': binary, 'block_size': block_size, 'verbose': False}
    for f in (sim.cosine, sim.jaccard, sim.dot_product):
        _same(f(dup.T, **kw), f(clean.T, **kw))
        _same(f(m, dup.T.tocsr(), **kw), f(m, clean.T, **kw))
    np.testing.assert_array_equal(dup.data, before[0])  # caller's matrix untouched


def test_coo_output_fed_back(monkeypatch):
    # Feeding an output back (chained calls) must equal feeding its CSR form,
    # and skip the canonicalization: the output is canonical already.
    from similaripy import _core
    m = sp.random_array((200, 60), density=0.01, format='csr', dtype=np.float32, rng=0)
    s = sim.cosine(m.T, k=20, verbose=False)  # few co-occurrences: rows shorter than k
    ref = sim.dot_product(m, s.tocsr(), k=10, verbose=False)

    def fail(*args):
        raise AssertionError('canonicalize_rows called on a canonical input')
    monkeypatch.setattr(_core, 'canonicalize_rows', fail)
    _same(sim.dot_product(m, s, k=10, verbose=False), ref)
    _same(sim.dot_product(m, s.tocsr(), k=10, verbose=False), ref)


def test_canonical_coo_with_zeros():
    # A canonical COO (scipy's sum_duplicates) may keep explicit zeros: dropped as for CSR.
    m = _m(60, 40)
    coo = m.tocoo()
    coo.data[::7] = 0
    coo.sum_duplicates()
    assert coo.has_canonical_format and (coo.data == 0).any()
    clean = coo.tocsr()
    clean.eliminate_zeros()
    kw = {'k': 10, 'binary': True, 'verbose': False}
    _same(sim.cosine(coo, **kw), sim.cosine(clean, **kw))


def _assert_canonical(out):
    """No padding or zeros, rows sorted with unique columns (scipy re-checks a fresh copy)."""
    assert out.has_canonical_format
    assert (out.data != 0).all()
    fresh = sp.csr_array((out.data, out.indices, out.indptr), shape=out.shape)
    assert fresh.has_canonical_format


@pytest.mark.parametrize('order', ['popularity', 'original'])
@pytest.mark.parametrize('block_size', [None, 8])
def test_output_exact_and_canonical(order, block_size, monkeypatch):
    # D37/F5: rows packed (no padding), sorted by column, canonical in both
    # formats; blocked, permuted and unblocked give the same arrays exactly.
    from similaripy import _s_plus
    monkeypatch.setattr(_s_plus, '_COLUMN_ORDER', order)
    m = sp.random_array((300, 60), density=0.03, format='csr', dtype=np.float32, rng=2)
    m = sp.csr_array(m.multiply((np.arange(60) != 5).astype(np.float32)))  # column 5 empty:
    m.eliminate_zeros()                                                     # output row 5 empty
    kw = {'k': 15, 'verbose': False}
    out = sim.cosine(m.T, block_size=block_size, format_output='csr', **kw)
    _assert_canonical(out)
    assert out.indptr[6] == out.indptr[5]
    ref = sim.cosine(m.T, block_size=None, format_output='csr', **kw)
    for a in ('indptr', 'indices', 'data'):
        np.testing.assert_array_equal(getattr(out, a), getattr(ref, a))
    coo = sim.cosine(m.T, block_size=block_size, **kw)  # 'coo' default until commit 7
    assert coo.has_canonical_format
    want = out.tocoo()
    for a in ('row', 'col', 'data'):
        np.testing.assert_array_equal(getattr(coo, a), getattr(want, a))


@pytest.mark.parametrize('fmt', ['csr', 'coo'])
def test_duplicate_and_unsorted_target_rows(fmt):
    # F15: a repeated target gave a row with every entry twice (scipy then summed them).
    m = _m(60, 40)
    kw = {'k': 5, 'format_output': fmt, 'verbose': False}
    out = sim.cosine(m, target_rows=[30, 3, 1, 3, 30], **kw)
    ref = sim.cosine(m, target_rows=[1, 3, 30], **kw)
    assert out.nnz == ref.nnz == 15
    _same(out, ref)
    _assert_canonical(out.tocsr())


def test_k_larger_than_columns():
    m = _m(60, 40)
    _same(sim.cosine(m, k=1000, verbose=False), sim.cosine(m, k=60, verbose=False))  # 60 output cols


_SIMILARITIES = ['dot_product', 'cosine', 'asymmetric_cosine', 'tversky', 'jaccard', 'dice',
                 'p3alpha', 'rp3beta', 's_plus']
_NORMALIZATIONS = [('normalize', {'norm': 'l1'}), ('normalize', {'norm': 'l2'}),
                   ('normalize', {'norm': 'max'}), ('binarize', {}), ('bm25', {}),
                   ('bm25plus', {}), ('tfidf', {})]


def _non_canonical(seed=0):
    """(CSR with duplicates, unsorted columns and explicit zeros, its canonical form)."""
    m = _m(60, 40, seed)
    rows, cols, vals = _with_duplicates(m, seed)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=m.shape[0]))])
    dup = sp.csr_array((vals, cols, indptr), shape=m.shape)
    assert not dup.has_canonical_format
    clean = sp.csr_array((vals, (rows, cols)), shape=m.shape)  # scipy sums duplicates
    clean.eliminate_zeros()
    return dup, clean


@pytest.mark.parametrize('name', _SIMILARITIES)
@pytest.mark.parametrize('binary', [False, True])
def test_similarity_canonicalizes_input(name, binary):
    # R6: p3alpha/rp3beta normalize (and take popularity) before s_plus canonicalizes
    dup, clean = _non_canonical()
    f = getattr(sim, name)
    kw = {'k': 10, 'binary': binary, 'verbose': False}
    _same(f(dup.T, **kw), f(clean.T, **kw))


@pytest.mark.parametrize('name, kw', _NORMALIZATIONS)
@pytest.mark.parametrize('axis', [0, 1])
def test_normalization_canonicalizes_input(name, kw, axis):
    # R6: duplicates gave e.g. l2 row norms of 1.414 and negative bm25/tfidf weights
    from similaripy import normalization
    dup, clean = _non_canonical()
    f = getattr(normalization, name)
    if name != 'binarize':
        kw = {**kw, 'axis': axis}
    before = dup.data.copy()
    _same(f(dup, **kw), f(clean, **kw))
    np.testing.assert_array_equal(dup.data, before)  # not in place by default
    f(dup, inplace=True, **kw)
    if axis == 1 or name == 'binarize':  # a CSR caller's matrix is canonicalized in place
        assert dup.has_canonical_format and (dup.data != 0).all()
