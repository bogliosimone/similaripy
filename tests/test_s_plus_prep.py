"""s_plus prep: input canonicalization, layout, selectors, output build, and the
regressions they fixed."""
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
    # the popularity permutation is the identity: m2 must still be sorted
    # for blocking (regression: the sort used to be skipped).
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


def _input_formats():
    """One square matrix in each input form `_canonical_csr` handles."""
    m = _m(40, 40, seed=6)
    zeros = m.copy()
    zeros.data[::5] = 0
    wide = m.astype(np.float64)
    wide.indices, wide.indptr = wide.indices.astype(np.int64), wide.indptr.astype(np.int64)
    unsorted = m.tocoo()
    order = np.random.default_rng(0).permutation(unsorted.nnz)
    unsorted = sp.coo_array((unsorted.data[order], (unsorted.row[order], unsorted.col[order])), shape=m.shape)
    return {'csr': m, 'csr_zeros': zeros, 'csr_f64_int64': wide, 'csc': m.tocsc(),
            'coo_canonical': m.tocoo(), 'coo_unsorted': unsorted}


def _arrays(m):
    return [m.data, m.row, m.col] if m.format == 'coo' else [m.data, m.indices, m.indptr]


@pytest.mark.parametrize('fmt', list(_input_formats()))
@pytest.mark.parametrize('name', ['dot_product', 'cosine', 'rp3beta'])
def test_inputs_not_mutated_any_format(name, fmt):
    # prep copies each input once, then works in place (`_canonical_csr`): the
    # caller's arrays never change, also as matrix2 (sorted in place) and when
    # the same object is both inputs
    m = _input_formats()[fmt]
    before = [a.copy() for a in _arrays(m)]
    f = getattr(sim, name)
    ref = f(m, m.T.tocsr(), k=5, verbose=False)
    for kw in ({}, {'block_size': 8}):
        f(m, k=5, verbose=False, **kw)
        out = f(m, m, k=5, verbose=False, **kw)  # matrix2 = the caller's own (square) matrix
        _same(out, f(m, m.copy(), k=5, verbose=False, **kw))
    _same(f(m, k=5, verbose=False), ref)
    for a, b in zip(_arrays(m), before):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


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

    def boom(cols, num_threads):
        raise RuntimeError('prep failed')
    monkeypatch.setattr(_s_plus, '_matrix_selector', boom)
    with pytest.raises(RuntimeError, match='prep failed'):
        sim.cosine(_m(), k=5, verbose=True)
    assert capfd.readouterr().err.split('\r')[-1].startswith('Failed:')


@pytest.mark.parametrize('limit, dtype', [(None, np.int32), (100, np.int64)])
def test_csr_index_dtype_past_int32_nnz(limit, dtype, monkeypatch):
    # Past 2^31 results the CSR gets int64 indptr and indices; tested at a
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


def _csr_bytes(m):
    m = m.tocsr()
    return [(a.dtype, a.tobytes()) for a in (m.data, m.indices, m.indptr)]


# without shrink rp3beta takes `_product_form` instead
@pytest.mark.parametrize('kwargs', [{'shrink': 3}, {'shrink': 3, 'shrink_type': 'bayesian'},
                                    {'alpha': 0.7, 'beta': 0.4, 'shrink': 0.5}])
@pytest.mark.parametrize('block_size', [None, 8])
def test_depop_as_cosine_same_bits(kwargs, block_size, monkeypatch):
    # rp3beta (depop only) through the cosine slots: COSINE mode, FULL's bits
    from similaripy import _s_plus
    d = _m(80, 50, seed=3).toarray()
    d[5], d[:, 7] = 0, 0  # an empty row and column
    m = sp.csr_array(d)
    kw = dict(k=20, verbose=False, block_size=block_size, **kwargs)
    swapped = sim.rp3beta(m, **kw)
    monkeypatch.setattr(_s_plus, '_depop_as_cosine', lambda l1, l2, l3, norms: (l2, l3, norms))
    assert _csr_bytes(swapped) == _csr_bytes(sim.rp3beta(m, **kw))


@pytest.mark.parametrize('order', ['popularity', 'original'])
@pytest.mark.parametrize('block_size', [None, 8, 64])
def test_tie_rule_independent_of_blocking(order, block_size, monkeypatch):
    # Binary dot scores are small integers, so ties at the k-th place are
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


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1e39])  # 1e39: finite in f64, inf in f32
@pytest.mark.parametrize('side', ['matrix1', 'matrix2'])
@pytest.mark.parametrize('binary', [False, True])
def test_rejects_non_finite_values(bad, side, binary):
    # Checked on the f32 values in both modes: binary must not turn NaN/inf into 1.0.
    m = _m(60, 40).astype(np.float64)
    bad_m = m.copy()
    bad_m.data[3] = bad
    m1, m2 = (bad_m, m.T) if side == 'matrix1' else (m, bad_m.T)
    with pytest.raises(ValueError, match=f'{side} values must be finite'):
        sim.dot_product(m1, m2, k=5, binary=binary, verbose=False)
    if bad != 1e39:  # the l1 normalization divides 1e39 back into range
        with pytest.raises(ValueError, match='values must be finite'):
            sim.p3alpha(bad_m, k=5, verbose=False)  # checked after the wrapper's normalization


def _signed_m(seed=0):
    m = _m(60, 40, seed)
    m.data[::3] *= -1
    return m


@pytest.mark.parametrize('name, kwargs, match', [
    ('cosine', {'shrink': 5, 'shrink_type': 'bayesian'}, "shrink_type='bayesian'.*not supported on signed"),
    ('dot_product', {'shrink': 5, 'shrink_type': 'bayesian'}, "shrink_type='bayesian'.*not supported on signed"),
    ('s_plus', {'l1': 1, 'l2': 0, 't1': 0.2, 't2': 0.2}, 'denominator <= 0 on some pairs of signed input'),
    ('s_plus', {'alpha': 2}, r'a1 \(alpha\) must be 1 on signed'),
    ('s_plus', {'alpha': 0.5}, r'a1 \(alpha\) must be 1 on signed'),
    ('tversky', {'alpha': 0.2, 'beta': 0.2}, 'denominator <= 0 on some pairs of signed input'),
])
def test_signed_rejections(name, kwargs, match):
    f = getattr(sim, name)
    with pytest.raises(ValueError, match=match):
        f(_signed_m(), k=5, verbose=False, **kwargs)
    # binary input is never signed: the same parameters run, as on the binarized matrix
    m = _signed_m()
    ones = sp.csr_array((np.ones(m.nnz, dtype=np.float32), m.indices, m.indptr), shape=m.shape)
    _same(f(m, k=5, binary=True, verbose=False, **kwargs), f(ones, k=5, verbose=False, **kwargs))


@pytest.mark.parametrize('alpha, beta, data, ok', [
    # weighted non-negative: c = 1 - alpha - beta < 0 needs 4 alpha beta > c^2, c >= 0 is always safe
    (1, 1, 'weighted', True), (0.5, 0.5, 'weighted', True), (1, 0, 'weighted', True),
    (0.5, 0.1, 'weighted', True), (0.2, 0.2, 'weighted', True), (3, 0.05, 'weighted', False),
    # signed: c > 0 needs it too
    (1, 1, 'signed', True), (0.5, 0.5, 'signed', True), (1, 0, 'signed', True),
    (0.5, 0.1, 'signed', True), (0.2, 0.2, 'signed', False), (3, 0.05, 'signed', False),
    # binary: den >= |X & Y|, always safe
    (0.2, 0.2, 'binary', True), (3, 0.05, 'binary', True),
])
def test_tversky_denominator_rule(alpha, beta, data, ok):
    m = _signed_m() if data == 'signed' else _m(60, 40)
    kw = {'alpha': alpha, 'beta': beta, 'k': 5, 'binary': data == 'binary', 'verbose': False}
    if ok:
        sim.tversky(m, **kw)
        sim.s_plus(m, l1=1, l2=0, t1=alpha, t2=beta, k=5, binary=data == 'binary', verbose=False)
    else:
        with pytest.raises(ValueError, match=r't1, t2 \(tversky alpha, beta\)'):
            sim.tversky(m, **kw)


def test_tversky_rule_exempts_one_valued_data():
    # 0/1 data without binary=True: den >= |X & Y| as in binary mode (review finding)
    m = _m(60, 40)
    m.data[:] = 1
    for alpha, beta in ((2, 0), (3, 0.05)):
        _same(sim.tversky(m, alpha=alpha, beta=beta, k=5, verbose=False),
              sim.tversky(m, alpha=alpha, beta=beta, k=5, binary=True, verbose=False))


def test_weighted_tversky_rule_closes_negative_denominator():
    # (3, 0.05) on non-negative data: den = 3 (x - xy) + 0.05 (y - xy) + xy < 0 for a
    # pair with xy > x (it used to be dropped by the threshold); now a ValueError.
    m = sp.csr_array(np.array([[1, 0], [10, 10]], dtype=np.float32))
    x, y, xy = 1.0, 200.0, 10.0
    assert 3 * (x - xy) + 0.05 * (y - xy) + xy < 0
    with pytest.raises(ValueError, match='denominator <= 0'):
        sim.tversky(m, alpha=3, beta=0.05, verbose=False)


def test_signed_depop_sum_uses_abs():
    # pop 'sum' = sum of |x|: on |m| (same pattern) the denominators are the same
    m = _signed_m()
    kw = {'l1': 0, 'l2': 0, 'l3': 1, 'pop1': 'sum', 'pop2': 'sum', 'beta1': 0.5, 'beta2': 0.5,
          'k': 60, 'verbose': False}
    a = abs(m)
    xy, xy_abs = (m @ m.T).toarray(), (a @ a.T).toarray()
    den = np.divide(xy_abs, sim.s_plus(a, **kw).toarray(), out=np.zeros_like(xy_abs),
                    where=xy_abs != 0)  # pop1 pop2 per pair, from the non-negative run
    got = sim.s_plus(m, **kw).toarray()
    expected = np.divide(xy, den, out=np.zeros_like(xy), where=den != 0)
    expected[expected <= 0] = 0
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)


def _cancelling_selector(fmt):
    """60 x 60 selector: (0, 5) as +1/-1 duplicates (cancels), (0, 7) and (1, 3) kept, (2, 9) a stored 0."""
    rows, cols, vals = [0, 0, 0, 1, 2], [5, 7, 5, 3, 9], np.array([1, 1, -1, 1, 0], dtype=np.float32)
    if fmt == 'coo':
        return sp.coo_array((vals, (rows, cols)), shape=(60, 60))
    order = np.argsort(rows, kind='stable')
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=60))])
    return sp.csr_array((vals[order], np.array(cols)[order], indptr), shape=(60, 60))


@pytest.mark.parametrize('fmt', ['csr', 'coo'])
@pytest.mark.parametrize('selector', ['filter_cols', 'target_cols'])
def test_selector_matrix_uses_summed_values(fmt, selector):
    # An entry is selected iff its summed value is non-zero, in every format
    m = _m(60, 40)
    sel = _cancelling_selector(fmt)
    before = sel.copy()
    clean = sp.csr_array((np.ones(2, dtype=np.float32), ([0, 1], [7, 3])), shape=(60, 60))
    kw = {'k': 60, 'verbose': False}
    _same(sim.cosine(m, **{selector: sel}, **kw), sim.cosine(m, **{selector: clean}, **kw))
    np.testing.assert_array_equal(sel.data, before.data)  # caller's matrix untouched


@pytest.mark.parametrize('fmt', ['csr', 'coo'])
def test_selector_matrix_all_cancelled_selects_nothing(fmt):
    m = _m(60, 40)
    sel = _cancelling_selector(fmt)
    sel = sel.tocoo()
    keep = (sel.row == 0) & (sel.col == 5)
    sel = sp.coo_array((sel.data[keep], (sel.row[keep], sel.col[keep])), shape=sel.shape)
    sel = sel if fmt == 'coo' else sp.csr_array((sel.data, sel.col, [0, 2] + [2] * 59), shape=sel.shape)
    kw = {'k': 10, 'verbose': False}
    _same(sim.cosine(m, filter_cols=sel, **kw), sim.cosine(m, **kw))
    assert sim.cosine(m, target_cols=sel, **kw).nnz == 0


_BAD_PARAMS = [
    ('dot_product', {'k': 0}, 'k must be'),
    ('dot_product', {'k': 2.5}, 'k must be'),
    ('dot_product', {'num_threads': -1}, 'num_threads'),
    ('dot_product', {'block_size': 1.5}, 'block_size'),
    ('dot_product', {'threshold': -0.1}, 'threshold'),
    ('dot_product', {'threshold': np.nan}, 'threshold'),
    ('dot_product', {'target_rows': [0, 60]}, 'target_rows ids'),
    ('dot_product', {'target_rows': [0.0, 1.5]}, 'integer ids'),
    ('dot_product', {'filter_cols': [3, 60]}, 'filter_cols ids'),
    ('dot_product', {'target_cols': [-1]}, 'target_cols ids'),
    ('cosine', {'shrink': -1}, 'shrink must be'),
    ('cosine', {'shrink_type': 'other'}, 'shrink_type'),
    ('dot_product', {'shrink': 5, 'shrink_type': 'stabilized'}, 'only rescales'),
    ('p3alpha', {'shrink': 5, 'shrink_type': 'stabilized'}, 'only rescales'),
    ('jaccard', {'shrink': 5, 'shrink_type': 'additive'}, 'cosine term'),
    ('rp3beta', {'shrink': 5, 'shrink_type': 'additive'}, 'cosine term'),
    ('dot_product', {'shrink': 5, 'shrink_type': 'additive'}, 'cosine term'),
    ('tversky', {'shrink': 5, 'shrink_type': 'additive'}, 'cosine term'),
    ('dice', {'shrink': 5, 'shrink_type': 'additive'}, 'cosine term'),
    ('p3alpha', {'shrink': 5, 'shrink_type': 'additive'}, 'cosine term'),
    ('dot_product', {'k': True}, 'k must be'),
    ('dot_product', {'format_output': 'dense'}, 'format_output'),
    ('dot_product', {'filter_cols': {3}}, 'filter_cols must be'),
    ('dot_product', {'target_cols': sp.csr_array(np.eye(5, dtype=np.float32))}, 'target_cols shape'),
    ('dot_product', {'matrix2': np.ones((60, 60))}, 'matrix2 must be a sparse'),
    ('dot_product', {'matrix2': _m(40, 60)}, 'Incompatible matrix shapes'),
    ('asymmetric_cosine', {'alpha': 1.5}, 'alpha must be in'),
    ('tversky', {'alpha': -0.5}, 'tversky alpha'),
    ('p3alpha', {'alpha': 0}, 'alpha must be finite and > 0'),
    ('rp3beta', {'alpha': -1}, 'alpha must be finite and > 0'),
    ('rp3beta', {'beta': -0.5}, 'rp3beta beta'),
    ('s_plus', {'l2': -1}, 'l2'),
    ('s_plus', {'c1': np.inf}, 'c1'),
    ('s_plus', {'beta1': -1, 'l3': 1}, 'beta1'),
    ('s_plus', {'pop2': np.ones(5)}, 'array of length 60'),
    ('s_plus', {'pop2': 'max'}, 'pop2'),
    ('s_plus', {'pop1': np.zeros(60), 'l3': 1}, 'on every row/column with entries'),
    ('s_plus', {'pop2': -np.ones(60), 'l3': 1}, 'finite and >= 0'),
]


@pytest.mark.parametrize('name, kwargs, match', _BAD_PARAMS)
def test_rejects_bad_parameters(name, kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        getattr(sim, name)(_m(60, 60), verbose=False, **kwargs)


def test_rejects_dense_matrix1():
    with pytest.raises(TypeError, match='matrix1 must be a sparse'):
        sim.dot_product(np.ones((5, 5)), _m(5, 5), verbose=False)


def test_zero_weight_on_empty_columns_accepted():
    # pop as interaction counts is 0 for items nobody touched: never scored, so allowed.
    m = _m(60, 60)
    m = sp.csr_array(m.multiply((np.arange(60) != 7).astype(np.float32)))  # column 7 empty
    m.eliminate_zeros()
    counts = np.diff(m.tocsc().indptr).astype(np.float64)
    assert counts[7] == 0
    ones = np.where(counts == 0, 1.0, counts)
    kw = {'l1': 0, 'l2': 0, 'l3': 1, 'beta2': 0.5, 'k': 10, 'verbose': False}
    _same(sim.s_plus(m.T, m, pop2=counts, **kw), sim.s_plus(m.T, m, pop2=ones, **kw))


def test_accepts_numpy_scalars_and_sequences():
    # Validation must not reject numpy types or tuples the old code took.
    m = _m(60, 60)
    ref = sim.s_plus(m, l3=1, pop1=list(np.arange(1, 61)), k=np.int64(5), verbose=False)
    out = sim.s_plus(m, l3=1, pop1=tuple(np.arange(1, 61)), k=5, verbose=np.True_,
                     target_rows=np.arange(60, dtype=np.int64), num_threads=np.int32(0))
    _same(out, ref)


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
    # Regression: CSR duplicates reached the kernel unsummed (wrong norms and binary counts).
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
    s = sim.cosine(m.T, k=20, format_output='coo', verbose=False)  # rows shorter than k
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


@pytest.mark.parametrize('fmt', ['csr', 'coo', 'coo_unsorted'])
def test_float64_underflow_dropped(fmt):
    # Regression: canonical input dropped zeros before the float32 cast, so a
    # float64 value below float32's range stayed as an explicit 0.0 (binary: 1).
    d = np.array([[1e-50, 1.0], [1.0, 1.0]])
    m = sp.csr_array(d) if fmt == 'csr' else sp.coo_array(d)
    if fmt == 'coo_unsorted':
        m = sp.coo_array((m.data[::-1], (m.row[::-1], m.col[::-1])), shape=m.shape)
    out = sim.dot_product(m, binary=True, verbose=False)
    np.testing.assert_array_equal(out.toarray(), [[1, 1], [1, 2]])


def _assert_canonical(out):
    """No padding or zeros, rows sorted with unique columns (scipy re-checks a fresh copy)."""
    assert out.has_canonical_format
    assert (out.data != 0).all()
    fresh = sp.csr_array((out.data, out.indices, out.indptr), shape=out.shape)
    assert fresh.has_canonical_format


@pytest.mark.parametrize('order', ['popularity', 'original'])
@pytest.mark.parametrize('block_size', [None, 8])
def test_output_exact_and_canonical(order, block_size, monkeypatch):
    # Rows packed (no padding), sorted by column, canonical in both
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
    coo = sim.cosine(m.T, block_size=block_size, format_output='coo', **kw)
    assert coo.has_canonical_format
    want = out.tocoo()
    for a in ('row', 'col', 'data'):
        np.testing.assert_array_equal(getattr(coo, a), getattr(want, a))


@pytest.mark.parametrize('fmt', ['csr', 'coo'])
def test_duplicate_and_unsorted_target_rows(fmt):
    # Regression: a repeated target gave a row with every entry twice (scipy then summed them).
    m = _m(60, 40)
    kw = {'k': 5, 'format_output': fmt, 'verbose': False}
    out = sim.cosine(m, target_rows=[30, 3, 1, 3, 30], **kw)
    ref = sim.cosine(m, target_rows=[1, 3, 30], **kw)
    assert out.nnz == ref.nnz == 15
    _same(out, ref)
    _assert_canonical(out.tocsr())


@pytest.mark.parametrize('fmt', ['csr', 'coo'])
def test_underflowed_product_dropped(fmt):
    # 1e-30 * 1e-30 underflows to 0.0 in f32: the column was pushed again on its next
    # hit and came back twice, once as an explicit 0.0, in an output flagged canonical.
    m1 = sp.csr_array(np.array([[1e-30, 1.0]], dtype=np.float32))
    m2 = sp.csr_array(np.array([[1e-30, 0, 0], [1.0, 0, 0]], dtype=np.float32))
    out = sim.s_plus(m1, m2, k=5, format_output=fmt, verbose=False).tocsr()
    np.testing.assert_array_equal(out.indices, [0])
    np.testing.assert_array_equal(out.data, [1.0])
    _assert_canonical(out)
    # an underflow alone leaves the row empty
    assert sim.dot_product(sp.csr_array(np.array([[1e-30], [1e-30]], dtype=np.float32)), verbose=False).nnz == 0


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
    # p3alpha/rp3beta normalize (and take popularity) before s_plus canonicalizes
    dup, clean = _non_canonical()
    f = getattr(sim, name)
    kw = {'k': 10, 'binary': binary, 'verbose': False}
    _same(f(dup.T, **kw), f(clean.T, **kw))


@pytest.mark.parametrize('name, kw', _NORMALIZATIONS)
@pytest.mark.parametrize('axis', [0, 1])
def test_normalization_canonicalizes_input(name, kw, axis):
    # Regression: duplicates gave e.g. l2 row norms of 1.414 and negative bm25/tfidf weights
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


@pytest.mark.parametrize('args, expected', [
    ((1, 0, 1, 0, 0, 0), True),     # cosine, asymmetric cosine (additive shrink inside the norms)
    ((1, 0, 0, 1, 0, 0), True),     # rp3beta
    ((0.9, 0, 1, 0, 0, 0), False),  # s_plus alpha: (x s)^a1 != x^a1 s
    ((1, 1, 1, 0, 0, 0), False),    # tversky term
    ((1, 0, 1, 1, 0, 0), False),    # cosine + depop
    ((1, 0, 0, 0, 0, 0), False),    # dot product: nothing to scale
    ((1, 0, 1, 0, 2, 0), False),    # stabilized shrink
    ((1, 0, 0, 1, 0, 2), False),    # bayesian shrink
])
def test_product_form(args, expected):
    from similaripy import _s_plus
    assert _s_plus._product_form(*args) is expected


def test_prescale_in_place_zero_norm_gives_zero():
    from similaripy import _s_plus
    m = _m(4, 3, seed=1)
    d1, d2 = m.data.copy(), m.data.copy()
    X, Y = np.array([2.0, 0.0, 4.0, 1.0]), np.array([0.0, 0.5, 1.0])
    norms = _s_plus.Norms(x_cosine=X, y_cosine=Y)
    assert _s_plus._prescale((d1, m.indices, m.indptr), (d2, m.indices, m.indptr), 2, norms)
    rows = np.repeat(np.arange(4), np.diff(m.indptr))
    np.testing.assert_allclose(d1, m.data / np.where(X == 0, np.inf, 2 * X)[rows], rtol=1e-6)
    np.testing.assert_allclose(d2, m.data / np.where(Y == 0, np.inf, Y)[m.indices], rtol=1e-6)
    assert not d1[rows == 1].any() and not d2[m.indices == 0].any()


def test_prescale_overflow_changes_nothing():
    # a product past float32's max: refused before touching the values
    from similaripy import _s_plus
    m = _m(4, 3, seed=1)
    m.data[0] = 1e30
    d1, d2 = m.data.copy(), m.data.copy()
    norms = _s_plus.Norms(x_cosine=np.ones(4), y_cosine=np.full(3, 1e-10))  # 1 / Y = 1e10: 1e30 * 1e10 overflows
    assert not _s_plus._prescale((d1, m.indices, m.indptr), (d2, m.indices, m.indptr), 1, norms)
    assert d1.tobytes() == m.data.tobytes() and d2.tobytes() == m.data.tobytes()


def test_prescale_overflow_falls_back(monkeypatch):
    # pop ~ 1e-30, beta 2: 1 / pop^2 overflows float32 -> the kernel path, same bits
    from similaripy import _s_plus
    m = _m(60, 40, seed=2) * np.float32(1e-30)
    kw = dict(k=10, beta=2.0, verbose=False)
    prescale, results = _s_plus._prescale, []
    monkeypatch.setattr(_s_plus, '_prescale', lambda *a: results.append(prescale(*a)) or results[-1])
    out = sim.rp3beta(m, **kw)
    assert results == [False]  # the fallback was taken
    monkeypatch.setattr(_s_plus, '_product_form', lambda *a: False)
    assert _csr_bytes(out) == _csr_bytes(sim.rp3beta(m, **kw))


@pytest.mark.parametrize('scale', [1e19, 1e-25])  # float32 norms overflow / underflow
@pytest.mark.parametrize('name', ['cosine', 'asymmetric_cosine'])
def test_product_form_extreme_values(name, scale):
    # float64 scales: real scores where float32 norms would give none (accepted, D51)
    m = _m(60, 40, seed=4)
    _same(getattr(sim, name)(m * np.float32(scale), k=10, verbose=False),
          getattr(sim, name)(m, k=10, verbose=False))


def test_signed_power_alpha_one_same_bits():
    # p3alpha / rp3beta skip the power at alpha == 1: same bits as computing it
    from similaripy.similarity import _signed_power
    x = np.array([0.3, -0.0, 0.0, -2.5, 1e-40, 3e38, -1e-45], dtype=np.float32)
    assert _signed_power(x, 1).tobytes() == np.copysign(np.abs(x) ** 1.0, x).tobytes()
