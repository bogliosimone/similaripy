import similaripy as sim
import pytest
import numpy as np
import scipy.sparse as sp


def generate_random_matrix(n_rows=100, n_cols=50, density=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return sp.random_array((n_rows, n_cols), density=density, format='csr', dtype=np.float32, random_state=rng)


# ---- Dense float64 reference: every pair's score from the formulas ----

def _dense_random_walk(A, alpha):
    """P3alpha walk scores: l1 rows by sum |x|, sign-preserving power, both sides."""
    def walk(M):
        M = M / np.maximum(np.abs(M).sum(axis=1, keepdims=True), 1e-300)
        return np.sign(M) * np.abs(M) ** alpha
    return walk(A) @ walk(A.T)


def _dense_scores(m, name, kw):
    """Scores of `getattr(sim, name)(m, **kw)` for every pair (m @ m.T, rows of m), in float64."""
    A = m.toarray().astype(np.float64)
    h = kw.get('shrink', 0.0)
    shrink_type = kw.get('shrink_type', 'stabilized')
    stabilized, bayesian, additive = (h if shrink_type == t else 0.0
                                      for t in ('stabilized', 'bayesian', 'additive'))
    sq = (A ** 2).sum(axis=1)
    pop = np.abs(A).sum(axis=1)
    if name in ('p3alpha', 'rp3beta'):
        xy = _dense_random_walk(A, kw.get('alpha', 1.0))
        den = pop[None, :] ** kw.get('beta', 0.5) if name == 'rp3beta' else None
    else:
        xy = A @ A.T
        if name == 'dot_product':
            den = None
        elif name in ('cosine', 'asymmetric_cosine'):
            c1 = kw.get('alpha', 0.5)
            den = np.outer((sq + additive) ** c1, (sq + additive) ** (1 - c1))
        elif name == 's_plus':
            xy = xy ** kw.get('alpha', 1.0)
            l1, l2, l3 = kw.get('l1', 0.5), kw.get('l2', 0.5), kw.get('l3', 0.0)
            raw = A @ A.T
            p1 = pop ** kw.get('beta1', 0.0) if kw.get('pop1') == 'sum' else np.ones_like(pop)
            p2 = pop ** kw.get('beta2', 0.0) if kw.get('pop2') == 'sum' else np.ones_like(pop)
            den = (l1 * (kw.get('t1', 1.0) * (sq[:, None] - raw) + kw.get('t2', 1.0) * (sq[None, :] - raw) + raw)
                   + l2 * np.outer((sq + additive) ** kw.get('c1', 0.5), (sq + additive) ** kw.get('c2', 0.5))
                   + l3 * np.outer(p1, p2))
            if l1 == l2 == l3 == 0:
                den = None
        else:  # tversky family
            t = {'jaccard': (1, 1), 'dice': (0.5, 0.5)}.get(name, (kw.get('alpha'), kw.get('beta')))
            den = t[0] * (sq[:, None] - xy) + t[1] * (sq[None, :] - xy) + xy
    if den is not None:
        xy_pow = xy
        xy = np.divide(xy, den + stabilized, out=np.zeros_like(xy), where=xy != 0)
    else:
        xy_pow = xy
    if bayesian:
        xy = xy * np.divide(xy_pow, xy_pow + bayesian, out=np.zeros_like(xy), where=xy_pow != 0)
    return xy


def _check_scores(out, scores, k, threshold=0.0):
    """Each row of `out` holds the min(k, #candidates) best positive scores >= threshold,
    at the right columns: compares values, so any choice among tied columns passes."""
    out = out.tocsr()
    for i in range(scores.shape[0]):
        cols = out.indices[out.indptr[i]:out.indptr[i + 1]]
        vals = out.data[out.indptr[i]:out.indptr[i + 1]]
        assert len(np.unique(cols)) == len(cols), i
        assert (vals > 0).all() and (vals >= threshold).all(), i  # no 0.0, no negatives
        best = np.sort(scores[i][(scores[i] > 0) & (scores[i] >= threshold)])[::-1]
        assert len(vals) == min(k, len(best)), i
        np.testing.assert_allclose(vals, scores[i, cols], rtol=1e-4, err_msg=str(i))
        np.testing.assert_allclose(np.sort(vals)[::-1], best[:len(vals)], rtol=1e-4, err_msg=str(i))


# ---- Every similarity x shrink type vs the dense reference ----

_CASES = [
    ('dot_product', {}),
    ('cosine', {}),
    ('asymmetric_cosine', {'alpha': 0.2}),
    ('jaccard', {}),
    ('dice', {}),
    ('tversky', {'alpha': 0.8, 'beta': 0.4}),
    ('p3alpha', {'alpha': 0.8}),
    ('rp3beta', {'alpha': 0.8, 'beta': 0.4}),
    ('s_plus', {'l1': 0.5, 'l2': 0.5, 'l3': 1, 't1': 1, 't2': 1, 'c1': 0.3, 'c2': 0.6,
                'pop2': 'sum', 'beta2': 0.4, 'alpha': 1.5}),
]

# shrink types that do nothing for a similarity are rejected (tested in test_s_plus_prep)
_NO_COSINE = {'dot_product', 'tversky', 'jaccard', 'dice', 'p3alpha', 'rp3beta'}
_NO_DENOMINATOR = {'dot_product', 'p3alpha'}


@pytest.mark.parametrize('name, kw', _CASES, ids=[c[0] for c in _CASES])
@pytest.mark.parametrize('shrink_type', [None, 'stabilized', 'bayesian', 'additive'])
@pytest.mark.parametrize('k', [10, 400])
@pytest.mark.parametrize('block_size', [None, 64])
def test_similarity_matches_dense_reference(name, kw, shrink_type, k, block_size):
    # Regression (bayesian): without a normalization term it returned an empty matrix
    if (shrink_type == 'additive' and name in _NO_COSINE) or \
            (shrink_type == 'stabilized' and name in _NO_DENOMINATOR):
        pytest.skip('rejected: the shrink would do nothing')
    if shrink_type is not None:
        kw = {**kw, 'shrink': 2.0, 'shrink_type': shrink_type}
    m = generate_random_matrix(400, 300, density=0.03)
    out = getattr(sim, name)(m, k=k, block_size=block_size, verbose=False, **kw)
    _check_scores(out, _dense_scores(m, name, kw), k)


def test_output_format():
    m = generate_random_matrix(1000, 800, density=0.025)
    sim_csr = sim.cosine(m, format_output='csr', k=50, verbose=False)
    sim_coo = sim.cosine(m, format_output='coo', k=50, verbose=False)
    assert isinstance(sim_csr, sp.csr_array) and sim_csr.nnz > 0
    assert isinstance(sim_coo, sp.coo_array) and sim_coo.nnz > 0


def test_example_code():
    # The README example: item-item cosine on a BM25-weighted URM, then
    # recommendations for 3 users without the items they have seen
    urm = sp.random_array((1000, 2000), density=0.025, rng=0)
    urm = sim.normalization.bm25(urm)
    similarity_matrix = sim.cosine(urm.T, k=50)
    recommendations = sim.dot_product(urm, similarity_matrix.T, k=100, target_rows=[1, 14, 8],
                                      filter_cols=urm).tocsr()
    assert similarity_matrix.shape == (2000, 2000)
    assert set(np.flatnonzero(np.diff(recommendations.indptr))) == {1, 8, 14}
    assert recommendations.multiply(urm.tocsr() != 0).nnz == 0


# ---- Row and column selectors vs the dense reference ----

def _selector_case():
    m = generate_random_matrix(400, 300, density=0.03)
    ids = np.random.default_rng(42).choice(400, size=60, replace=False)
    return m, ids, _dense_scores(m, 'cosine', {})


def test_target_rows():
    m, rows, scores = _selector_case()
    out = sim.cosine(m, k=10, target_rows=rows.tolist(), verbose=False).tocsr()
    others = np.setdiff1d(np.arange(400), rows)
    assert np.diff(out.indptr)[others].sum() == 0
    _check_scores(out[np.sort(rows)], scores[np.sort(rows)], 10)


@pytest.mark.parametrize('selector', ['filter_cols', 'target_cols'])
def test_column_selector_array(selector):
    m, cols, scores = _selector_case()
    keep = np.zeros(400, dtype=bool)
    keep[cols] = True
    if selector == 'filter_cols':
        keep = ~keep
    out = sim.cosine(m, k=10, verbose=False, **{selector: cols.tolist()})
    _check_scores(out, np.where(keep[None, :], scores, 0.0), 10)


def test_filter_cols_matrix():
    # Real-world case: recommendations (URM @ item similarity) without the items each user has seen
    rng = np.random.default_rng(42)
    urm = sp.random_array((100, 200), density=0.05, format='csr', dtype=np.float32, random_state=rng)
    item_similarity = sp.random_array((200, 200), density=1, format='csr', dtype=np.float32, random_state=rng)
    out = sim.dot_product(urm, item_similarity, k=50, filter_cols=urm, verbose=False)
    scores = urm.toarray().astype(np.float64) @ item_similarity.toarray().astype(np.float64)
    scores[urm.toarray() != 0] = 0.0
    _check_scores(out, scores, 50)


def test_s_plus_alpha_without_normalization():
    # Regression: alpha was ignored when no normalization term was active
    m = generate_random_matrix(80, 30, density=0.1).tocsr()
    xy = (m @ m.T).toarray()
    out = sim.s_plus(m, l1=0, l2=0, l3=0, alpha=2, k=m.shape[0], verbose=False)
    np.testing.assert_allclose(out.toarray(), xy ** 2, rtol=1e-5)


@pytest.mark.parametrize('alpha', [0, -1, float('nan')])
def test_s_plus_rejects_non_positive_alpha(alpha):
    with pytest.raises(ValueError, match='alpha'):
        sim.s_plus(generate_random_matrix(), alpha=alpha, verbose=False)

@pytest.mark.parametrize('name', ['p3alpha', 'rp3beta'])
def test_binary_keeps_random_walk_normalization(name):
    # Regression: binary=True was applied after the l1 normalization and erased it
    from similaripy.normalization import binarize
    m = generate_random_matrix(80, 30, density=0.1).tocsr()
    f = getattr(sim, name)
    out = f(m, k=10, binary=True, verbose=False)
    np.testing.assert_allclose(out.toarray(), f(binarize(m), k=10, verbose=False).toarray(), rtol=1e-6)
    assert not np.allclose(out.toarray(), sim.dot_product(m, k=10, binary=True, verbose=False).toarray())


# ---- Signed input: only positive scores >= threshold are results ----

def _drop_cancellation_noise(out, scores, eps=1e-6):
    """
    Exactly cancelled pairs (reference 0) may come back as float noise (~1e-8):
    cosine and rp3beta run as a dot product of pre-scaled inputs (`_prescale`),
    whose rounding doesn't cancel exactly. Drop results below eps whose
    reference is ~0 too; any other result is left to `_check_scores`.
    """
    out = out.tocsr(copy=True)
    rows = np.repeat(np.arange(out.shape[0]), np.diff(out.indptr))
    noise = (out.data < eps) & (np.abs(scores[rows, out.indices]) < eps)
    out.data[noise] = 0
    out.eliminate_zeros()
    return out


def _signed_matrix(seed=0, integer=True):
    """60 x 40, ~15% dense, values in -3..3 (exact f32 dot products) or standard normal."""
    rng = np.random.default_rng(seed)
    m = sp.random_array((60, 40), density=0.15, format='csr', dtype=np.float32, rng=rng)
    m.data = (rng.integers(-3, 4, m.nnz) if integer else rng.standard_normal(m.nnz)).astype(np.float32)
    m.eliminate_zeros()
    return m


_SIGNED_CASES = [
    ('dot_product', {}),
    ('cosine', {}),
    ('cosine', {'shrink': 2}),
    ('cosine', {'shrink': 2, 'shrink_type': 'additive'}),
    ('asymmetric_cosine', {'alpha': 0.2}),
    ('jaccard', {}),
    ('dice', {}),
    ('tversky', {'alpha': 0.3, 'beta': 0.7}),
    ('tversky', {'alpha': 0.8, 'beta': 0.6}),
    ('s_plus', {'l1': 0.5, 'l2': 0.5, 'l3': 1, 't1': 1, 't2': 1, 'pop1': 'sum', 'pop2': 'sum',
                'beta1': 0.3, 'beta2': 0.4}),
]


@pytest.mark.parametrize('name, kw', _SIGNED_CASES)
@pytest.mark.parametrize('block_size', [None, 8])
@pytest.mark.parametrize('k, threshold', [(5, 0.0), (60, 0.0), (60, 0.25)])  # 0.25: exact in f32 and f64
def test_signed_input_matches_dense_reference(name, kw, block_size, k, threshold):
    m = _signed_matrix()
    scores = _dense_scores(m, name, kw)
    A = m.toarray()
    assert (scores < 0).any() and ((np.abs(A) @ np.abs(A).T > 0) & (A @ A.T == 0)).any()  # cancelled pairs
    out = getattr(sim, name)(m, k=k, threshold=threshold, block_size=block_size, verbose=False, **kw)
    if name in ('cosine', 'asymmetric_cosine') and (not kw.get('shrink') or kw.get('shrink_type') == 'additive'):
        out = _drop_cancellation_noise(out, scores)  # product form only (`_product_form`)
    _check_scores(out, scores, k, threshold)


@pytest.mark.parametrize('name, kw', _SIGNED_CASES)
def test_signed_float_input_matches_dense_reference(name, kw):
    m = _signed_matrix(seed=1, integer=False)
    out = getattr(sim, name)(m, k=5, verbose=False, **kw)
    _check_scores(out, _dense_scores(m, name, kw), 5, 0.0)


@pytest.mark.parametrize('name, beta', [('p3alpha', None), ('rp3beta', 0.6)])
@pytest.mark.parametrize('block_size', [None, 8])
@pytest.mark.parametrize('k', [5, 60])
@pytest.mark.parametrize('integer', [True, False])
def test_signed_random_walk_matches_dense_reference(name, beta, block_size, k, integer):
    # threshold 1e-4: well below the scores (~1e-2), above f32 residues of cancelled sums
    m = _signed_matrix(integer=integer)
    kw = {'alpha': 0.8} if beta is None else {'alpha': 0.8, 'beta': beta}
    scores = _dense_scores(m, name, kw)
    assert (scores < -1e-4).any()
    out = getattr(sim, name)(m, k=k, threshold=1e-4, block_size=block_size,
                             verbose=False, **kw)
    _check_scores(out, scores, k, 1e-4)


@pytest.mark.parametrize('fmt', ['coo', 'csr'])
def test_rp3beta_popularity_on_summed_duplicates(fmt):
    # (3, -1) duplicates are one entry of 2 (pop 2), not |3| + |-1| = 4
    m = _signed_matrix()
    coo = m.tocoo()
    i = np.flatnonzero(coo.data > 0)[0]
    rows = np.append(coo.row, coo.row[i])
    cols = np.append(coo.col, coo.col[i])
    vals = np.append(coo.data, np.float32(-1))
    vals[i] += 1  # the pair sums to the original value
    if fmt == 'coo':
        dup = sp.coo_array((vals, (rows, cols)), shape=m.shape)
    else:  # non-canonical CSR: rows grouped, the duplicate last in its row
        order = np.argsort(rows, kind='stable')
        rows, cols, vals = rows[order], cols[order], vals[order]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=m.shape[0]))])
        dup = sp.csr_array((vals, cols, indptr), shape=m.shape)
    before = dup.data.copy()  # dup shares vals' buffer
    np.testing.assert_array_equal(sim.rp3beta(dup, beta=0.6, verbose=False).toarray(),
                                  sim.rp3beta(m, beta=0.6, verbose=False).toarray())
    np.testing.assert_array_equal(dup.data, before)  # caller's matrix untouched


def test_signed_row_with_only_negative_scores_is_empty():
    m = sp.csr_array(np.array([[1, -1], [-1, 1], [1, 1]], dtype=np.float32))
    out = sim.dot_product(m, -m.T, k=3, verbose=False)
    # row 0 . -row 0 = -2, row 0 . -row 1 = 2, row 0 . -row 2 = 0 (cancelled): only column 1
    np.testing.assert_array_equal(out.toarray(), [[0, 2, 0], [2, 0, 0], [0, 0, 0]])

