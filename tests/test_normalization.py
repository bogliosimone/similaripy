import numpy as np
import pytest
import scipy.sparse as sp
from _helpers import assert_same_csr, random_csr

import similaripy.normalization as norm


def _signed(seed=0):
    X = random_csr(seed=seed)
    X.data[::3] *= -1
    return X


@pytest.mark.parametrize("signed", [False, True])
def test_binarize(signed):
    X = _signed() if signed else random_csr()
    X.data[::5] = 0  # explicit zeros must go, not become 1
    before = X.data.copy()
    B = norm.binarize(X)

    expected = (X.toarray() != 0).astype(np.float32)
    np.testing.assert_array_equal(B.toarray(), expected)
    assert B.nnz == np.count_nonzero(before) and B.data.dtype == np.float32
    np.testing.assert_array_equal(X.data, before)  # never in place
    # same scores as binary=True
    import similaripy as sim

    np.testing.assert_allclose(
        sim.jaccard(B, k=10, verbose=False).toarray(),
        sim.jaccard(X, k=10, binary=True, verbose=False).toarray(),
        rtol=1e-6,
    )


def _tf_idf_reference(X, tf_mode, idf_mode, logbase):
    """TF, IDF and document lengths as the core computes them (`tf` / `idf` in normalize.rs), dense float64."""
    A = X.toarray().astype(np.float64)
    doc_len = A.sum(axis=1, keepdims=True)
    tf = {
        "binary": (A != 0).astype(np.float64),
        "raw": A,
        "sqrt": np.sqrt(A),
        "freq": A / np.where(doc_len == 0, 1, doc_len),
        "log": np.log1p(A) / np.log(logbase),
    }[tf_mode]
    n, df = A.shape[0], (A > 0).sum(axis=0)
    with np.errstate(divide="ignore"):
        idf = {
            "unary": np.ones(A.shape[1]),
            "base": np.log(n / df),
            "smooth": np.log(n / (1 + df)),
            "prob": np.where(df == n, 0.0, np.log((n - df) / df)),  # a term in every document weighs 0
            "bm25": np.log1p((n - df + 0.5) / (df + 0.5)),
            "bm25_robertson": np.log((n - df + 0.5) / (df + 0.5)),
        }[idf_mode] / (1 if idf_mode == "unary" else np.log(logbase))
    return tf, np.where(df == 0, 0.0, idf), doc_len


def _docs(X, axis):
    """X with the documents as rows: the dense references work on rows (axis 0: X.T,
    their result transposed back by `_back`)."""
    return X if axis == 1 else X.T


def _back(A, axis):
    return A if axis == 1 else A.T


@pytest.mark.parametrize("axis", [1, 0])
@pytest.mark.parametrize("tf_mode", ["binary", "raw", "sqrt", "freq", "log"])
@pytest.mark.parametrize("idf_mode", ["unary", "base", "smooth", "prob", "bm25", "bm25_robertson"])
def test_tfidf_matches_formulas(tf_mode, idf_mode, axis):
    X = random_csr(n_rows=200, n_cols=100, density=0.05)
    X.data *= 5
    tf, idf, _ = _tf_idf_reference(_docs(X, axis), tf_mode, idf_mode, 2.0)
    out = norm.tfidf(X, axis=axis, tf_mode=tf_mode, idf_mode=idf_mode, logbase=2.0)
    np.testing.assert_allclose(out.toarray(), _back(tf * idf, axis), rtol=1e-6, atol=1e-7)


def _bm25_tf(A, doc_len, k1, b):
    """BM25's TF part on raw counts, dense float64: tf / (tf + k1 (1 - b + b len / avg_len))."""
    return A / (A + k1 * (1 - b + b * doc_len / doc_len.mean()))


@pytest.mark.parametrize("axis", [1, 0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("delta", [0.0, 0.5])
@pytest.mark.parametrize("k1, b", [(1.2, 0.75), (2.0, 0.3)])
def test_bm25_matches_formula(k1, b, delta, dtype, axis):
    X = random_csr(n_rows=200, n_cols=100, density=0.05).astype(dtype)
    X.data *= 5
    D = _docs(X, axis)
    A = D.toarray().astype(np.float64)
    _, idf, doc_len = _tf_idf_reference(D, "raw", "bm25", np.e)
    kw = {"axis": axis, "k1": k1, "b": b}
    out = norm.bm25plus(X, delta=delta, **kw) if delta else norm.bm25(X, **kw)
    want = np.where(A > 0, (_bm25_tf(A, doc_len, k1, b) + delta) * idf, 0.0)
    np.testing.assert_allclose(out.toarray(), _back(want, axis), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_bm25_robertson_gives_v070_values(delta):
    # the migration recipe: 0.7.0 = (k1 + 1) x bm25_robertson with delta / (k1 + 1)
    X = random_csr(n_rows=200, n_cols=100, density=0.55)  # some terms in > half the documents
    X.data *= 5
    A, k1, b = X.toarray().astype(np.float64), 1.2, 0.75
    _, idf, doc_len = _tf_idf_reference(X, "raw", "bm25_robertson", np.e)
    assert (idf < 0).any()
    v070 = np.where(A > 0, ((k1 + 1) * _bm25_tf(A, doc_len, k1, b) + delta) * idf, 0.0)
    out = norm.bm25plus(X, k1=k1, b=b, delta=delta / (k1 + 1), idf_mode="bm25_robertson")
    np.testing.assert_allclose(out.toarray() * (k1 + 1), v070, rtol=1e-5, atol=1e-6)


def test_bm25_weights_positive_for_common_terms():
    # Regression: Robertson's IDF (0.7.0's default) is negative for a term in more than
    # half the documents, so cosine on two such rows multiplied two negatives
    X = sp.csr_array(np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=np.float32))  # term 0 in every document
    for f in (norm.bm25, norm.bm25plus):
        assert (f(X).data > 0).all()
    assert (norm.bm25(X, idf_mode="bm25_robertson")[:, [0]].data < 0).all()


def test_bm25plus_delta_adds_delta_idf():
    # bm25plus = bm25 + delta * idf on every stored entry
    X = random_csr(n_rows=200, n_cols=100, density=0.05)
    _, idf, _ = _tf_idf_reference(X, "raw", "bm25", np.e)
    plus, plain = norm.bm25plus(X, delta=0.5), norm.bm25(X)
    assert np.array_equal(plus.indices, plain.indices)
    np.testing.assert_allclose(plus.data - plain.data, 0.5 * idf[plus.indices], rtol=1e-4, atol=1e-6)


# ---- Signed input ----


@pytest.mark.parametrize("axis", [1, 0])
@pytest.mark.parametrize("signed", [False, True])
def test_normalize(signed, axis):
    X = _signed() if signed else random_csr()
    A = _docs(X, axis).toarray().astype(np.float64)
    for name, d in (("l1", np.abs(A).sum(axis=1)), ("l2", np.sqrt((A**2).sum(axis=1))), ("max", np.abs(A).max(axis=1))):
        expected = np.divide(A, d[:, None], out=np.zeros_like(A), where=d[:, None] != 0)
        out = norm.normalize(X, norm=name, axis=axis).toarray()
        np.testing.assert_allclose(out, _back(expected, axis), rtol=1e-6, err_msg=name)
    # all-negative row: max divides by its largest |x| (before: left unchanged)
    neg = sp.csr_array(np.array([[-2.0, -4.0]], dtype=np.float32))
    np.testing.assert_array_equal(norm.normalize(neg, norm="max").toarray(), [[-0.5, -1.0]])


@pytest.mark.parametrize("axis", [1, 0])
def test_sums_in_float64(axis):
    # one 1e8 and 1000 ones: a float32 sum stays 1e8 (its ulp there is 8), so the
    # l1-normalized vector would sum to 1 + 1e-5; the float64 sum gives 1 to float32 rounding
    v = np.r_[1e8, np.ones(1000)].astype(np.float32)
    X = sp.csr_array(v[None, :]) if axis == 1 else sp.csr_array(v[:, None])
    out = norm.normalize(X, norm="l1", axis=axis)
    assert abs(out.data.astype(np.float64).sum() - 1) < 1e-6


def test_term_weightings_reject_negatives():
    import pytest

    X = _signed()
    for f in (norm.tfidf, norm.bm25, norm.bm25plus):
        with pytest.raises(ValueError, match=f"{f.__name__} needs non-negative values"):
            f(X)
    # a +1/-1 duplicate pair sums to 0 (dropped), not a negative value
    dup = sp.coo_array((np.array([1, -1, 2], dtype=np.float32), ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    for f in (norm.tfidf, norm.bm25, norm.bm25plus):
        assert f(dup).nnz == 1


@pytest.mark.parametrize("name", ["normalize", "binarize", "tfidf", "bm25", "bm25plus"])
def test_output_dtypes(name):
    # float32 values and int32 indices for any input, as the similarities (one
    # cleaner, `_canonical_csr`); counts are exact in every dtype: same values
    X = random_csr()
    X.data = np.ceil(X.data * 9)
    X.indices, X.indptr = X.indices.astype(np.int64), X.indptr.astype(np.int64)
    f = getattr(norm, name)
    for kw in ({},) if name == "binarize" else ({"axis": 0}, {"axis": 1}):
        want = f(X, **kw)
        for dtype in (np.float32, np.float64, np.int64):
            out = f(X.astype(dtype), **kw)
            assert out.data.dtype == np.float32 and out.indices.dtype == out.indptr.dtype == np.int32
            assert_same_csr(out, want)


@pytest.mark.parametrize("name", ["normalize", "binarize", "tfidf", "bm25", "bm25plus"])
def test_int64_indptr_path_same_values(name, monkeypatch):
    # past 2^31 - 1 entries the output's indptr is int64 (and its indices, scipy's
    # rule; the canonical copy is int64 always); forced with a limit of 0
    from similaripy import _s_plus

    X = random_csr()
    f = getattr(norm, name)
    for kw in ({},) if name == "binarize" else ({"axis": 0}, {"axis": 1}):
        ref = f(X, **kw)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_s_plus, "_MAX_INT32_NNZ", 0)
            out = f(X, **kw)
        assert ref.indptr.dtype == np.int32 and out.indptr.dtype == out.indices.dtype == np.int64
        assert_same_csr(out, ref)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("name", ["normalize", "binarize", "tfidf", "bm25", "bm25plus"])
def test_rejects_non_finite(name, bad):
    # NaN/inf would spread through the row's norm or idf; binarize checks before writing 1s
    X = random_csr()
    X.data[3] = bad
    with pytest.raises(ValueError, match="finite"):
        getattr(norm, name)(X)


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda X: norm.normalize(X, axis=2), ValueError, "axis must be 0 or 1"),
        (lambda X: norm.normalize(X.toarray()), TypeError, "X must be a sparse matrix"),
        (lambda X: norm.normalize(X, norm="l3"), ValueError, "norm must be one of"),
        (lambda X: norm.tfidf(X, tf_mode="x"), ValueError, "tf_mode must be one of"),
        (lambda X: norm.bm25(X, idf_mode="x"), ValueError, "idf_mode must be one of"),
        # weighting parameters that would give negative or inf weights silently
        (lambda X: norm.tfidf(X, logbase=1), ValueError, "logbase must be finite and > 1"),
        (lambda X: norm.bm25(X, logbase=0.5), ValueError, "logbase must be finite and > 1"),
        (lambda X: norm.bm25plus(X, logbase=np.nan), ValueError, "logbase must be finite and > 1"),
        (lambda X: norm.bm25(X, k1=-2), ValueError, "k1 must be finite and >= 0"),
        (lambda X: norm.bm25plus(X, k1=np.inf), ValueError, "k1 must be finite and >= 0"),
        (lambda X: norm.bm25(X, b=5), ValueError, r"b must be in \[0, 1\]"),
        (lambda X: norm.bm25plus(X, b=np.nan), ValueError, r"b must be in \[0, 1\]"),
        (lambda X: norm.bm25plus(X, delta=-1), ValueError, "delta must be finite and >= 0"),
    ],
)
def test_rejects_bad_arguments(call, error, match):
    with pytest.raises(error, match=match):
        call(random_csr())


def test_weighting_parameter_bounds_accepted():
    # the edges of the accepted ranges: k1 = 0 (TF part 1 for any count), b = 0 and 1, delta = 0
    X = random_csr()
    for kw in ({"k1": 0.0}, {"b": 0.0}, {"b": 1.0}, {"logbase": 1.0001}):
        out = norm.bm25(X, **kw)
        assert np.isfinite(out.data).all() and (out.data > 0).all()
    np.testing.assert_array_equal(norm.bm25plus(X, delta=0.0).toarray(), norm.bm25(X).toarray())


@pytest.mark.parametrize("axis", [1, 0])
@pytest.mark.parametrize("fn", [norm.tfidf, norm.bm25plus])
def test_prob_idf_term_in_every_document_weighs_zero(fn, axis):
    # 4 documents; column 0 in all (df = N: 0), column 1 in 3 (log(1/3) < 0), column 2 in 1 (log 3 > 0)
    A = np.array([[1.0, 1, 1], [2, 1, 0], [1, 1, 0], [3, 0, 0]])
    out = fn(sp.csr_array(A if axis == 1 else A.T), axis=axis, tf_mode="raw", idf_mode="prob").toarray()
    out = out if axis == 1 else out.T
    assert np.isfinite(out).all()
    assert (out[:, 0] == 0).all()
    assert (out[A[:, 1] > 0, 1] < 0).all() and (out[A[:, 2] > 0, 2] > 0).all()
