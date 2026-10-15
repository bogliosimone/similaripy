import numpy as np
import scipy.sparse as sp

import similaripy.normalization as norm


def generate_random_matrix(n_rows=100, n_cols=50, density=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return sp.random_array((n_rows, n_cols), density=density, format='csr', dtype=np.float32, random_state=rng)


def test_binarize():
    X = generate_random_matrix()
    X.data[::5] = 0  # explicit zeros must go, not become 1
    before = X.data.copy()
    B = norm.binarize(X)

    expected = (X.toarray() != 0).astype(np.float32)
    np.testing.assert_array_equal(B.toarray(), expected)
    assert B.nnz == np.count_nonzero(before) and B.data.dtype == np.float32
    np.testing.assert_array_equal(X.data, before)  # not in place by default
    # same scores as binary=True
    import similaripy as sim
    np.testing.assert_allclose(sim.jaccard(B, k=10, verbose=False).toarray(),
                               sim.jaccard(X, k=10, binary=True, verbose=False).toarray(), rtol=1e-6)
    print('✅ binarize correctness passed')


def test_normalize_l1():
    X = generate_random_matrix()
    X_norm = norm.normalize(X, norm='l1')

    expected = X.copy()
    row_sums = np.asarray(expected.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    expected.data /= np.repeat(row_sums, np.diff(expected.indptr))

    np.testing.assert_allclose(X_norm.toarray(), expected.toarray(), rtol=1e-5)
    print('✅ normalize l1 correctness passed')


def test_normalize_l2():
    X = generate_random_matrix()
    X_norm = norm.normalize(X, norm='l2')

    expected = X.copy()
    row_norms = np.asarray(np.sqrt(expected.multiply(expected).sum(axis=1))).ravel()
    row_norms[row_norms == 0] = 1
    expected.data /= np.repeat(row_norms, np.diff(expected.indptr))

    np.testing.assert_allclose(X_norm.toarray(), expected.toarray(), rtol=1e-5)
    print('✅ normalize l2 correctness passed')


def test_normalize_max():
    X = generate_random_matrix()
    X_norm = norm.normalize(X, norm='max')

    expected = X.copy()
    max_values = expected.max(axis=1).toarray().flatten()
    max_values[max_values == 0] = 1
    expected.data /= np.repeat(max_values, np.diff(expected.indptr))

    np.testing.assert_allclose(X_norm.toarray(), expected.toarray(), rtol=1e-5)
    print('✅ normalize max correctness passed')


def test_tfidf():
    X = generate_random_matrix(n_rows=200, n_cols=100, density=0.05)
    X_tfidf = norm.tfidf(X, tf_mode='sqrt', idf_mode='smooth', logbase=np.e)

    tf = X.copy().tocsr()

    # compute tf (sqrt of raw freq)
    tf.data = np.sqrt(tf.data)

    # document frequency (DF): count of docs where term appears
    df = np.diff((X > 0).tocsc().indptr)
    idf = np.log(X.shape[0] / (1 + df))

    tf = tf.tocsc()
    tf.data *= np.repeat(idf, np.diff(tf.indptr))
    tf = tf.tocsr()

    np.testing.assert_allclose(X_tfidf.toarray(), tf.toarray(), rtol=1e-4)
    print("✅ tfidf correctness passed")


def test_bm25():
    X = generate_random_matrix(n_rows=200, n_cols=100, density=0.05)
    X_bm25 = norm.bm25(X, k1=1.2, b=0.75, tf_mode='raw', idf_mode='bm25', logbase=np.e)

    tf = X.copy().tocsr()
    dl = np.array(tf.sum(axis=1)).flatten()
    avgdl = np.mean(dl)

    # Compute IDF
    df = np.diff((tf > 0).tocsc().indptr)
    idf = np.log((tf.shape[0] - df + 0.5) / (df + 0.5))

    # Compute BM25 weights
    row, col = tf.nonzero()
    values = tf.data.copy()
    new_data = []
    for i, j, tf_ij in zip(row, col, values):
        denom = tf_ij + 1.2 * (1 - 0.75 + 0.75 * dl[i] / avgdl)
        score = tf_ij * (1.2 + 1) / denom * idf[j]
        new_data.append(score)

    bm25_ref = sp.csr_array((new_data, (row, col)), shape=tf.shape)

    np.testing.assert_allclose(X_bm25.toarray(), bm25_ref.toarray(), rtol=1e-3)
    print("✅ bm25 correctness passed")


def test_binarize_sums_duplicates():
    # Entry (0, 1) stored twice, (1, 2) as +1 and -1: summed first, so one 1.0 and a dropped zero
    X = sp.csr_array((np.array([1, 1, 1, -1, 2], dtype=np.float32), np.array([1, 1, 2, 2, 0]),
                      np.array([0, 2, 4, 5])), shape=(3, 3))
    B = norm.binarize(X)
    np.testing.assert_array_equal(B.toarray(), [[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    import similaripy as sim
    np.testing.assert_array_equal(sim.dot_product(B, k=3, verbose=False).toarray(),
                                  sim.dot_product(X, k=3, binary=True, verbose=False).toarray())


if __name__ == "__main__":
        test_binarize()
        test_normalize_l1()
        test_normalize_l2()
        test_normalize_max()
        test_tfidf()
        test_bm25()


# ---- Signed input (D45) ----

def _signed(seed=0):
    X = generate_random_matrix(seed=seed)
    X.data[::3] *= -1
    return X


def test_normalize_signed():
    X = _signed()
    A = X.toarray().astype(np.float64)
    for name, d in (('l1', np.abs(A).sum(axis=1)), ('l2', np.sqrt((A ** 2).sum(axis=1))),
                    ('max', np.abs(A).max(axis=1))):
        expected = np.divide(A, d[:, None], out=np.zeros_like(A), where=d[:, None] != 0)
        np.testing.assert_allclose(norm.normalize(X, norm=name).toarray(), expected, rtol=1e-6, err_msg=name)
    # all-negative row: max divides by its largest |x| (before: left unchanged)
    neg = sp.csr_array(np.array([[-2.0, -4.0]], dtype=np.float32))
    np.testing.assert_array_equal(norm.normalize(neg, norm='max').toarray(), [[-0.5, -1.0]])


def test_binarize_signed():
    X = _signed()
    np.testing.assert_array_equal(norm.binarize(X).toarray(), (X.toarray() != 0).astype(np.float32))


def test_term_weightings_reject_negatives():
    import pytest
    X = _signed()
    for f in (norm.tfidf, norm.bm25, norm.bm25plus):
        with pytest.raises(ValueError, match=f'{f.__name__} needs non-negative values'):
            f(X)
    # a +1/-1 duplicate pair sums to 0 (dropped), not a negative value
    dup = sp.coo_array((np.array([1, -1, 2], dtype=np.float32), ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    for f in (norm.tfidf, norm.bm25, norm.bm25plus):
        assert f(dup).nnz == 1
