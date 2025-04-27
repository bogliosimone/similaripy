import numpy as np
import scipy.sparse as sp

import similaripy.normalization as norm


def generate_random_matrix(n_rows=100, n_cols=50, density=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return sp.random(n_rows, n_cols, density=density, format='csr', dtype=np.float32, random_state=rng)


def test_normalize_l1():
    X = generate_random_matrix()
    X_norm = norm.normalize(X, norm='l1')

    expected = X.copy()
    row_sums = expected.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    expected.data /= np.repeat(row_sums, np.diff(expected.indptr))

    np.testing.assert_allclose(X_norm.toarray(), expected.toarray(), rtol=1e-5)
    print('✅ normalize l1 correctness passed')


def test_normalize_l2():
    X = generate_random_matrix()
    X_norm = norm.normalize(X, norm='l2')

    expected = X.copy()
    row_norms = np.sqrt(expected.multiply(expected).sum(axis=1)).A1
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

    bm25_ref = sp.csr_matrix((new_data, (row, col)), shape=tf.shape)

    np.testing.assert_allclose(X_bm25.toarray(), bm25_ref.toarray(), rtol=1e-3)
    print("✅ bm25 correctness passed")


if __name__ == "__main__":
        test_normalize_l1()
        test_normalize_l2()
        test_normalize_max()
        test_tfidf()
        test_bm25()
