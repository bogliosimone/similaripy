import similaripy as sim
import numpy as np
import scipy.sparse as sp
from similaripy.normalization import normalize


def check_sum(x):
    # function used for testing the library
    # sum of each row, next square of each value, next sum the values
    # since we use topk, we can't check on the other axis
    aux = x.sum(axis=1).A1
    aux = np.power(aux, 2)
    return np.sum(aux)


def check_full(x1, x2, rtol=0.001):
    # this function could be used only if we don't use topk
    # bescause we could have same value for different indices in a row
    # so one method could chose one indice and one another 
    x1 = x1.tocsr()
    x2 = x2.tocsr()
    for i in range(x1.shape[0]):
        indices = x1.indices[x1.indptr[i]:x1.indptr[i+1]]
        for i in range(indices.shape[0]):
            r = i
            c = indices[i]
            np.testing.assert_allclose(x1[r,c], x2[r,c], rtol=rtol, err_msg='error test_full')
    return 0


def py_dot(m, k):
    s = m * m.T
    return top_k(s, k)


def py_cosine(m, k):
    m2 = m.copy()
    m2.data = np.power(m2.data,2)
    X = np.power(m2.sum(axis=1).A1,0.5)
    m_aux = (m * m.T).tocsr()
    r, c, v = [], [], []
    for idx1 in range(0,m.shape[0]):
        for idx2 in range(m_aux.indptr[idx1], m_aux.indptr[idx1+1]):
            row = idx1
            col = m_aux.indices[idx2]
            val = m_aux.data[idx2]
            r.append(row)
            c.append(col)
            v.append(val/ (X[row] * X[col]))
    s = sp.csr_matrix((v,(r,c)),shape=(m.shape[0],m.shape[0]))
    return top_k(s, k)


def py_asy_cosine(m, alpha, k):
    m2 = m.copy()
    m2.data = np.power(m2.data,2)
    X = np.power(m2.sum(axis=1).A1,alpha)
    Y = np.power(m2.sum(axis=1).A1,1-alpha)
    m_aux = (m * m.T).tocsr()
    r, c, v = [], [], []
    for idx1 in range(0,m.shape[0]):
        for idx2 in range(m_aux.indptr[idx1], m_aux.indptr[idx1+1]):
            row = idx1
            col = m_aux.indices[idx2]
            val = m_aux.data[idx2]
            r.append(row)
            c.append(col)
            v.append(val/ (X[row] * Y[col]))
    s = sp.csr_matrix((v,(r,c)),shape=(m.shape[0],m.shape[0]))
    return top_k(s, k)


def py_jaccard(m, k):
    X = m.power(2).sum(axis=1).A1
    m_aux = (m * m.T).tocsr()
    r, c, v = [], [], []
    for idx1 in range(0,m.shape[0]):
        for idx2 in range(m_aux.indptr[idx1], m_aux.indptr[idx1+1]):
            row = idx1
            col = m_aux.indices[idx2]
            val = m_aux.data[idx2]
            r.append(row)
            c.append(col)
            v.append(val/ (X[row] + X[col] - val))
    s = sp.csr_matrix((v,(r,c)),shape=(m.shape[0],m.shape[0]))
    return top_k(s, k)


def py_dice(m, k):
    X = m.power(2).sum(axis=1).A1
    m_aux = (m * m.T).tocsr()
    r, c, v = [], [], []
    for idx1 in range(0,m.shape[0]):
        for idx2 in range(m_aux.indptr[idx1], m_aux.indptr[idx1+1]):
            row = idx1
            col = m_aux.indices[idx2]
            val = m_aux.data[idx2]
            r.append(row)
            c.append(col)
            v.append(2*val/ (X[row] + X[col]))
    s = sp.csr_matrix((v,(r,c)),shape=(m.shape[0],m.shape[0]))
    return top_k(s, k)


def py_tversky(m, alpha, beta, k):
    X = m.power(2).sum(axis=1).A1
    m_aux = (m * m.T).tocsr()
    r, c, v = [], [], []
    for idx1 in range(0,m.shape[0]):
        for idx2 in range(m_aux.indptr[idx1], m_aux.indptr[idx1+1]):
            row = idx1
            col = m_aux.indices[idx2]
            val = m_aux.data[idx2]
            r.append(row)
            c.append(col)
            v.append(val/ (alpha*(X[row]-val) + beta*(X[col]-val) + val))
    s = sp.csr_matrix((v,(r,c)),shape=(m.shape[0],m.shape[0]))
    return top_k(s, k)


def py_p3alpha(m, alpha, k):
    m2 = m.copy().T
    m1 = normalize(m, axis=1, norm='l1')
    m2 = normalize(m2, axis=1, norm='l1')
    m1.data = np.power(m1.data, alpha)
    m2.data = np.power(m2.data, alpha)
    m_aux =  m1 * m2
    return top_k(m_aux, k)


def py_rp3beta(m, alpha, beta, k):
    pop = np.power(m.sum(axis=1).A1, beta)
    pop_inv = np.divide(1, pop, where=pop!=0)
    m2 = m.copy().T
    m1 = normalize(m, axis=1, norm='l1')
    m2 = normalize(m2, axis=1, norm='l1')
    m1.data = np.power(m1.data, alpha)
    m2.data = np.power(m2.data, alpha)
    m_aux = m1 * m2
    m_aux = col_scale(m_aux, pop_inv)
    return top_k(m_aux, k)


def top_k(X, k):
    X = X.tocsr()
    r, c, d = [], [], []
    for i in range(X.shape[0]):
        data = X.data[X.indptr[i]:X.indptr[i+1]]
        topk = min(k, data.shape[0])
        indices = X.indices[X.indptr[i]:X.indptr[i+1]]
        topk_idx = np.argpartition(data, -topk)[-topk:]
        data = data[topk_idx]
        indices = indices[topk_idx]
        r += np.full(topk, i).tolist()
        c += indices.tolist()
        d += data.tolist()
    return sp.csr_matrix((d, (r, c)), shape=X.shape)


def col_scale(X, array_scale):
    X = X.tocsr()
    X.data *= array_scale.take(X.indices, mode='clip')
    return X


def check_similarity(m, k, rtol=0.0001, full=False):
    # cython
    dot = sim.dot_product(m, k=k)
    cosine = sim.cosine(m, k=k)
    asy_cosine = sim.asymmetric_cosine(m, alpha=0.2, k=k)
    jaccard = sim.jaccard(m, k=k)
    dice = sim.dice(m, k=k)
    tversky = sim.tversky(m, alpha=0.8, beta=0.4, k=k)
    p3alpha = sim.p3alpha(m, alpha=0.8, k=k)
    rp3beta = sim.rp3beta(m, alpha=0.8, beta= 0.4, k=k)

    # python
    dot2 = py_dot(m, k)
    cosine2 = py_cosine(m, k).tocsr()
    asy_cosine2 = py_asy_cosine(m, 0.2, k=k)
    jaccard2 = py_jaccard(m, k)
    dice2 = py_dice(m, k)
    tversky2 = py_tversky(m, alpha=0.8, beta=0.4, k=k)
    p3alpha2 = py_p3alpha(m, alpha=0.8, k=k)
    rp3beta2 = py_rp3beta(m, alpha=0.8, beta=0.4, k=k)

    # test
    np.testing.assert_allclose(check_sum(dot), check_sum(dot2), rtol=rtol, err_msg='dot error')
    np.testing.assert_allclose(check_sum(cosine), check_sum(cosine2), rtol=rtol, err_msg='cosine error')
    np.testing.assert_allclose(check_sum(asy_cosine), check_sum(asy_cosine2), rtol=rtol, err_msg='asy_cosine error')
    np.testing.assert_allclose(check_sum(jaccard), check_sum(jaccard2), rtol=rtol, err_msg='jaccard error')
    np.testing.assert_allclose(check_sum(dice), check_sum(dice2), rtol=rtol, err_msg='dice error')
    np.testing.assert_allclose(check_sum(tversky), check_sum(tversky2), rtol=rtol, err_msg='tversky error')
    np.testing.assert_allclose(check_sum(p3alpha), check_sum(p3alpha2), rtol=rtol, err_msg='p3alpha error')
    np.testing.assert_allclose(check_sum(rp3beta), check_sum(rp3beta2), rtol=rtol, err_msg='rp3beta error')

    # test full rows
    if full:
        np.testing.assert_(check_full(dot, dot2, rtol) == 0, msg='dot error')
        np.testing.assert_(check_full(cosine, cosine2, rtol) == 0, msg='cosine error')
        np.testing.assert_(check_full(asy_cosine, asy_cosine2, rtol) == 0, msg='asy_cosine error')
        np.testing.assert_(check_full(jaccard, jaccard2, rtol) == 0, msg='jaccard error')
        np.testing.assert_(check_full(dice, dice2, rtol) == 0, msg='dice error')
        np.testing.assert_(check_full(tversky, tversky2, rtol) == 0, msg='tversky error')
        np.testing.assert_(check_full(p3alpha, p3alpha2, rtol) == 0, msg='p3alpha error')
        np.testing.assert_(check_full(rp3beta, rp3beta2, rtol) == 0, msg='rp3beta error')

    return


def test_similarity_topk():
    rows = 1000
    cols = 800
    density = 0.025
    rtol= 0.0001
    k = 50

    m = sp.random(rows, cols, density=density).tocsr()
    
    check_similarity(m=m, k=k, rtol=rtol, full=False)

    print('All similarity topk tests passed!!!')


def test_similarity_full():
    rows = 400
    cols = 50
    density = 0.025
    rtol= 0.0001
    k = cols

    m = sp.random(rows, cols, density=density).tocsr()
    
    check_similarity(m=m, k=k, rtol=rtol, full=True)

    print('All similarity full row tests passed!!!')


def test_readme_code():
    import similaripy as sim
    import scipy.sparse as sps

    # create a random user-rating matrix (URM)
    urm = sps.random(1000, 2000, density=0.025)

    # normalize matrix with bm25
    urm = sim.normalization.bm25(urm)

    # train the model with 50 knn per item 
    model = sim.cosine(urm.T, k=50)

    # recommend 100 items to users 1, 14 and 8 filtering the items already seen by each users
    user_recommendations = sim.dot_product(urm, model.T, k=100, target_rows=[1,14,8], filter_cols=urm)

    print('Test README.md code passed!!!')

if __name__ == "__main__":
    test_similarity_topk()
    test_similarity_full()
    test_readme_code()



