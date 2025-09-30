import similaripy as sim
import numpy as np
import scipy.sparse as sp
from similaripy.normalization import normalize

VERBOSE = False

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


def py_cosine(m, k, h=0, shrink_mode='stabilized'):
    if shrink_mode == 'additive':
        additive_h = h
    else:
        additive_h = 0

    m2 = m.copy()
    m2.data = np.power(m2.data,2)

    # normalization terms
    X = np.power((m2.sum(axis=1).A1) + additive_h, 0.5)

    m_aux = (m * m.T).tocsr()
    r, c, v = [], [], []
    for idx1 in range(0,m.shape[0]):
        for idx2 in range(m_aux.indptr[idx1], m_aux.indptr[idx1+1]):
            row = idx1
            col = m_aux.indices[idx2]
            val = m_aux.data[idx2]
            r.append(row)
            c.append(col)
            if shrink_mode == 'stabilized':
                v.append(val/ (X[row] * X[col] + h))
            elif shrink_mode == 'bayesian':
                v.append(val / (X[row] * X[col]) * (val / (val + h)))
            elif shrink_mode == 'additive':
                v.append(val / (X[row] * X[col]))
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


def py_s_plus(m, k,
              l1=0.5, l2=0.5, l3=0.0,
              t1=1.0, t2=1.0,
              c1=0.5, c2=0.5,
              alpha=1.0,
              beta1=0.0, beta2=0.0,
              pop1='none', pop2='none'
              ):
    m_aux = (m * m.T).tocsr()

    # squared norms
    sq = m.copy()
    sq.data **= 2
    Xtversky = sq.sum(axis=1).A1
    Ytversky = Xtversky.copy()

    # cosine exponents
    Xcosine = np.power(Xtversky, c1)
    Ycosine = np.power(Ytversky, c2)

    # popularity (sum)
    if pop1 == 'sum':
        Xdepop = np.power(m.sum(axis=1).A1, beta1)
    else:
        Xdepop = np.ones(m.shape[0])
    if pop2 == 'sum':
        Ydepop = np.power(m.sum(axis=1).A1, beta2)
    else:
        Ydepop = np.ones(m.shape[0])
    
    r, c, v = [], [], []
    for i in range(m_aux.shape[0]):
        for j in range(m_aux.indptr[i], m_aux.indptr[i+1]):
            row = i
            col = m_aux.indices[j]
            xy = m_aux.data[j]

            valTversky = l1 * (t1 * (Xtversky[row] - xy) + t2 * (Ytversky[col] - xy) + xy) if l1 != 0 else 0
            valCosine  = l2 * (Xcosine[row] * Ycosine[col]) if l2 != 0 else 0
            valDepop   = l3 * (Xdepop[row] * Ydepop[col]) if l3 != 0 else 0

            denom = valTversky + valCosine + valDepop
            if alpha != 1.0:
                xy = np.power(xy, alpha)
            val = xy / denom if denom > 0 else 0
            r.append(row)
            c.append(col)
            v.append(val)

    s = sp.csr_matrix((v, (r, c)), shape=(m.shape[0], m.shape[0]))
    return top_k(s, k)


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
    dot = sim.dot_product(m, k=k, verbose=VERBOSE)
    cosine = sim.cosine(m, k=k, verbose=VERBOSE)
    asy_cosine = sim.asymmetric_cosine(m, alpha=0.2, k=k, verbose=VERBOSE)
    jaccard = sim.jaccard(m, k=k, verbose=VERBOSE)
    dice = sim.dice(m, k=k, verbose=VERBOSE)
    tversky = sim.tversky(m, alpha=0.8, beta=0.4, k=k, verbose=VERBOSE)
    p3alpha = sim.p3alpha(m, alpha=0.8, k=k, verbose=VERBOSE)
    rp3beta = sim.rp3beta(m, alpha=0.8, beta= 0.4, k=k, verbose=VERBOSE)
    splus = sim.s_plus(m, l1=0.5, l2=0.5, l3=1, t1=1, t2=1, c1=0.5, c2=0.5, 
                       alpha=1, beta1=0, beta2=0, pop1='none',pop2='sum', k=k, verbose=VERBOSE)

    # python
    dot2 = py_dot(m, k)
    cosine2 = py_cosine(m, k).tocsr()
    asy_cosine2 = py_asy_cosine(m, 0.2, k=k)
    jaccard2 = py_jaccard(m, k)
    dice2 = py_dice(m, k)
    tversky2 = py_tversky(m, alpha=0.8, beta=0.4, k=k)
    p3alpha2 = py_p3alpha(m, alpha=0.8, k=k)
    rp3beta2 = py_rp3beta(m, alpha=0.8, beta=0.4, k=k)
    splus2 = py_s_plus(m, l1=0.5, l2=0.5, l3=1, t1=1, t2=1, c1=0.5, c2=0.5, 
                       alpha=1, beta1=0, beta2=0, pop1='none',pop2='sum', k=k)

    # test
    np.testing.assert_allclose(check_sum(dot), check_sum(dot2), rtol=rtol, err_msg='dot error')
    np.testing.assert_allclose(check_sum(cosine), check_sum(cosine2), rtol=rtol, err_msg='cosine error')
    np.testing.assert_allclose(check_sum(asy_cosine), check_sum(asy_cosine2), rtol=rtol, err_msg='asy_cosine error')
    np.testing.assert_allclose(check_sum(jaccard), check_sum(jaccard2), rtol=rtol, err_msg='jaccard error')
    np.testing.assert_allclose(check_sum(dice), check_sum(dice2), rtol=rtol, err_msg='dice error')
    np.testing.assert_allclose(check_sum(tversky), check_sum(tversky2), rtol=rtol, err_msg='tversky error')
    np.testing.assert_allclose(check_sum(p3alpha), check_sum(p3alpha2), rtol=rtol, err_msg='p3alpha error')
    np.testing.assert_allclose(check_sum(rp3beta), check_sum(rp3beta2), rtol=rtol, err_msg='rp3beta error')
    np.testing.assert_allclose(check_sum(splus), check_sum(splus2), rtol=rtol, err_msg='splus error')

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
        np.testing.assert_(check_full(splus, splus2, rtol) == 0, msg='splus error')

    return

def generate_random_matrix(n_rows=100, n_cols=50, density=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return sp.random(n_rows, n_cols, density=density, format='csr', dtype=np.float32, random_state=rng)


def test_similarity_topk():
    rows = 1000
    cols = 800
    density = 0.025
    rtol= 0.0001
    k = 50

    m = generate_random_matrix(rows, cols, density=density).tocsr()
    
    check_similarity(m=m, k=k, rtol=rtol, full=False)

    print('✅ All similarity topk tests passed')


def test_similarity_full():
    rows = 400
    cols = 50
    density = 0.025
    rtol= 0.0001
    k = cols

    m = generate_random_matrix(rows, cols, density=density).tocsr()
    
    check_similarity(m=m, k=k, rtol=rtol, full=True)

    print('✅ All similarity full row tests passed')


def test_shrink_types():
    rows = 400
    cols = 50
    density = 0.025
    rtol= 0.0001
    k = cols
    m = generate_random_matrix(rows, cols, density=density).tocsr()
    
    for mode in ('stabilized', 'bayesian', 'additive'):
        # cython
        cosine = sim.cosine(m, k=k, shrink=10, shrink_type=mode, verbose=VERBOSE)
        # python
        cosine2 = py_cosine(m, k, h=10, shrink_mode=mode).tocsr()

        np.testing.assert_allclose(check_sum(cosine), check_sum(cosine2), rtol=rtol, err_msg=f'Mismatch for shrink_type={mode}')
        np.testing.assert_(check_full(cosine, cosine2, rtol) == 0, msg=f'Mismatch for shrink_type={mode}')

    print('✅ All shrink tests passed')


def test_output_format():
    rows = 1000
    cols = 800
    density = 0.025
    k = 50
    m = generate_random_matrix(rows, cols, density=density).tocsr()

    # CSR output
    sim_csr = sim.cosine(m, format_output='csr', k=k, verbose=VERBOSE)
    assert sp.issparse(sim_csr), "Output is not a sparse matrix"
    assert isinstance(sim_csr, sp.csr_matrix), "CSR format not returned"

    # COO output
    sim_coo = sim.cosine(m, format_output='coo', k=k, verbose=VERBOSE)
    assert sp.issparse(sim_coo), "Output is not a sparse matrix"
    assert isinstance(sim_coo, sp.coo_matrix), "COO format not returned"

    assert sim_csr.nnz > 0, "CSR output is empty"
    assert sim_coo.nnz > 0, "COO output is empty"

    print("✅ Test output CSR and COO passed")

def test_example_code():
    import similaripy as sim
    import scipy.sparse as sps

   # Create a random User-Rating Matrix (URM)
    urm = sps.random(1000, 2000, density=0.025)

    # Normalize the URM using BM25
    urm = sim.normalization.bm25(urm)

    # Train an item-item cosine similarity model
    similarity_matrix = sim.cosine(urm.T, k=50)

    # Compute recommendations for user 1, 14, 8 
    # filtering out already-seen items
    recommendations = sim.dot_product(
        urm,
        similarity_matrix.T,
        k=100,
        target_rows=[1, 14, 8],
        filter_cols=urm
    )
    print('✅ Test README.md sample code passed')


def test_openmp_enabled():
    try:
        threads = sim.cython_code.s_plus.get_num_threads()
        print("✅ OpenMP detected — using {} threads".format(threads))
        assert threads >= 1
    except AttributeError:
        print("⚠️ OpenMP not detected or extension built without OpenMP — skipping test")


if __name__ == "__main__":
    test_openmp_enabled()
    test_similarity_topk()
    test_similarity_full()
    test_shrink_types()
    test_output_format()
    test_example_code()



