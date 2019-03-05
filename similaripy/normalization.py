from .cython_code import normalization as _norm
import scipy.sparse as sps
from math import e

_NORMALIZATIONS = ('l1', 'l2', 'max')
_TF = ('binary', 'raw', 'sqrt', 'freq', 'log')
_IDF = ('unary', 'base', 'smooth', 'prob', 'bm25')

def normalize(X, norm='l2', axis=1, inplace=False):
    assert(norm in _NORMALIZATIONS)
    assert(sps.isspmatrix(X))

    if not inplace: X = X.copy()
    if axis == 0: X = X.T
    
    X = X.tocsr()
    if norm == 'l1':
        _norm.inplace_normalize_csr_l1(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    elif norm == 'l2':
        _norm.inplace_normalize_csr_l2(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    elif norm == 'max':
        _norm.inplace_normalize_csr_max(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    
    if axis == 0: X = X.T
    return X.tocsr()


def bm25(X, axis=1, k1=1.2, b=0.75, logbase=e, tf_mode='raw', idf_mode='bm25', inplace=False):
    assert(tf_mode in _TF)
    assert(idf_mode in _IDF)
    assert(sps.isspmatrix(X))

    if not inplace: X = X.copy()
    if axis == 0: X = X.T

    X = X.tocsr()
    _norm.inplace_normalize_csr_bm25plus(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
                                         k1=k1, b=b, delta=0.0,
                                         tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)

    if axis == 0: X = X.T
    return X.tocsr()
    

def bm25plus(X, axis=1, k1=1.2, b=0.75, delta=1.0, logbase=e, tf_mode='raw', idf_mode='bm25', inplace=False):
    assert(tf_mode in _TF)
    assert(idf_mode in _IDF)
    assert(sps.isspmatrix(X))

    if not inplace: X = X.copy()
    if axis == 0: X = X.T

    X = X.tocsr()
    _norm.inplace_normalize_csr_bm25plus(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
                                         k1=k1, b=b, delta=delta,
                                         tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)

    if axis == 0: X = X.T
    return X.tocsr()


def tfidf(X, axis=1, logbase=e, tf_mode='sqrt', idf_mode='smooth', inplace=False):
    assert(tf_mode in _TF)
    assert(idf_mode in _IDF)
    assert(sps.isspmatrix(X))

    if not inplace: X = X.copy()
    if axis == 0: X = X.T

    X = X.tocsr()
    _norm.inplace_normalize_csr_tfidf(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
                                         tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)

    if axis == 0: X = X.T
    return X.tocsr()
    