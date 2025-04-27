from .cython_code import normalization as _norm
from math import e
import scipy.sparse as sps
import numpy as np


_NORMALIZATIONS = ('l1', 'l2', 'max')
_TF = ('binary', 'raw', 'sqrt', 'freq', 'log')
_IDF = ('unary', 'base', 'smooth', 'prob', 'bm25')

def normalize(
    X: sps.spmatrix,
    norm: str = 'l2',
    axis: int = 1,
    inplace: bool = False
) -> sps.csr_matrix:
    """
    Normalize a sparse matrix along rows or columns using L1, L2, or max-norm.

    Args:
        X: Input sparse matrix.
        norm: Normalization method ('l1', 'l2', or 'max').
        axis: Normalize rows (1) or columns (0).
        inplace: Whether to modify the matrix in place.

    Returns:
        Normalized CSR matrix.
    """
    assert(norm in _NORMALIZATIONS)
    X = check_matrix(X)

    if not inplace: 
        X = X.copy()
    if axis == 0: 
        X = X.T
    
    X = X.tocsr()
    if norm == 'l1':
        _norm.inplace_normalize_csr_l1(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    elif norm == 'l2':
        _norm.inplace_normalize_csr_l2(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    elif norm == 'max':
        _norm.inplace_normalize_csr_max(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    
    if axis == 0: X = X.T
    return X.tocsr()


def bm25(
    X: sps.spmatrix,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    logbase: float = e,
    tf_mode: str = 'raw',
    idf_mode: str = 'bm25',
    inplace: bool = False
) -> sps.csr_matrix:
    """
    Apply BM25 normalization to a sparse matrix.

    Args:
        X: Input sparse matrix.
        axis: Normalize rows (1) or columns (0).
        k1: Term saturation parameter.
        b: Length normalization parameter.
        logbase: Logarithm base.
        tf_mode: Term frequency mode ('raw', 'log', 'sqrt', etc.).
        idf_mode: Inverse document frequency mode ('bm25', 'smooth', etc.).
        inplace: Modify the matrix in place.

    Returns:
        BM25-normalized CSR matrix.
    """
    assert(tf_mode in _TF)
    assert(idf_mode in _IDF)
    X = check_matrix(X)

    if not inplace: 
        X = X.copy()
    if axis == 0: 
        X = X.T

    X = X.tocsr()
    _norm.inplace_normalize_csr_bm25plus(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
                                         k1=k1, b=b, delta=0.0,
                                         tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)

    if axis == 0: X = X.T
    return X.tocsr()
    

def bm25plus(
    X: sps.spmatrix,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    logbase: float = e,
    tf_mode: str = 'raw',
    idf_mode: str = 'bm25',
    inplace: bool = False
) -> sps.csr_matrix:
    """
    Apply BM25+ normalization to a sparse matrix.

    Args:
        X: Input sparse matrix.
        axis: Normalize rows (1) or columns (0).
        k1: Term saturation parameter.
        b: Length normalization parameter.
        delta: BM25+ boosting parameter.
        logbase: Logarithm base.
        tf_mode: Term frequency mode.
        idf_mode: Inverse document frequency mode.
        inplace: Modify the matrix in place.

    Returns:
        BM25+ normalized CSR matrix.
    """
    assert(tf_mode in _TF)
    assert(idf_mode in _IDF)
    X = check_matrix(X)

    if not inplace: 
        X = X.copy()
    if axis == 0: 
        X = X.T

    X = X.tocsr()
    _norm.inplace_normalize_csr_bm25plus(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
                                         k1=k1, b=b, delta=delta,
                                         tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)

    if axis == 0: 
        X = X.T
    return X.tocsr()


def tfidf(
    X: sps.spmatrix,
    axis: int = 1,
    logbase: float = e,
    tf_mode: str = 'sqrt',
    idf_mode: str = 'smooth',
    inplace: bool = False
) -> sps.csr_matrix:
    """
    Apply TF-IDF normalization to a sparse matrix.

    Args:
        X: Input sparse matrix.
        axis: Normalize rows (1) or columns (0).
        logbase: Logarithm base.
        tf_mode: Term frequency mode.
        idf_mode: Inverse document frequency mode.
        inplace: Modify the matrix in place.

    Returns:
        TF-IDF normalized CSR matrix.
    """
    assert(tf_mode in _TF)
    assert(idf_mode in _IDF)
    X = check_matrix(X)

    if not inplace: 
        X = X.copy()
    if axis == 0: 
        X = X.T

    X = X.tocsr()
    _norm.inplace_normalize_csr_tfidf(shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
                                         tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)

    if axis == 0: 
        X = X.T
    return X.tocsr()


def check_matrix(X: sps.spmatrix) -> sps.csr_matrix:
    """
    Ensure the input is a valid CSR sparse matrix of float32 or float64 dtype.

    Args:
        X: Input sparse matrix.

    Returns:
        Converted CSR matrix with float32 dtype if needed.
    """
    assert sps.issparse(X), 'X must be a sparse matrix'
    if X.data.dtype not in (np.float32, np.float64):
        X = sps.csr_matrix(X, dtype=np.float32)
    return X

    