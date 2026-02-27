from .cython_code import normalization as _norm
from math import e

import scipy.sparse as sps
import numpy as np


# ---- Valid mode strings ----
_NORMALIZATIONS = ('l1', 'l2', 'max')
_TF_MODES = ('binary', 'raw', 'sqrt', 'freq', 'log')
_IDF_MODES = ('unary', 'base', 'smooth', 'prob', 'bm25')

# ---- Norm function dispatch ----
_NORM_DISPATCH = {
    'l1': _norm.inplace_normalize_csr_l1,
    'l2': _norm.inplace_normalize_csr_l2,
    'max': _norm.inplace_normalize_csr_max,
}


# ---- Private helpers ----

def _check_matrix(X: sps.sparray) -> sps.sparray:
    """
    Ensure the input is a valid sparse matrix with float32 or float64 dtype.

    Args:
        X: Input sparse matrix.

    Returns:
        Sparse matrix with float32 dtype if original dtype was unsupported.

    Raises:
        TypeError: If X is not a sparse matrix.
    """
    if not sps.issparse(X):
        raise TypeError("X must be a sparse matrix")
    if X.data.dtype not in (np.float32, np.float64):
        X = sps.csr_array(X, dtype=np.float32)
    return X


def _prepare_csr(X: sps.sparray, axis: int, inplace: bool) -> sps.csr_array:
    """
    Validate, optionally copy, handle axis transposition, and convert to CSR.

    Args:
        X: Input sparse matrix.
        axis: Normalize rows (1) or columns (0).
        inplace: Whether to modify the matrix in place.

    Returns:
        CSR matrix ready for in-place normalization.

    Raises:
        TypeError: If X is not a sparse matrix.
        ValueError: If axis is not 0 or 1.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    X = _check_matrix(X)
    if not inplace:
        X = X.copy()
    if axis == 0:
        X = X.T
    return X.tocsr()


def _finalize_csr(X: sps.sparray, axis: int) -> sps.csr_array:
    """Undo axis transposition and return final CSR matrix."""
    if axis == 0:
        X = X.T
    return X.tocsr()


def _validate_modes(tf_mode: str, idf_mode: str) -> None:
    """
    Validate TF and IDF mode strings.

    Raises:
        ValueError: If tf_mode or idf_mode is not a recognized mode.
    """
    if tf_mode not in _TF_MODES:
        raise ValueError(f"tf_mode must be one of {_TF_MODES}, got '{tf_mode}'")
    if idf_mode not in _IDF_MODES:
        raise ValueError(f"idf_mode must be one of {_IDF_MODES}, got '{idf_mode}'")


# ---- Public API ----

def normalize(
    X: sps.sparray,
    norm: str = 'l2',
    axis: int = 1,
    inplace: bool = False,
) -> sps.csr_array:
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
    if norm not in _NORMALIZATIONS:
        raise ValueError(f"norm must be one of {_NORMALIZATIONS}, got '{norm}'")
    X = _prepare_csr(X, axis, inplace)
    _NORM_DISPATCH[norm](shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr)
    return _finalize_csr(X, axis)


def bm25(
    X: sps.sparray,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    logbase: float = e,
    tf_mode: str = 'raw',
    idf_mode: str = 'bm25',
    inplace: bool = False,
) -> sps.csr_array:
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
    _validate_modes(tf_mode, idf_mode)
    X = _prepare_csr(X, axis, inplace)
    _norm.inplace_normalize_csr_bm25plus(
        shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
        k1=k1, b=b, delta=0.0,
        tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase,
    )
    return _finalize_csr(X, axis)


def bm25plus(
    X: sps.sparray,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    logbase: float = e,
    tf_mode: str = 'raw',
    idf_mode: str = 'bm25',
    inplace: bool = False,
) -> sps.csr_array:
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
    _validate_modes(tf_mode, idf_mode)
    X = _prepare_csr(X, axis, inplace)
    _norm.inplace_normalize_csr_bm25plus(
        shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
        k1=k1, b=b, delta=delta,
        tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase,
    )
    return _finalize_csr(X, axis)


def tfidf(
    X: sps.sparray,
    axis: int = 1,
    logbase: float = e,
    tf_mode: str = 'sqrt',
    idf_mode: str = 'smooth',
    inplace: bool = False,
) -> sps.csr_array:
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
    _validate_modes(tf_mode, idf_mode)
    X = _prepare_csr(X, axis, inplace)
    _norm.inplace_normalize_csr_tfidf(
        shape=X.shape, data=X.data, indices=X.indices, indptr=X.indptr,
        tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase,
    )
    return _finalize_csr(X, axis)

    