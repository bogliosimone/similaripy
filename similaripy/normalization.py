from . import _core
from ._s_plus import _canonical_csr
from math import e

import scipy.sparse as sps
import numpy as np


# ---- Valid mode strings ----
_NORMALIZATIONS = ('l1', 'l2', 'max')
_TF_MODES = ('binary', 'raw', 'sqrt', 'freq', 'log')
_IDF_MODES = ('unary', 'base', 'smooth', 'prob', 'bm25')


# ---- Private helpers ----

def _canonical(X: sps.sparray, axis: int = 1) -> sps.csr_array:
    """
    The call's own canonical CSR copy of X (axis 1) or X.T (axis 0), see
    `_canonical_csr`: float32/float64 values kept, other value types -> float32;
    int32 indices.

    Raises:
        ValueError: If axis is not 0 or 1, or X has more than 2^31 - 1 entries.
        TypeError: If X is not a sparse matrix.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if not sps.issparse(X):
        raise TypeError("X must be a sparse matrix")
    dtype = X.dtype if X.dtype in (np.float32, np.float64) else np.float32
    return _canonical_csr(X.T if axis == 0 else X, dtype, 0)


def _transformed(X: sps.sparray, axis: int, op, *args, name: str = None) -> sps.csr_array:
    """
    The canonical copy of X with `op` (an in-place `_core` normalization) applied
    to its rows (axis 1) or columns (axis 0); `name`: a term weighting, whose
    input must be non-negative.
    """
    X = _canonical(X, axis)
    if name is not None:
        _check_non_negative(X, name)
    op(X.data, X.indices, X.indptr, X.shape, *args)
    return X.T.tocsr() if axis == 0 else X


def _check_non_negative(X: sps.csr_array, name: str) -> None:
    """Term weightings need counts: a negative value would give a NaN log or a
    negative document length (signed input). Checked on the canonical data,
    so +1/-1 duplicates sum to 0 and pass."""
    if (X.data < 0).any():
        raise ValueError(f'{name} needs non-negative values (term frequencies); '
                         'signed input (values < 0) is not supported')


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
) -> sps.csr_array:
    """
    Normalize a sparse matrix along rows or columns using L1, L2, or max-norm.

    Signed input is fine: l1 divides by the sum of |x|, l2 by sqrt(sum of x^2),
    max by the largest |x|. Rows of zeros (empty after canonicalization) stay
    unchanged.

    Args:
        X: Input sparse matrix.
        norm: Normalization method ('l1', 'l2', or 'max').
        axis: Normalize rows (1) or columns (0).

    Returns:
        Normalized CSR matrix.
    """
    if norm not in _NORMALIZATIONS:
        raise ValueError(f"norm must be one of {_NORMALIZATIONS}, got '{norm}'")
    return _transformed(X, axis, _core.normalize, norm)


def binarize(X: sps.sparray) -> sps.csr_array:
    """
    Set every non-zero entry to 1.0 (set semantics): duplicates are summed
    first, then explicit zeros dropped. A negative value is an interaction
    too: it becomes 1.0.

    Same values as `binary=True` in the similarity functions, as a separate
    step: binarize only one of the two matrices, or binarize before another
    normalization.

    Args:
        X: Input sparse matrix.

    Returns:
        Binarized CSR matrix.
    """
    X = _canonical(X)
    X.data[:] = 1
    return X


def bm25(
    X: sps.sparray,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    logbase: float = e,
    tf_mode: str = 'raw',
    idf_mode: str = 'bm25',
) -> sps.csr_array:
    """
    Apply BM25 normalization to a sparse matrix.

    Args:
        X: Input sparse matrix; values >= 0 (term frequencies), checked after
            summing duplicates.
        axis: Normalize rows (1) or columns (0).
        k1: Term saturation parameter.
        b: Length normalization parameter.
        logbase: Logarithm base.
        tf_mode: Term frequency mode ('raw', 'log', 'sqrt', etc.).
        idf_mode: Inverse document frequency mode ('bm25', 'smooth', etc.).

    Returns:
        BM25-normalized CSR matrix.

    Raises:
        ValueError: on a negative value.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(X, axis, _core.bm25plus, k1, b, 0.0, tf_mode, idf_mode, logbase, name='bm25')


def bm25plus(
    X: sps.sparray,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    logbase: float = e,
    tf_mode: str = 'raw',
    idf_mode: str = 'bm25',
) -> sps.csr_array:
    """
    Apply BM25+ normalization to a sparse matrix.

    Args:
        X: Input sparse matrix; values >= 0 (term frequencies), checked after
            summing duplicates.
        axis: Normalize rows (1) or columns (0).
        k1: Term saturation parameter.
        b: Length normalization parameter.
        delta: BM25+ boosting parameter.
        logbase: Logarithm base.
        tf_mode: Term frequency mode.
        idf_mode: Inverse document frequency mode.

    Returns:
        BM25+ normalized CSR matrix.

    Raises:
        ValueError: on a negative value.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(X, axis, _core.bm25plus, k1, b, delta, tf_mode, idf_mode, logbase, name='bm25plus')


def tfidf(
    X: sps.sparray,
    axis: int = 1,
    logbase: float = e,
    tf_mode: str = 'sqrt',
    idf_mode: str = 'smooth',
) -> sps.csr_array:
    """
    Apply TF-IDF normalization to a sparse matrix.

    Args:
        X: Input sparse matrix; values >= 0 (term frequencies), checked after
            summing duplicates.
        axis: Normalize rows (1) or columns (0).
        logbase: Logarithm base.
        tf_mode: Term frequency mode.
        idf_mode: Inverse document frequency mode.

    Returns:
        TF-IDF normalized CSR matrix.

    Raises:
        ValueError: on a negative value.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(X, axis, _core.tfidf, tf_mode, idf_mode, logbase, name='tfidf')

    