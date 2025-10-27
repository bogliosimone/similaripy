# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

"""
    s_plus specific utility functions
"""

import cython
import numpy as np
import scipy.sparse as sp

# Column selector mode constants
cdef int MODE_NONE = 0  # No filtering/targeting (use all columns)
cdef int MODE_ARRAY = 1  # Column selector is an array/list
cdef int MODE_MATRIX = 2  # Column selector is a sparse matrix


def validate_s_plus_inputs(
    matrix1, matrix2,
    weight_depop_matrix1, weight_depop_matrix2,
    k, target_rows, filter_cols, target_cols,
    verbose, format_output):
    """
    Validate input parameters for s_plus function.

    Parameters
    ----------
    matrix1 : scipy.sparse matrix
        First input matrix
    matrix2 : scipy.sparse matrix
        Second input matrix
    weight_depop_matrix1 : array-like or str
        Depopularization weights for matrix1
    weight_depop_matrix2 : array-like or str
        Depopularization weights for matrix2
    k : int
        Number of top items to keep
    target_rows : array-like or None
        Rows to compute
    filter_cols : array-like, sparse matrix, or None
        Columns to filter out
    target_cols : array-like, sparse matrix, or None
        Columns to target
    verbose : bool
        Whether to show progress
    format_output : str
        Output format ('coo' or 'csr')

    Raises
    ------
    TypeError
        If matrix1 or matrix2 are not sparse matrices
    ValueError
        If any parameter has invalid value or incompatible dimensions
    """
    # Check matrix types
    if not sp.issparse(matrix1):
        raise TypeError('matrix1 must be a sparse matrix')
    if not sp.issparse(matrix2):
        raise TypeError('matrix2 must be a sparse matrix')

    # Check matrix dimensions
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(
            f'Incompatible matrix shapes: matrix1.shape[1]={matrix1.shape[1]} '
            f'must equal matrix2.shape[0]={matrix2.shape[0]}'
        )

    # Check k parameter
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')

    # Check depopularization weights
    if not (len(weight_depop_matrix1) == matrix1.shape[0] or
            weight_depop_matrix1 in ('none', 'sum')):
        raise ValueError(
            f'weight_depop_matrix1 must be array of length {matrix1.shape[0]} '
            f'or one of ("none", "sum"), got length {len(weight_depop_matrix1)}'
        )

    if not (len(weight_depop_matrix2) == matrix2.shape[1] or
            weight_depop_matrix2 in ('none', 'sum')):
        raise ValueError(
            f'weight_depop_matrix2 must be array of length {matrix2.shape[1]} '
            f'or one of ("none", "sum"), got length {len(weight_depop_matrix2)}'
        )

    # Check target_rows
    if target_rows is not None and len(target_rows) > matrix1.shape[0]:
        raise ValueError(
            f'target_rows length ({len(target_rows)}) cannot exceed '
            f'matrix1.shape[0] ({matrix1.shape[0]})'
        )

    # Check filter_cols format and shape
    if filter_cols is not None:
        if not (sp.issparse(filter_cols) or isinstance(filter_cols, (list, np.ndarray))):
            raise TypeError(
                'filter_cols must be a sparse matrix, list, numpy array, or None'
            )
        # Validate shape for sparse matrices
        if sp.issparse(filter_cols) and filter_cols.data.shape[0] != 0:
            expected_shape = (matrix1.shape[0], matrix2.shape[1])
            if filter_cols.shape != expected_shape:
                raise ValueError(
                    f'filter_cols shape {filter_cols.shape} does not match expected '
                    f'shape {expected_shape}'
                )

    # Check target_cols format and shape
    if target_cols is not None:
        if not (sp.issparse(target_cols) or isinstance(target_cols, (list, np.ndarray))):
            raise TypeError(
                'target_cols must be a sparse matrix, list, numpy array, or None'
            )
        # Validate shape for sparse matrices
        if sp.issparse(target_cols) and target_cols.data.shape[0] != 0:
            expected_shape = (matrix1.shape[0], matrix2.shape[1])
            if target_cols.shape != expected_shape:
                raise ValueError(
                    f'target_cols shape {target_cols.shape} does not match expected '
                    f'shape {expected_shape}'
                )

    # Check verbose
    if not isinstance(verbose, bool):
        raise TypeError(f'verbose must be boolean, got {type(verbose).__name__}')

    # Check format_output
    if format_output not in ('coo', 'csr'):
        raise ValueError(f"format_output must be 'coo' or 'csr', got '{format_output}'")


def _build_squared_norms(matrix1, matrix2):
    """
    Build squared norms for both matrices.

    This computation is shared between Tversky and Cosine normalizations
    and should only be computed once when needed by either.

    Parameters
    ----------
    matrix1 : scipy.sparse.csr_matrix
        First matrix (already in CSR format with zeros eliminated)
    matrix2 : scipy.sparse.csr_matrix
        Second matrix (already in CSR format with zeros eliminated)

    Returns
    -------
    tuple
        (m1_sq_norms, m2_sq_norms) as float32 arrays
    """
    cdef float[:] m1_sq_norms = np.array(matrix1.power(2).sum(axis=1).A1, dtype=np.float32)
    cdef float[:] m2_sq_norms = np.array(matrix2.power(2).sum(axis=0).A1, dtype=np.float32)
    return m1_sq_norms, m2_sq_norms


def _build_tversky_normalization(m1_sq_norms, m2_sq_norms, float l1):
    """
    Build Tversky normalization arrays based on pre-computed squared norms.

    Parameters
    ----------
    m1_sq_norms : np.ndarray
        Squared norms for matrix1
    m2_sq_norms : np.ndarray
        Squared norms for matrix2
    l1 : float
        Weight for Tversky normalization

    Returns
    -------
    tuple
        (Xtversky, Ytversky) as float32 arrays.
        If l1 == 0, returns empty arrays.
    """
    cdef float[:] empty = np.array([], dtype=np.float32)
    cdef float[:] Xtversky = empty
    cdef float[:] Ytversky = empty

    if l1 != 0:
        Xtversky = m1_sq_norms
        Ytversky = m2_sq_norms

    return Xtversky, Ytversky


def _build_cosine_normalization(m1_sq_norms, m2_sq_norms, float l2, float c1, float c2, float additive_shrink):
    """
    Build Cosine normalization arrays based on squared norms.

    Parameters
    ----------
    m1_sq_norms : np.ndarray or None
        Squared norms for matrix1 (can be None if not computed)
    m2_sq_norms : np.ndarray or None
        Squared norms for matrix2 (can be None if not computed)
    l2 : float
        Weight for Cosine normalization
    c1 : float
        Power for Cosine normalization on matrix1
    c2 : float
        Power for Cosine normalization on matrix2
    additive_shrink : float
        Additive shrinkage for Cosine normalization

    Returns
    -------
    tuple
        (Xcosine, Ycosine) as float32 arrays
    """
    cdef float[:] empty = np.array([], dtype=np.float32)
    cdef float[:] Xcosine = empty
    cdef float[:] Ycosine = empty

    if l2 != 0:
        Xcosine = np.power(np.asarray(m1_sq_norms) + additive_shrink, c1, dtype=np.float32)
        Ycosine = np.power(np.asarray(m2_sq_norms) + additive_shrink, c2, dtype=np.float32)

    return Xcosine, Ycosine


def _build_depop_normalization(matrix1, matrix2, weight_spec1, weight_spec2, float p1, float p2, float l3):
    """
    Build Depopularization normalization arrays.

    Parameters
    ----------
    matrix1 : scipy.sparse.csr_matrix
        First matrix (already in CSR format with zeros eliminated)
    matrix2 : scipy.sparse.csr_matrix
        Second matrix (already in CSR format with zeros eliminated)
    weight_spec1 : array-like or str
        Either 'none', 'sum', or an array of custom weights for matrix1
    weight_spec2 : array-like or str
        Either 'none', 'sum', or an array of custom weights for matrix2
    p1 : float
        Power to raise weights to for matrix1
    p2 : float
        Power to raise weights to for matrix2
    l3 : float
        Weight for Depopularization normalization

    Returns
    -------
    tuple
        (Xdepop, Ydepop) as float32 arrays. Returns empty arrays if l3 == 0.
    """
    cdef float[:] empty = np.array([], dtype=np.float32)
    cdef float[:] Xdepop = empty
    cdef float[:] Ydepop = empty

    if l3 != 0:
        # Build depopularization weights for matrix1
        if isinstance(weight_spec1, (list, np.ndarray)):
            Xdepop = np.power(weight_spec1, p1, dtype=np.float32)
        elif weight_spec1 == 'none':
            Xdepop = np.power(np.ones(matrix1.shape[0]), p1, dtype=np.float32)
        elif weight_spec1 == 'sum':
            Xdepop = np.power(np.array(matrix1.sum(axis=1).A1, dtype=np.float32), p1, dtype=np.float32)
        else:
            raise ValueError(f"Invalid weight_spec1: {weight_spec1}")

        # Build depopularization weights for matrix2
        if isinstance(weight_spec2, (list, np.ndarray)):
            Ydepop = np.power(weight_spec2, p2, dtype=np.float32)
        elif weight_spec2 == 'none':
            Ydepop = np.power(np.ones(matrix2.shape[1]), p2, dtype=np.float32)
        elif weight_spec2 == 'sum':
            Ydepop = np.power(np.array(matrix2.sum(axis=0).A1, dtype=np.float32), p2, dtype=np.float32)
        else:
            raise ValueError(f"Invalid weight_spec2: {weight_spec2}")

    return Xdepop, Ydepop


def _build_matrix_data(matrix1, matrix2, binary):
    """
    Build matrix data arrays, handling binary mode.

    In binary mode, all non-zero values become 1.0 (set theory).
    Otherwise, convert data to float32.

    Parameters
    ----------
    matrix1 : scipy.sparse.csr_matrix
        First matrix
    matrix2 : scipy.sparse.csr_matrix
        Second matrix
    binary : bool
        If True, use set theory (all values = 1.0)

    Returns
    -------
    tuple
        (old_m1_data, old_m2_data) - original data arrays to restore later
    """
    # Save original data for restoration
    old_m1_data = matrix1.data
    old_m2_data = matrix2.data

    # Build data arrays based on binary flag
    if binary:
        # Set theory: all non-zero values become 1.0
        matrix1.data = np.ones(matrix1.data.shape[0], dtype=np.float32)
        matrix2.data = np.ones(matrix2.data.shape[0], dtype=np.float32)
    else:
        # Convert to float32 (copy if needed)
        matrix1.data = np.array(matrix1.data, dtype=np.float32)
        matrix2.data = np.array(matrix2.data, dtype=np.float32)

    return old_m1_data, old_m2_data


def _build_column_selector(cols):
    """
    Build column selector (filter or target) for similarity computation.

    Handles three cases:
    - None or empty: MODE_NONE (0) - use all columns
    - List/array: MODE_ARRAY (1) - specific column indices
    - Sparse matrix: MODE_MATRIX (2) - row-specific column selections

    Note: Shape validation is done in validate_s_plus_inputs() before calling this function.

    Parameters
    ----------
    cols : None, list, np.ndarray, or scipy.sparse matrix
        Column selector specification

    Returns
    -------
    tuple
        (mode, indptr, indices) where:
        - mode: int (0=none, 1=array, 2=matrix)
        - indptr: int32 array (CSR indptr format)
        - indices: int32 array (sorted column indices)
    """
    cdef int mode
    cdef int[:] indptr
    cdef int[:] indices

    # Case 1: Sparse matrix with data
    # Note: Shape validation is done in validate_s_plus_inputs()
    if sp.issparse(cols) and cols.data.shape[0] != 0:
        mode = MODE_MATRIX
        # Convert to CSR format and sort indices for binary search
        cols = cols.tocsr()
        cols.eliminate_zeros()
        cols.sort_indices()
        indptr = np.array(cols.indptr, dtype=np.int32)
        indices = np.array(cols.indices, dtype=np.int32)

    # Case 2: List or array with elements
    elif isinstance(cols, (list, np.ndarray)) and len(cols) != 0:
        mode = MODE_ARRAY
        # Sort array for binary search
        indptr = np.array([0, len(cols)], dtype=np.int32)
        indices = np.array(np.sort(cols), dtype=np.int32)

    # Case 3: None, empty, or sparse matrix with no data
    else:
        mode = MODE_NONE
        indptr = np.array([], dtype=np.int32)
        indices = np.array([], dtype=np.int32)

    return mode, indptr, indices
