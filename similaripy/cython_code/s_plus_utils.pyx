# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

"""
    s_plus specific utility functions
"""

import cython
import numpy as np
import scipy.sparse as sp
from typing import Union, Optional, Tuple, List

# Column selector mode constants
cdef int MODE_NONE = 0  # No filtering/targeting (use all columns)
cdef int MODE_ARRAY = 1  # Column selector is an array/list
cdef int MODE_MATRIX = 2  # Column selector is a sparse matrix


def validate_s_plus_inputs(
    matrix1: sp.spmatrix,
    matrix2: sp.spmatrix,
    weight_depop_matrix1: Union[str, np.ndarray],
    weight_depop_matrix2: Union[str, np.ndarray],
    k: int,
    target_rows: Optional[Union[List, np.ndarray]],
    filter_cols: Optional[Union[List, np.ndarray, sp.spmatrix]],
    target_cols: Optional[Union[List, np.ndarray, sp.spmatrix]],
    verbose: bool,
    format_output: str
) -> None:
    """
    Validate input parameters for s_plus function.

    Args:
        matrix1: First input matrix.
        matrix2: Second input matrix.
        weight_depop_matrix1: Depopularization weights for matrix1.
        weight_depop_matrix2: Depopularization weights for matrix2.
        k: Number of top items to keep.
        target_rows: Rows to compute.
        filter_cols: Columns to filter out.
        target_cols: Columns to target.
        verbose: Whether to show progress.
        format_output: Output format ('coo' or 'csr').

    Raises:
        TypeError: If matrix1 or matrix2 are not sparse matrices.
        ValueError: If any parameter has invalid value or incompatible dimensions.
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


@cython.boundscheck(False)
@cython.wraparound(False)
def csr_sum(
    matrix: sp.csr_matrix,
    int axis
) -> np.ndarray:
    """
    Sum CSR matrix along the specified axis using optimized NumPy operations.

    Uses np.add.reduceat for row sums (axis=1) and np.bincount for column sums (axis=0),
    which are significantly faster than manual loops.

    Args:
        matrix: CSR sparse matrix
        axis: 0 for column sums, 1 for row sums

    Returns:
        1D array of sums along the specified axis (float32)
    """
    # Extract CSR data as NumPy arrays
    cdef float[:] data = matrix.data.astype(np.float32, copy=False)
    cdef int[:] indices = matrix.indices.astype(np.int32, copy=False)
    cdef int[:] indptr = matrix.indptr.astype(np.int32, copy=False)
    cdef int n_cols = matrix.shape[1]

    if axis == 1:
        # Row sums via reduceat (assumes no empty rows)
        return np.add.reduceat(np.asarray(data), np.asarray(indptr[:-1])).astype(np.float32, copy=False)
    elif axis == 0:
        # Column sums via bincount (fast C path)
        # bincount returns float64; cast once to float32
        out64 = np.bincount(np.asarray(indices), weights=np.asarray(data), minlength=n_cols)
        return out64.astype(np.float32, copy=False)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")


def _build_squared_norms(
    matrix1: sp.csr_matrix,
    matrix2: sp.csr_matrix
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build squared norms for both matrices using NumPy-only operations.

    This computation is shared between Tversky and Cosine normalizations
    and should only be computed once when needed by either.

    Args:
        matrix1: First matrix (already in CSR format with zeros eliminated).
        matrix2: Second matrix (already in CSR format with zeros eliminated).

    Returns:
        Tuple of (m1_sq_norms, m2_sq_norms) as float32 arrays.
    """
    # Create temporary CSR matrices with squared data
    # This avoids scipy's .power() which creates unnecessary intermediate objects
    m1_squared = sp.csr_matrix(
        (np.square(matrix1.data, dtype=np.float32), matrix1.indices, matrix1.indptr),
        shape=matrix1.shape
    )
    m2_squared = sp.csr_matrix(
        (np.square(matrix2.data, dtype=np.float32), matrix2.indices, matrix2.indptr),
        shape=matrix2.shape
    )

    # Compute sums using custom csr_sum function
    cdef float[:] m1_sq_norms = csr_sum(m1_squared, axis=1)
    cdef float[:] m2_sq_norms = csr_sum(m2_squared, axis=0)

    return np.asarray(m1_sq_norms), np.asarray(m2_sq_norms)


def _build_tversky_normalization(
    m1_sq_norms: np.ndarray,
    m2_sq_norms: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Tversky normalization arrays based on pre-computed squared norms.

    Args:
        m1_sq_norms: Squared norms for matrix1.
        m2_sq_norms: Squared norms for matrix2.

    Returns:
        Tuple of (Xtversky, Ytversky) as float32 arrays.
    """
    return m1_sq_norms, m2_sq_norms


def _build_cosine_normalization(
    m1_sq_norms: np.ndarray,
    m2_sq_norms: np.ndarray,
    float c1,
    float c2,
    float additive_shrink
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Cosine normalization arrays based on squared norms.

    Args:
        m1_sq_norms: Squared norms for matrix1.
        m2_sq_norms: Squared norms for matrix2.
        c1: Power for Cosine normalization on matrix1.
        c2: Power for Cosine normalization on matrix2.
        additive_shrink: Additive shrinkage for Cosine normalization.

    Returns:
        Tuple of (Xcosine, Ycosine) as float32 arrays.
    """
    m1_sq = np.asarray(m1_sq_norms)
    m2_sq = np.asarray(m2_sq_norms)
    cdef float[:] Xcosine = np.power(m1_sq + additive_shrink, c1, dtype=np.float32)
    cdef float[:] Ycosine = np.power(m2_sq + additive_shrink, c2, dtype=np.float32)
    return Xcosine, Ycosine


def _build_depop_normalization(
    matrix1: sp.csr_matrix,
    matrix2: sp.csr_matrix,
    weight_spec1: Union[str, np.ndarray],
    weight_spec2: Union[str, np.ndarray],
    float p1,
    float p2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Depopularization normalization arrays.

    Args:
        matrix1: First matrix (already in CSR format with zeros eliminated).
        matrix2: Second matrix (already in CSR format with zeros eliminated).
        weight_spec1: Either 'none', 'sum', or an array of custom weights for matrix1.
        weight_spec2: Either 'none', 'sum', or an array of custom weights for matrix2.
        p1: Power to raise weights to for matrix1.
        p2: Power to raise weights to for matrix2.

    Returns:
        Tuple of (Xdepop, Ydepop) as float32 arrays.
    """
    cdef float[:] Xdepop
    cdef float[:] Ydepop

    # Build depopularization weights for matrix1
    if isinstance(weight_spec1, (list, np.ndarray)):
        Xdepop = np.power(weight_spec1, p1, dtype=np.float32)
    elif weight_spec1 == 'none':
        # Optimization: 1^p1 = 1 for any p1, so directly create ones array
        Xdepop = np.ones(matrix1.shape[0], dtype=np.float32)
    elif weight_spec1 == 'sum':
        Xdepop = np.power(csr_sum(matrix1, axis=1), p1, dtype=np.float32)
    else:
        raise ValueError(f"Invalid weight_spec1: {weight_spec1}")

    # Build depopularization weights for matrix2
    if isinstance(weight_spec2, (list, np.ndarray)):
        Ydepop = np.power(weight_spec2, p2, dtype=np.float32)
    elif weight_spec2 == 'none':
        # Optimization: 1^p2 = 1 for any p2, so directly create ones array
        Ydepop = np.ones(matrix2.shape[1], dtype=np.float32)
    elif weight_spec2 == 'sum':
        Ydepop = np.power(csr_sum(matrix2, axis=0), p2, dtype=np.float32)
    else:
        raise ValueError(f"Invalid weight_spec2: {weight_spec2}")

    return Xdepop, Ydepop


def _build_matrix_data(
    matrix1: sp.csr_matrix,
    matrix2: sp.csr_matrix,
    binary: bool
) -> None:
    """
    Convert matrix data to float32 and optionally binarize.

    In binary mode, all non-zero values become 1.0 (set theory).
    Otherwise, convert data to float32.

    Note: This function modifies matrices in-place. Caller should backup
    original data before calling if restoration is needed.

    Args:
        matrix1: First matrix (modified in-place).
        matrix2: Second matrix (modified in-place).
        binary: If True, use set theory (all values = 1.0).
    """
    # Modify matrices in-place
    if binary:
        # Set theory: all non-zero values become 1.0
        matrix1.data = np.ones(matrix1.data.shape[0], dtype=np.float32)
        matrix2.data = np.ones(matrix2.data.shape[0], dtype=np.float32)
    else:
        # Convert to float32 (copy if needed)
        matrix1.data = matrix1.data.astype(np.float32, copy=False)
        matrix2.data = matrix2.data.astype(np.float32, copy=False)


def _build_column_selector(
    cols: Optional[Union[List, np.ndarray, sp.spmatrix]]
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Build column selector (filter or target) for similarity computation.

    Handles three cases:
    - None or empty: MODE_NONE (0) - use all columns
    - List/array: MODE_ARRAY (1) - pre-filtered in Python, no C++ data needed
    - Sparse matrix: MODE_MATRIX (2) - row-specific column selections

    Note: Shape validation is done in validate_s_plus_inputs() before calling this function.

    Args:
        cols: Column selector specification.

    Returns:
        Tuple of (mode, indptr, indices) where:
        - mode: int (0=none, 1=array, 2=matrix)
        - indptr: int32 array (CSR indptr format, empty for NONE/ARRAY)
        - indices: int32 array (sorted column indices, empty for NONE/ARRAY)
    """
    cdef int mode
    cdef int[:] indptr
    cdef int[:] indices

    # Case 1: Sparse matrix with data - needs per-row C++ checks
    # Note: Shape validation is done in validate_s_plus_inputs()
    if sp.issparse(cols) and cols.data.shape[0] != 0:
        mode = MODE_MATRIX
        # Convert to CSR format and sort indices for binary search
        cols = cols.tocsr()
        cols.eliminate_zeros()
        cols.sort_indices()
        indptr = np.array(cols.indptr, dtype=np.int32)
        indices = np.array(cols.indices, dtype=np.int32)

    # Case 2: List or array with elements - will be pre-filtered in Python
    elif isinstance(cols, (list, np.ndarray)) and len(cols) != 0:
        mode = MODE_ARRAY
        # Return empty arrays - C++ won't use them (handles ARRAY same as NONE)
        indptr = np.array([], dtype=np.int32)
        indices = np.array([], dtype=np.int32)

    # Case 3: None, empty, or sparse matrix with no data
    else:
        mode = MODE_NONE
        indptr = np.array([], dtype=np.int32)
        indices = np.array([], dtype=np.int32)

    return mode, indptr, indices


def _compute_target_columns(
    filter_cols: Optional[Union[List, np.ndarray, sp.spmatrix]],
    target_cols: Optional[Union[List, np.ndarray, sp.spmatrix]],
    int n_cols
) -> np.ndarray:
    """
    Compute the target columns based on filter_cols and target_cols specifications.

    This function determines which columns should be included in the computation
    by combining filter and target specifications.

    Args:
        filter_cols: Columns to exclude (can be None, array/list, or sparse matrix).
        target_cols: Columns to include (can be None, array/list, or sparse matrix).
        n_cols: Total number of columns in matrix2.

    Returns:
        Array of target column indices (sorted, int32).

    Rules:
        - If both empty/None: return all columns [0, 1, ..., n_cols-1]
        - If both are matrices: return all columns (can't pre-filter)
        - If filter_cols is array: return all columns except filtered ones
        - If target_cols is array: return only target columns
        - If both are arrays: return (target_cols - filter_cols)
    """
    cdef bint filter_is_empty = filter_cols is None or (isinstance(filter_cols, (list, np.ndarray)) and len(filter_cols) == 0)
    cdef bint target_is_empty = target_cols is None or (isinstance(target_cols, (list, np.ndarray)) and len(target_cols) == 0)
    cdef bint filter_is_matrix = sp.issparse(filter_cols) and filter_cols.data.shape[0] != 0
    cdef bint target_is_matrix = sp.issparse(target_cols) and target_cols.data.shape[0] != 0

    # Case 1: Both empty/None - return all columns
    if filter_is_empty and target_is_empty:
        return np.arange(n_cols, dtype=np.int32)

    # Case 2: Both are matrices - can't pre-filter (different columns per row)
    if filter_is_matrix and target_is_matrix:
        return np.arange(n_cols, dtype=np.int32)

    # Case 3: Only one is matrix, other is empty - return all columns
    if (filter_is_matrix and target_is_empty) or (target_is_matrix and filter_is_empty):
        return np.arange(n_cols, dtype=np.int32)

    # Case 4: At least one is array-based - compute valid columns
    if not target_is_empty and not target_is_matrix:
        mask = np.zeros(n_cols, dtype=bool)
        target_idx = np.asarray(target_cols, dtype=np.int32)
        target_idx = target_idx[(target_idx >= 0) & (target_idx < n_cols)]
        mask[target_idx] = True
    else:
        mask = np.ones(n_cols, dtype=bool)

    if not filter_is_empty and not filter_is_matrix:
        filter_idx = np.asarray(filter_cols, dtype=np.int32)
        filter_idx = filter_idx[(filter_idx >= 0) & (filter_idx < n_cols)]
        mask[filter_idx] = False

    return np.flatnonzero(mask).astype(np.int32, copy=False)


def _filter_matrix_columns(
    matrix: sp.csr_matrix,
    target_cols: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter CSR matrix to keep only target columns while preserving original column indices.

    This function removes all entries from the matrix whose column index is not in
    target_cols, but keeps the original column indices for remaining entries.

    Args:
        matrix: CSR matrix to filter.
        target_cols: Array of column indices to keep.

    Returns:
        Tuple of (data, indices, indptr) as float32/int32 arrays ready for Cython.
    """
    # Build target mask (drop out-of-range indices)
    cdef int n_rows = matrix.shape[0]
    cdef int n_cols = matrix.shape[1]
    target_idx = np.asarray(target_cols, dtype=np.int32)
    target_idx = target_idx[(target_idx >= 0) & (target_idx < n_cols)]
    mask = np.zeros(n_cols, dtype=np.uint8)
    mask[target_idx] = 1

    # Get matrix attributes with proper types
    cdef int[:] indptr = matrix.indptr
    cdef int[:] indices = matrix.indices
    cdef float[:] data = matrix.data.astype(np.float32, copy=False)
    cdef unsigned char[:] mask_view = mask
    cdef int row, start, end, i, col_idx
    cdef int nnz = 0

    # First pass: count kept entries and build indptr
    cdef int[:] new_indptr = np.empty(n_rows + 1, dtype=np.int32)
    new_indptr[0] = 0
    for row in range(n_rows):
        start = indptr[row]
        end = indptr[row + 1]
        for i in range(start, end):
            col_idx = indices[i]
            if mask_view[col_idx]:
                nnz += 1
        new_indptr[row + 1] = nnz

    # Allocate output arrays
    cdef float[:] new_data = np.empty(nnz, dtype=np.float32)
    cdef int[:] new_indices = np.empty(nnz, dtype=np.int32)

    # Second pass: fill data and indices
    cdef int pos = 0
    for row in range(n_rows):
        start = indptr[row]
        end = indptr[row + 1]
        for i in range(start, end):
            col_idx = indices[i]
            if mask_view[col_idx]:
                new_data[pos] = data[i]
                new_indices[pos] = col_idx
                pos += 1

    # Return arrays in correct format for Cython
    return (
        np.asarray(new_data),
        np.asarray(new_indices),
        np.asarray(new_indptr)
    )
