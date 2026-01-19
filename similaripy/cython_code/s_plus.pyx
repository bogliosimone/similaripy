# cython: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

"""
    s_plus: top-K similarity search between rows of two sparse matrices
"""

import cython
import numpy as np
import scipy.sparse as sp
from typing import Optional, Union, Literal

from .utils import (
    build_coo_matrix,
    build_csr_matrix,
    get_num_threads
)
from .s_plus_utils import (
    validate_s_plus_inputs,
    _build_matrix_data,
    _build_squared_norms,
    _build_tversky_normalization,
    _build_cosine_normalization,
    _build_depop_normalization,
    _build_column_selector,
    _compute_target_columns,
    _filter_matrix_columns
)

from cython import float

from libcpp cimport bool
from libcpp.string cimport string

# Progress bar configuration constants
cdef int PROGRESS_BAR_REFRESH_RATE = 3  # Refresh rate in Hz (updates per second)
cdef int PROGRESS_BAR_WIDTH = 25        # Width of the progress bar in characters

# Column selector mode constants
cdef int MODE_NONE = 0  # No filtering/targeting (use all columns)
cdef int MODE_ARRAY = 1  # Column selector is an array/list
cdef int MODE_MATRIX = 2  # Column selector is a sparse matrix

cdef extern from "progress_bar.h" namespace "progress" nogil:
    cdef cppclass ProgressBar:
        ProgressBar(int total, bool disabled, int max_refresh_rate, int bar_width) except +
        void set_description(const string& desc) except +
        void update(int n) except +
        void close(const string& final_desc) except +

cdef extern from "s_plus.h" namespace "s_plus" nogil:
    cdef void compute_similarities_parallel[Index, Value](
        Index n_targets,
        const Index* targets,
        const Value* m1_data,
        const Index* m1_indices,
        const Index* m1_indptr,
        const Value* m2_data,
        const Index* m2_indices,
        const Index* m2_indptr,
        const Value* Xtversky,
        const Value* Ytversky,
        const Value* Xcosine,
        const Value* Ycosine,
        const Value* Xdepop,
        const Value* Ydepop,
        Value a1,
        Value l1,
        Value l2,
        Value l3,
        Value t1,
        Value t2,
        Value stabilized_shrink,
        Value bayesian_shrink,
        Value threshold,
        Index k,
        Index n_output_cols,
        Index filter_mode,
        Index* filter_m_indptr,
        Index* filter_m_indices,
        Index target_col_mode,
        Index* target_col_m_indptr,
        Index* target_col_m_indices,
        Index* rows,
        Index* cols,
        Value* values,
        ProgressBar* progress,
        int num_threads
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def s_plus(
    matrix1: sp.csr_matrix,
    matrix2: Optional[sp.csr_matrix] = None,
    weight_depop_matrix1: Union[str, np.ndarray] = 'none',
    weight_depop_matrix2: Union[str, np.ndarray] = 'none',
    float p1 = 0,
    float p2 = 0,
    float a1 = 1,
    float l1 = 0,
    float l2 = 0,
    float l3 = 0,
    float t1 = 1,
    float t2 = 1,
    float c1 = 0.5,
    float c2 = 0.5,
    unsigned int k = 100,
    float stabilized_shrink = 0,
    float bayesian_shrink = 0,
    float additive_shrink = 0,
    float threshold = 0,
    binary: bool = False,
    target_rows: Optional[Union[list, np.ndarray]] = None,
    filter_cols: Optional[Union[list, np.ndarray, sp.spmatrix]] = None,
    target_cols: Optional[Union[list, np.ndarray, sp.spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    int num_threads = 0
) -> Union[sp.csr_matrix, sp.coo_matrix]:
    """
    Compute top-K similarity between rows of two sparse matrices using the S_Plus algorithm.

    The S_Plus algorithm combines multiple similarity metrics (Tversky, Cosine, Depopularization)
    with configurable weights and supports various normalization strategies.

    Args:
        matrix1: First sparse matrix (typically item-user interactions).
        matrix2: Optional second matrix. If None, uses matrix1.T (self-similarity).
        weight_depop_matrix1: Depopularization weights for matrix1: 'none', 'sum', or custom array.
        weight_depop_matrix2: Depopularization weights for matrix2: 'none', 'sum', or custom array.
        p1: Power for depopularization weights on matrix1.
        p2: Power for depopularization weights on matrix2.
        a1: Power applied to dot product values.
        l1: Weight for Tversky similarity term.
        l2: Weight for Cosine similarity term.
        l3: Weight for Depopularization term.
        t1: Tversky parameter for matrix1.
        t2: Tversky parameter for matrix2.
        c1: Cosine power exponent for matrix1.
        c2: Cosine power exponent for matrix2.
        k: Number of top similar items to keep per row.
        stabilized_shrink: Stabilized shrinkage parameter.
        bayesian_shrink: Bayesian shrinkage parameter.
        additive_shrink: Additive shrinkage for cosine normalization.
        threshold: Minimum similarity threshold.
        binary: If True, treat all non-zero values as 1.0 (set theory).
        target_rows: Specific rows to compute. If None, compute all rows.
        filter_cols: Columns to exclude from results (e.g., already seen items).
        target_cols: Columns to include in results (only compute these).
        verbose: Show progress bar during computation.
        format_output: Output matrix format: 'csr' or 'coo'.
        num_threads: Number of OpenMP threads (0 = use all available cores).

    Returns:
        A sparse matrix of shape (n_rows, n_cols) in the specified format,
        containing the top-k similarity scores.
    """

    # if receive only matrix1 in input
    if matrix2 is None:
        matrix2 = matrix1.T

    # Validate all inputs
    validate_s_plus_inputs(
        matrix1=matrix1,
        matrix2=matrix2,
        weight_depop_matrix1=weight_depop_matrix1,
        weight_depop_matrix2=weight_depop_matrix2,
        k=k,
        target_rows=target_rows,
        filter_cols=filter_cols,
        target_cols=target_cols,
        verbose=verbose,
        format_output=format_output
    )

    # do not allocate unnecessary space
    if k > matrix2.shape[1]:
        k = matrix2.shape[1]

    # build target rows (only the row that must be computed)
    if target_rows is None:
        targets_arr = np.arange(matrix1.shape[0], dtype=np.int32)
    else:
        targets_arr = np.ascontiguousarray(np.asarray(target_rows, dtype=np.int32))
    cdef int[:] targets = targets_arr
    cdef int n_targets = targets.shape[0]

    # Initialize progress bar
    cdef ProgressBar * progress = NULL
    if verbose:
        progress = new ProgressBar(n_targets, False, PROGRESS_BAR_REFRESH_RATE, PROGRESS_BAR_WIDTH)
        progress.set_description(b'Preprocessing')

    # be sure to use csr matrixes
    matrix1 = matrix1.tocsr()
    matrix2 = matrix2.tocsr()

    # eliminates zeros to avoid 0 division and get right values when using the binary flag (also speed up the computation)
    # note: this is an in-place operation implemented for csr matrix in the sparse package
    matrix1.eliminate_zeros()
    matrix2.eliminate_zeros()

    # useful variables
    cdef int n_output_rows = matrix1.shape[0]
    cdef int n_output_cols = matrix2.shape[1]

    ### START PREPROCESSING ###

    # Backup original data before modifications
    original_m1_data = matrix1.data
    original_m2_data = matrix2.data

    # Build matrix data (handle binary mode and float32 conversion)
    _build_matrix_data(matrix1, matrix2, binary)

    # build the data terms (avoid copy if already float32)
    cdef float[:] m1_data = matrix1.data.astype(np.float32, copy=False)
    cdef float[:] m2_data = matrix2.data.astype(np.float32, copy=False)

    # build indices and indptrs (avoid copy if already int32)
    cdef int[:] m1_indptr = matrix1.indptr.astype(np.int32, copy=False)
    cdef int[:] m1_indices = matrix1.indices.astype(np.int32, copy=False)
    cdef int[:] m2_indptr = matrix2.indptr.astype(np.int32, copy=False)
    cdef int[:] m2_indices = matrix2.indices.astype(np.int32, copy=False)

    # build normalization terms for tversky, cosine and depop
    # initialize all as empty arrays and fill only if needed
    cdef float[:] empty = np.array([], dtype=np.float32)
    cdef float[:] Xtversky = empty
    cdef float[:] Ytversky = empty
    cdef float[:] Xcosine = empty
    cdef float[:] Ycosine = empty
    cdef float[:] Xdepop = empty
    cdef float[:] Ydepop = empty
    cdef float[:] m1_sq_norms = empty
    cdef float[:] m2_sq_norms = empty

    # Compute squared norms once if needed by either Tversky or Cosine (avoid redundant computation)
    if l1 != 0 or l2 != 0:
        m1_sq_norms, m2_sq_norms = _build_squared_norms(matrix1, matrix2)

    if l1 != 0:
        Xtversky, Ytversky = _build_tversky_normalization(m1_sq_norms, m2_sq_norms)

    if l2 != 0:
        Xcosine, Ycosine = _build_cosine_normalization(m1_sq_norms, m2_sq_norms, c1, c2, additive_shrink)

    if l3 != 0:
        Xdepop, Ydepop = _build_depop_normalization(matrix1, matrix2, weight_depop_matrix1, weight_depop_matrix2, p1, p2)

    # restore original data terms
    matrix1.data, matrix2.data = original_m1_data, original_m2_data

    ### END OF PREPROCESSING ###

    # Prepare filter and target column selectors
    cdef int filter_col_mode
    cdef int[:] filter_m_indptr
    cdef int[:] filter_m_indices
    cdef int target_col_mode
    cdef int[:] target_m_indptr
    cdef int[:] target_m_indices

    filter_col_mode, filter_m_indptr, filter_m_indices = _build_column_selector(filter_cols)
    target_col_mode, target_m_indptr, target_m_indices = _build_column_selector(target_cols)

    # Pre-filter matrix2 if we have array-based filter/target columns
    # C++ will handle SELECTION_ARRAY same as SELECTION_NONE (no extra checks needed)
    cdef int[:] target_columns
    if filter_col_mode == MODE_ARRAY or target_col_mode == MODE_ARRAY:
        # Compute which columns to keep
        target_columns = _compute_target_columns(filter_cols, target_cols, n_output_cols)

        # Filter matrix2 and get data arrays directly in correct format
        m2_data, m2_indices, m2_indptr = _filter_matrix_columns(matrix2, target_columns)

    ### START COMPUTATION ###

    # Pre-allocate output arrays
    cdef float[:] values = np.zeros(n_targets * k, dtype=np.float32)
    cdef int[:] rows = np.zeros(n_targets * k, dtype=np.int32)
    cdef int[:] cols = np.zeros(n_targets * k, dtype=np.int32)

    if progress != NULL:
        progress.set_description(b'Computing')

    # Call C++ parallel computation function
    with nogil:
        compute_similarities_parallel[int, float](
            n_targets,
            &targets[0],
            &m1_data[0], &m1_indices[0], &m1_indptr[0],
            &m2_data[0], &m2_indices[0], &m2_indptr[0],
            &Xtversky[0], &Ytversky[0],
            &Xcosine[0], &Ycosine[0],
            &Xdepop[0], &Ydepop[0],
            a1,
            l1, l2, l3,
            t1, t2,
            stabilized_shrink,
            bayesian_shrink,
            threshold,
            k,
            n_output_cols,
            filter_col_mode,
            &filter_m_indptr[0], &filter_m_indices[0],
            target_col_mode,
            &target_m_indptr[0], &target_m_indices[0],
            &rows[0], &cols[0], &values[0],
            progress,
            num_threads
        )

    # Deallocate intermediate memory
    del Xcosine, Ycosine, Xtversky, Ytversky, Xdepop, Ydepop
    del m1_data, m1_indices, m1_indptr
    del m2_data, m2_indices, m2_indptr
    del targets

    ### BUILD OUTPUT MATRIX ###

    if progress != NULL:
        progress.set_description(f'Building {format_output} matrix'.encode())

    # Build result in requested format
    if format_output == 'coo':
        res = build_coo_matrix(
            rows=rows,
            cols=cols,
            values=values,
            item_count=n_output_rows,
            user_count=n_output_cols
        )
    else:
        res = build_csr_matrix(
            rows=rows,
            cols=cols,
            values=values,
            item_count=n_output_rows,
            user_count=n_output_cols
        )
        if progress != NULL:
            progress.set_description(b'Removing zeros')
        res.eliminate_zeros()

    # Finalize and cleanup
    if progress != NULL:
        progress.close(b'Done')
        del progress

    return res
