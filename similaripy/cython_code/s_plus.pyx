# cython: language_level=3
# cython: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# distutils: sources = s_plus.cpp, coo_to_csr.cpp

"""
    s_plus: top-K similarity search between rows of two sparse matrices
"""

import cython
import numpy as np
import scipy.sparse as sp
import tqdm
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
    _build_column_selector
)

from cython.operator import dereference
from cython.parallel import parallel, prange
from cython import float, address

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool

# Progress bar configuration constants
cdef int PROGRESS_BAR_THRESHOLD = 5000  # Show progress bar only for large computations
cdef int PROGRESS_UPDATE_FREQUENCY = 500  # Update progress bar every N rows

# Column selector mode constants
cdef int MODE_NONE = 0  # No filtering/targeting (use all columns)
cdef int MODE_ARRAY = 1  # Column selector is an array/list
cdef int MODE_MATRIX = 2  # Column selector is a sparse matrix

cdef extern from "s_plus.h" namespace "s_plus" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier( Index column_count, 
                                Value * Xtversky, Value * Ytversky,
                                Value * Xcosine, Value * Ycosine,
                                Value * Xdepop, Value * Ydepop,
                                Value a1,
                                Value l1, Value l2, Value l3,
                                Value t1, Value t2,
                                Value c1, Value c2,
                                Value stabilized_shrink,
                                Value bayesian_shrink,
                                Value threshold,
                                Index filter_mode,
                                Index * filter_m_indptr,
                                Index * filter_m_indices,
                                Index target_col_mode,
                                Index * target_col_m_indptr,
                                Index * target_col_m_indices
                                )
        void add(Index index, Value value)
        void setIndexRow(Index index)
        void foreach[Function](Function & f)

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
        target_rows = np.arange(matrix1.shape[0], dtype=np.int32)
    cdef int[:] targets = np.array(target_rows, dtype=np.int32)
    cdef int n_targets = targets.shape[0]

    # start progress bar
    progress = tqdm.tqdm(total=n_targets, disable=not verbose)
    progress.desc = 'Preprocessing'
    progress.refresh()

    # be sure to use csr matrixes
    matrix1 = matrix1.tocsr()
    matrix2 = matrix2.tocsr()

    # eliminates zeros to avoid 0 division and get right values when using the binary flag (also speed up the computation)
    # note: this is an in-place operation implemented for csr matrix in the sparse package
    matrix1.eliminate_zeros()
    matrix2.eliminate_zeros()

    # useful variables
    cdef int item_count = matrix1.shape[0]
    cdef int user_count = matrix2.shape[1]
    cdef int i, u, t, index1, index2
    cdef long index3
    cdef float v1

    ### START PREPROCESSING ###

    # Build matrix data (handle binary mode and float32 conversion)
    old_m1_data, old_m2_data = _build_matrix_data(matrix1, matrix2, binary)

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
    matrix1.data, matrix2.data = old_m1_data, old_m2_data

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

    # set progress bar
    cdef int counter = 0
    cdef int * counter_add = address(counter)
    cdef int verb
    cdef int progress_update_interval
    if n_targets <= PROGRESS_BAR_THRESHOLD or not verbose:
        verb = 0
        progress_update_interval = 1  # Not used but must be defined
    else:
        verb = 1
        progress_update_interval = max(1, n_targets // PROGRESS_UPDATE_FREQUENCY)

    
    # structures for multiplications
    cdef SparseMatrixMultiplier[int, float] * neighbours
    cdef TopK[int, float] * topk
    cdef pair[float, int] result

    # triples of output
    cdef float[:] values = np.zeros(n_targets * k, dtype=np.float32)
    cdef int[:] rows = np.zeros(n_targets * k, dtype=np.int32)
    cdef int[:] cols = np.zeros(n_targets * k, dtype=np.int32)

    progress.desc = 'Allocate memory per threads'
    progress.refresh()
    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, float](user_count,
                                                            &Xtversky[0], &Ytversky[0],
                                                            &Xcosine[0], &Ycosine[0],
                                                            &Xdepop[0], &Ydepop[0],
                                                            a1,
                                                            l1, l2, l3,
                                                            t1, t2,
                                                            c1, c2,
                                                            stabilized_shrink, 
                                                            bayesian_shrink,
                                                            threshold,
                                                            filter_col_mode, 
                                                            &filter_m_indptr[0], &filter_m_indices[0],
                                                            target_col_mode, 
                                                            &target_m_indptr[0], &target_m_indices[0],
                                                            )
        topk = new TopK[int, float](k)
        try:
            for i in prange(n_targets, schedule='dynamic'):
                # progress bar (note: update once per PROGRESS_UPDATE_FREQUENCY rows or with big matrix taking gil at each cycle destroy the performance)
                if verb == 1:
                    # here, without gil, we can get WAR (Write-After-Read), WAW (Write-After-Write),
                    # RAW (Read-After-Write) race conditions, it's not important as it's just a counter for the progress bar
                    counter_add[0] = counter_add[0] + 1
                    if counter_add[0] % progress_update_interval == 0:
                        with gil:
                            progress.desc = 'Computing'
                            progress.n = counter_add[0]
                            progress.refresh()
                # compute row
                t = targets[i]
                neighbours.setIndexRow(t)
                for index1 in range(m1_indptr[t], m1_indptr[t+1]):
                    u = m1_indices[index1]
                    v1 = m1_data[index1]
                    for index2 in range(m2_indptr[u], m2_indptr[u+1]):
                        neighbours.add(m2_indices[index2], m2_data[index2] * v1)
                topk.results.clear()
                neighbours.foreach(dereference(topk))
                index3 = k * i
                for result in topk.results:
                    rows[index3] = t
                    cols[index3] = result.second
                    values[index3] = result.first
                    index3 = index3 + 1

        finally:
            del neighbours
            del topk

    progress.n = n_targets
    progress.refresh()

    # deallocate memory
    del Xcosine, Ycosine, Xtversky, Ytversky, Xdepop, Ydepop
    del m1_data, m1_indices, m1_indptr
    del m2_data, m2_indices, m2_indptr
    del targets

    progress.desc = f'Build {format_output} matrix'
    progress.refresh()

    # build result in coo or csr format
    if format_output == 'coo':
        res = build_coo_matrix(
            rows=rows,
            cols=cols,
            values=values,
            item_count=item_count,
            user_count=user_count
        )
    else:
        res = build_csr_matrix(
            rows=rows,
            cols=cols,
            values=values,
            item_count=item_count,
            user_count=user_count
        )
        progress.desc = 'Remove zeros'
        progress.refresh()
        res.eliminate_zeros()

    # finally update progress bar and return the result matrix
    progress.desc = 'Done'
    progress.refresh()
    progress.close()
    return res
