# cython: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

"""
    Utility functions for similaripy cython code
"""

import cython
import numpy as np
import scipy.sparse as sp
from typing import Optional, Union, Tuple

cdef extern from "coo_to_csr.h" nogil:
    void coo32_to_csr64(int n_row, int n_col, long nnz, int Ai[], int Aj[], float Ax[], long Bp[], long Bj[], float Bx[])
    void coo32_to_csr32(int n_row, int n_col, int nnz, int Ai[], int Aj[], float Ax[], int Bp[], int Bj[], float Bx[])

cdef extern from "omp.h":
    int omp_get_max_threads()


def get_num_threads() -> int:
    """
    Get the maximum number of OpenMP threads available.

    Returns:
        Maximum number of threads that OpenMP can use.
    """
    return omp_get_max_threads()


def get_index_dtype(
    arrays: Union[Tuple, np.ndarray] = (),
    maxval: Optional[Union[int, np.integer]] = None,
    check_contents: bool = False
) -> type:
    """
    Determine a suitable index data type that can hold the data in the arrays.

    Args:
        arrays: Input integer arrays to analyze.
        maxval: Optional maximum value to check against int32 limits.
        check_contents: Whether to check array contents for min/max values.

    Returns:
        Either np.int32 or np.int64 based on the data requirements.
    """
    # not using intc directly due to misinteractions with pythran
    if np.intc().itemsize != 4:
        return np.int64

    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    if maxval is not None:
        maxval = np.int64(maxval)
        if maxval > int32max:
            return np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if not np.can_cast(arr.dtype, np.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed  
                        continue
            return np.int64
    return np.int32


def build_coo_matrix(
    int[:] rows,
    int[:] cols,
    float[:] values,
    int item_count,
    int user_count
) -> sp.coo_matrix:
    """
    Build a sparse matrix in COO (Coordinate) format.

    Args:
        rows: Row indices from computation.
        cols: Column indices from computation.
        values: Values from computation.
        item_count: Number of rows in output matrix.
        user_count: Number of columns in output matrix.

    Returns:
        The result matrix in COO format.
    """
    res = sp.coo_matrix((values, (rows, cols)), shape=(item_count, user_count), dtype=np.float32)
    del values, rows, cols
    return res


def build_csr_matrix_32(
    int[:] rows,
    int[:] cols,
    float[:] values,
    int item_count,
    int user_count
) -> sp.csr_matrix:
    """
    Build a sparse matrix in CSR (Compressed Sparse Row) format using 32-bit indices.

    The resulting matrix has zeros eliminated.

    Args:
        rows: Row indices from computation.
        cols: Column indices from computation.
        values: Values from computation.
        item_count: Number of rows in output matrix.
        user_count: Number of columns in output matrix.

    Returns:
        The result matrix in CSR format with 32-bit indices and zeros eliminated.
    """
    cdef int M = item_count
    cdef int N = user_count
    cdef int nnz = len(values)
    cdef float[:] data
    cdef int[:] indices32, indptr32

    indptr32 = np.empty(M + 1, dtype=np.int32)
    indices32 = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    if nnz != 0:
        coo32_to_csr32(M, N, nnz, &rows[0], &cols[0], &values[0], &indptr32[0], &indices32[0], &data[0])
    del values, rows, cols
    res = sp.csr_matrix((data, indices32, indptr32), shape=(item_count, user_count), dtype=np.float32)

    return res


def build_csr_matrix_64(
    int[:] rows,
    int[:] cols,
    float[:] values,
    int item_count,
    int user_count
) -> sp.csr_matrix:
    """
    Build a sparse matrix in CSR (Compressed Sparse Row) format using 64-bit indices.

    The resulting matrix has zeros eliminated.

    Args:
        rows: Row indices from computation.
        cols: Column indices from computation.
        values: Values from computation.
        item_count: Number of rows in output matrix.
        user_count: Number of columns in output matrix.

    Returns:
        The result matrix in CSR format with 64-bit indices and zeros eliminated.
    """
    cdef int M = item_count
    cdef int N = user_count
    cdef long nnz = len(values)
    cdef float[:] data
    cdef long[:] indices64, indptr64

    indptr64 = np.empty(M + 1, dtype=np.int64)
    indices64 = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float32)
    if nnz != 0:
        coo32_to_csr64(M, N, nnz, &rows[0], &cols[0], &values[0], &indptr64[0], &indices64[0], &data[0])
    del values, rows, cols
    res = sp.csr_matrix((data, indices64, indptr64), shape=(item_count, user_count), dtype=np.float32)

    return res


def build_csr_matrix(
    int[:] rows,
    int[:] cols,
    float[:] values,
    int item_count,
    int user_count
) -> sp.csr_matrix:
    """
    Build a sparse matrix in CSR (Compressed Sparse Row) format.

    Automatically determines whether to use 32-bit or 64-bit indices based on
    matrix size and dispatches to the appropriate specialized function.
    The resulting matrix has zeros eliminated.

    Args:
        rows: Row indices from computation.
        cols: Column indices from computation.
        values: Values from computation.
        item_count: Number of rows in output matrix.
        user_count: Number of columns in output matrix.

    Returns:
        The result matrix in CSR format with zeros eliminated.
    """
    cdef int nnz = len(values)

    # Determine appropriate index dtype (32/64 bit) based on total entries and max value
    idx_dtype = get_index_dtype(maxval=max(nnz, user_count))

    if idx_dtype == np.int32:
        return build_csr_matrix_32(rows, cols, values, item_count, user_count)
    else:  # idx_dtype == np.int64:
        return build_csr_matrix_64(rows, cols, values, item_count, user_count)
