"""
    author: Simone Boglio
    mail: bogliosimone@gmail.com
"""

import cython
import numpy as np
import scipy.sparse as sp
import tqdm

from scipy.sparse.sputils import get_index_dtype

from cython.operator import dereference
from cython.parallel import parallel, prange
from cython import float, address

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool

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
                                Value shrink, Value threshold,
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

cdef extern from "coo_to_csr.h" nogil:
    void coo32_to_csr64(int n_row,int n_col,long nnz,int Ai[],int Aj[],float Ax[],long Bp[],long Bj[],float Bx[])
    void coo32_to_csr32(int n_row,int n_col,int nnz,int Ai[],int Aj[],float Ax[],int Bp[],int Bj[],float Bx[])


@cython.boundscheck(False)
@cython.wraparound(False)
def s_plus(
    matrix1, matrix2=None,
    weight_depop_matrix1='none' , weight_depop_matrix2='none',
    float p1=0, float p2=0, 
    float a1=1,
    float l1=0, float l2=0, float l3=0,
    float t1=1, float t2=1,
    float c1=0.5,float c2=0.5,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_rows=None,
    filter_cols=None,
    target_cols=None,
    verbose=True,
    format_output='csr',
    int num_threads=0):  

    assert sp.issparse(matrix1), 'matrix m1 must be a sparse matrix'
    # if receive only matrix1 in input
    if matrix2 is None:
        matrix2=matrix1.T
    assert sp.issparse(matrix2), 'matrix m2 must be a sparse matrix'

    # check that all parameters are consistent
    assert matrix1.shape[1]==matrix2.shape[0], 'error shape matrixs'
    assert k >= 1, 'k must be >=1'
    assert len(weight_depop_matrix1)==matrix1.shape[0] or weight_depop_matrix1 in ('none','sum'), 'error format weighs_depop matrix1'
    assert len(weight_depop_matrix2)==matrix2.shape[1] or weight_depop_matrix2 in ('none','sum'), 'error format weighs_depop matrix2'
    assert target_rows is None or len(target_rows)<=matrix1.shape[0], 'error target rows'
    assert filter_cols is None or sp.issparse(filter_cols) or isinstance(filter_cols,(list,np.ndarray)), 'error format filter_cols'
    assert target_cols is None or sp.issparse(target_cols) or isinstance(target_cols,(list,np.ndarray)), 'error format target_cols' 
    assert verbose==True or verbose==False, 'verbose must be boolean'
    assert format_output=='coo' or format_output=='csr', 'output format must be \'coo\' or \'csr\''

    # do not allocate unecessary space
    if k > matrix2.shape[1]:
        k = matrix2.shape[1]

    # build target rows (only the row that must be computed)
    if target_rows is None:
        target_rows=np.arange(matrix1.shape[0],dtype=np.int32)
    cdef int[:] targets = np.array(target_rows,dtype=np.int32)
    cdef int n_targets = targets.shape[0]

    # start progress bar
    progress = tqdm.tqdm(total=n_targets, disable=not verbose)
    progress.desc = 'Preprocessing'
    progress.refresh()

    # be sure to use csr matrixes
    matrix1 = matrix1.tocsr()
    matrix2 = matrix2.tocsr()

    # eliminates zeros to avoid 0 division and get right values if use binary flag (also speed up the computation)
    # note: is an implace operation implemented for csr matrix in the sparse package
    matrix1.eliminate_zeros()
    matrix2.eliminate_zeros()

    # usefull variables
    cdef int item_count = matrix1.shape[0]
    cdef int user_count = matrix2.shape[1]
    cdef int i, u, t, index1, index2
    cdef int norm, depop
    cdef long index3
    cdef float v1

    ### START PREPROCESSING ###

    # save original data
    old_m1_data, old_m2_data = matrix1.data, matrix2.data

    # if binary use set theory otherwise copy data and use float32
    if binary:
        matrix1.data, matrix2.data = np.ones(matrix1.data.shape[0], dtype= np.float32), np.ones(matrix2.data.shape[0], dtype=np.float32)
    else:
        matrix1.data, matrix2.data = np.array(matrix1.data, dtype=np.float32), np.array(matrix2.data, dtype=np.float32)

    # build popularities array
    if l3!=0:
        if isinstance(weight_depop_matrix1,(list,np.ndarray)): 
            weight_depop_matrix1 = np.power(weight_depop_matrix1, p1, dtype=np.float32)    
        elif weight_depop_matrix1=='none':
            weight_depop_matrix1 = np.ones(matrix1.shape[0], dtype=np.float32)
        elif weight_depop_matrix1 == 'sum':
            weight_depop_matrix1 = np.power(np.array(matrix1.sum(axis = 1).A1, dtype=np.float32), p1, dtype=np.float32)
        
        if isinstance(weight_depop_matrix1,(list,np.ndarray)): 
            weight_depop_matrix2 = np.power(weight_depop_matrix2, p2, dtype=np.float32)    
        elif weight_depop_matrix2=='none':
            weight_depop_matrix2 = np.power(np.ones(matrix1.shape[1]), p2, dtype=np.float32)  
        elif weight_depop_matrix2 == 'sum':
            weight_depop_matrix2 = np.power(np.array(matrix2.sum(axis = 0).A1, dtype=np.float32), p2, dtype=np.float32)

    # build the data terms 
    cdef float[:] m1_data = np.array(matrix1.data, dtype=np.float32)
    cdef float[:] m2_data = np.array(matrix2.data, dtype=np.float32)

    # build indices and indptrs
    cdef int[:] m1_indptr = np.array(matrix1.indptr, dtype=np.int32), m1_indices = np.array(matrix1.indices, dtype=np.int32)
    cdef int[:] m2_indptr = np.array(matrix2.indptr, dtype=np.int32), m2_indices = np.array(matrix2.indices, dtype=np.int32)

    # build normalization terms for tversky, cosine and depop
    cdef float[:] Xtversky
    cdef float[:] Ytversky
    cdef float[:] Xcosine
    cdef float[:] Ycosine 
    cdef float[:] Xdepop
    cdef float[:] Ydepop 

    if l1!=0:
        Xtversky = np.array(matrix1.power(2).sum(axis = 1).A1, dtype=np.float32)
        Ytversky = np.array(matrix2.power(2).sum(axis = 0).A1, dtype=np.float32)
    else:
        Xtversky = np.array([],dtype=np.float32)
        Ytversky = np.array([],dtype=np.float32)

    if l2!=0:
        Xcosine = np.power(matrix1.power(2).sum(axis = 1).A1, c1, dtype=np.float32)
        Ycosine = np.power(matrix2.power(2).sum(axis = 0).A1, c2, dtype=np.float32)
    else:
        Xcosine = np.array([],dtype=np.float32)
        Ycosine = np.array([],dtype=np.float32)
    
    if l3!=0:
        Xdepop = np.array(weight_depop_matrix1,dtype=np.float32)
        Ydepop = np.array(weight_depop_matrix2,dtype=np.float32)
    else:
        Xdepop = np.array([],dtype=np.float32)
        Ydepop = np.array([],dtype=np.float32)

    # restore original data terms
    matrix1.data, matrix2.data = old_m1_data, old_m2_data

    ### END OF PREPROCESSING ###

    # filter col matrix
    # mode: 0 no filter, 1 filter array, 2 filter matrix
    cdef int filter_col_mode
    cdef int[:] filter_m_indptr
    cdef int[:] filter_m_indices

    if sp.issparse(filter_cols) and filter_cols.data.shape[0] != 0:
        assert filter_cols.shape == (item_count, user_count), 'shape filter_cols matrix not correct'
        filter_col_mode = 2
        # build indices and indptrs and sort indices since we will use binary search
        filter_cols = filter_cols.tocsr()
        filter_cols.eliminate_zeros()
        filter_cols.sort_indices()
        filter_m_indptr = np.array(filter_cols.indptr, dtype=np.int32)
        filter_m_indices = np.array(filter_cols.indices, dtype=np.int32)
    elif isinstance(filter_cols, (list, np.ndarray)) and len(filter_cols) != 0:
        filter_col_mode = 1
        # sort array since we will use binary search
        filter_m_indptr = np.array([0,len(filter_cols)], dtype=np.int32)
        filter_m_indices = np.array(np.sort(filter_cols), dtype=np.int32)
    else:
        # filter cols is empty or None
        filter_col_mode = 0
        filter_m_indptr = np.array([],dtype=np.int32)
        filter_m_indices = np.array([],dtype=np.int32)

    # target col matrix
    # mode: 0 target all, 1 target array, 2 target matrix
    cdef int target_col_mode = 0
    cdef int[:] target_m_indptr
    cdef int[:] target_m_indices

    if sp.issparse(target_cols):
        assert target_cols.shape == (item_count, user_count), 'shape target_cols matrix not correct'
        target_col_mode = 2
        # build indices and indptrs and sort indices since we will use binary search
        target_cols = target_cols.tocsr()
        target_cols.eliminate_zeros()
        target_cols.sort_indices()
        target_m_indptr = np.array(target_cols.indptr, dtype=np.int32)
        target_m_indices = np.array(target_cols.indices, dtype=np.int32)
    elif isinstance(target_cols, (list, np.ndarray)):
        target_col_mode = 1
        target_m_indptr = np.array([0,len(target_cols)], dtype=np.int32)
        target_m_indices = np.array(np.sort(target_cols), dtype=np.int32)
    else:
        # target cols is None
        target_col_mode = 0
        target_m_indptr = np.array([],dtype=np.int32)
        target_m_indices = np.array([],dtype=np.int32)

    # set progress bar
    cdef int counter = 0
    cdef int * counter_add = address(counter)
    cdef int verb
    if n_targets<=5000 or verbose==False: verb = 0
    else: verb = 1

    
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
                                                            shrink, threshold,
                                                            filter_col_mode, 
                                                            &filter_m_indptr[0], &filter_m_indices[0],
                                                            target_col_mode, 
                                                            &target_m_indptr[0], &target_m_indices[0],
                                                            )
        topk = new TopK[int, float](k)
        try:
            for i in prange(n_targets, schedule='dynamic'):
                # progress bar (note: update once per 500 rows or with big matrix taking gil at each cycle destroy the performance)
                if verb==1:
                    # here, without gil, we can get some war, waw, raw error but is not so much important (it is better doesn't lost performance)
                    counter_add[0]=counter_add[0]+1
                    if counter_add[0]%(n_targets/500)==0:
                        with gil:
                            progress.desc = 'Computing'
                            progress.n = counter_add[0]
                            progress.refresh()
                # compute row
                t = targets[i]
                neighbours.setIndexRow(i)
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

    # build result in coo or csr format
    cdef int M,N
    cdef float [:] data
    cdef int [:] indices32, indptr32
    cdef long [:] indices64, indptr64

    if format_output=='coo':
        # return the result matrix in coo format
        progress.desc = 'Build coo matrix'
        progress.refresh()
        res = sp.coo_matrix((values, (rows, cols)),shape=(item_count, user_count), dtype=np.float32)
        del values, rows, cols
    else:
        # return the result matrix in csr format taking care of conversion in 32/64bit of indices if needed
        # note: normally require less memory than coo at the end of the conversion, but require to allocate more memory during the conversion
        progress.desc = 'Build csr matrix'
        progress.refresh()
        M = item_count
        N = user_count
        idx_dtype = get_index_dtype(maxval=max(n_targets*k,long(N))) #32/64 bit dtype based on total entry and max value
        if idx_dtype==np.int32:
            indptr32 = np.empty(M + 1, dtype=np.int32)
            indices32 = np.empty(n_targets * k, dtype=np.int32)
            data = np.empty(n_targets * k, dtype=np.float32)
            coo32_to_csr32(M, N, n_targets*k, &rows[0], &cols[0], &values[0], &indptr32[0], &indices32[0], &data[0])
            del values, rows, cols
            res = sp.csr_matrix((data, indices32, indptr32) ,shape=(item_count, user_count), dtype=np.float32)
            del indptr32,indices32
        else: # idx_dtype==np.int64:
            indptr64 = np.empty(M + 1, dtype=np.int64)
            indices64 = np.empty(n_targets * k, dtype=np.int64)
            data = np.empty(n_targets * k, dtype=np.float32)
            coo32_to_csr64(M, N, n_targets*k, &rows[0], &cols[0], &values[0], &indptr64[0], &indices64[0], &data[0])
            del values, rows, cols
            res = sp.csr_matrix((data, indices64, indptr64) ,shape=(item_count, user_count), dtype=np.float32)
            del indptr64,indices64
        del data
        progress.desc = 'Remove zeros'
        progress.refresh()
        res.eliminate_zeros() # routine for csr matrix
    
    # finally update progress bar and return the result matrix
    progress.desc = 'Done'
    progress.refresh()    
    progress.close()
    return res
