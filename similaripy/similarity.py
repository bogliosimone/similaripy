from .cython_code import s_plus as _sim
from .normalization import normalize as _normalize
import numpy as _np


_FORMAT_OUTPUT='coo'
_VERBOSE = True
_K = 100
_SHRINK = 0
_THRESHOLD = 0
_BINARY = False
_TARGET_ROWS = None # compute all rows
_TARGET_COLS = None # compute all cols
_FILTER_COLS = None # do not filter cols
_M2 = None


def dot_product(
    matrix1, matrix2=_M2,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output) 


def cosine(
    matrix1, matrix2=_M2,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=0.5, c2=0.5,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output)


def asymmetric_cosine(
    matrix1, matrix2=_M2,
    alpha=0.5,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=alpha, c2=1-alpha,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output)


def tversky(
    matrix1, matrix2=_M2,
    alpha=1,beta=1,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=alpha, t2=beta,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output) 


def jaccard(
    matrix1, matrix2=_M2,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=1, t2=1,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output) 


def dice(
    matrix1, matrix2=_M2,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=0.5, t2=0.5,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output)


def p3alpha(
    matrix1, matrix2=_M2,
    alpha=1,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    if matrix2==_M2:
        matrix2=matrix1.T
    matrix1 = _normalize(matrix1, norm='l1', axis=1, inplace=False)
    matrix1.data = _np.power(matrix1.data, alpha)
    matrix2 = _normalize(matrix2, norm='l1', axis=1, inplace=False)
    matrix2.data = _np.power(matrix2.data, alpha)
    m = _sim.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output)
    return m


def rp3beta(
    matrix1, matrix2=_M2,
    alpha=1,
    beta=1,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    if matrix2==_M2:
        matrix2=matrix1.T
    pop_m2 = matrix2.sum(axis = 0).A1
    matrix1 = _normalize(matrix1, norm='l1', axis=1, inplace=False)
    matrix1.data = _np.power(matrix1.data, alpha)
    matrix2 = _normalize(matrix2, norm='l1', axis=1, inplace=False)
    matrix2.data = _np.power(matrix2.data, alpha)
    m = _sim.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        weight_depop_matrix2=pop_m2,
        p2=beta,
        l3=1,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output)
    return m


def s_plus(
    matrix1, matrix2=_M2,
    l=0.5,
    t1=1, t2=1,
    c=0.5,
    k=_K, shrink=_SHRINK, threshold=_THRESHOLD,
    binary=_BINARY,
    target_rows=_TARGET_ROWS,
    target_cols=_TARGET_COLS,
    filter_cols=_FILTER_COLS,
    verbose=_VERBOSE,
    format_output=_FORMAT_OUTPUT
    ):
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=l, l2=1-l,
        t1=t1, t2=t2,
        c1=c, c2=1-c,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output)
