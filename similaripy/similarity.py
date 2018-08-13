from .cython_code import s_plus as s
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import numpy as np


FORMAT_OUTPUT='csr'
VERBOSE = True
K = 100
SHRINK = 0
THRESHOLD = 0
BINARY = False
TARGET_ROWS = None # compute all the rows
M2 = None


def dot_product(
    matrix1, matrix2=M2,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s.s_plus(
        matrix1, matrix2=matrix2,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output) 


def cosine(
    matrix1, matrix2=M2,
    alpha=0.5,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s.s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=alpha, c2=1-alpha,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output)


def tversky(
    matrix1, matrix2=M2,
    alpha=1,beta=1,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=alpha, t2=beta,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output) 


def jaccard(
    matrix1, matrix2=M2,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=1, t2=1,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output) 


def dice(
    matrix1, matrix2=M2,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=0.5, t2=0.5,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output)


def p3alpha(
    matrix1, matrix2=M2,
    alpha=1,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    if matrix2==M2:
        matrix2=matrix1
    matrix1 = normalize(matrix1, norm='l1', axis=1)
    matrix2 = normalize(matrix2, norm='l1', axis=1)
    m = s.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output)
    m.data = np.power(m.data, alpha)
    return m


def rp3beta(
    matrix1, matrix2=M2,
    alpha=1,
    beta=1,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    if matrix2==M2:
        matrix2=matrix1.T
    pop_m2 = matrix2.sum(axis = 0).A1
    matrix1 = normalize(matrix1, norm='l1', axis=1)
    matrix2 = normalize(matrix2, norm='l1', axis=1)
    m = s.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        weight_depop_matrix2=pop_m2,
        p2=beta,
        l3=1,
        a1=alpha,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output)
    return m


def s_plus(
    matrix1, matrix2=M2,
    l=0.5,
    t1=1, t2=1,
    c=0.5,
    k=K, shrink=SHRINK, threshold=THRESHOLD,
    binary=BINARY,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s.s_plus(
        matrix1, matrix2=matrix2,
        l1=l, l2=1-l,
        t1=t1, t2=t2,
        c1=c, c2=1-c,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output)
