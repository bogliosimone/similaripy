# cython: language_level=3
# cython: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport fabs, sqrt, log
from libc.math cimport M_E  # base of natural logarithm

from cython cimport floating, integral, char
from cython.view cimport array
from typing import Tuple


cdef floating tf(floating freq, floating doc_len=1, str mode='raw', floating logbase=M_E):
    """
    Compute term frequency (TF) using various modes.

    Args:
        freq: Raw frequency of the term.
        doc_len: Document length (default 1).
        mode: TF computation mode ('binary', 'raw', 'sqrt', 'freq', 'log').
        logbase: Logarithm base for 'log' mode (default M_E).

    Returns:
        Computed term frequency value.
    """
    if mode == 'binary':
        return 1 if freq != 0 else 0
    elif mode == 'raw':
        return freq
    elif mode == 'sqrt':
        return sqrt(freq)
    elif mode == 'freq':
        return freq / doc_len
    elif mode == 'log':
        return log(1 + freq) / log(logbase)


cdef floating idf(floating inv_freq, floating n_docs=1, str mode='smooth', floating logbase=M_E):
    """
    Compute inverse document frequency (IDF) using various modes.

    Args:
        inv_freq: Document frequency (number of documents containing the term).
        n_docs: Total number of documents (default 1).
        mode: IDF computation mode ('unary', 'base', 'smooth', 'prob', 'bm25').
        logbase: Logarithm base (default M_E).

    Returns:
        Computed inverse document frequency value.
    """
    if mode == 'unary':
        return 1
    elif mode == 'base':
        return log(n_docs / inv_freq) / log(logbase)
    elif mode == 'smooth':
        return log(n_docs / (1 + inv_freq)) / log(logbase)
    elif mode == 'prob':
        return log((n_docs - inv_freq) / inv_freq) / log(logbase)
    elif mode == 'bm25':
        return log((n_docs - inv_freq + 0.5) / (inv_freq + 0.5)) / log(logbase)


def inplace_normalize_csr_l2(
    shape: Tuple[int, int],
    floating[:] data,
    integral[:] indices,
    integral[:] indptr
) -> None:
    """
    In-place L2 normalization of CSR sparse matrix rows.

    Each row is normalized by dividing by its L2 norm (Euclidean length).
    Empty rows are skipped.

    Args:
        shape: Matrix shape (n_rows, n_cols).
        data: CSR data array (will be modified in-place).
        indices: CSR indices array.
        indptr: CSR indptr array.
    """
    cdef integral n_rows = shape[0]
    cdef floating sum_
    cdef integral i, j

    for i in range(n_rows):
        sum_ = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            sum_ += (data[j] * data[j])
        # handle empty row
        if sum_ == 0.0:
            continue
        sum_ = sqrt(sum_)
        for j in range(indptr[i], indptr[i + 1]):
            data[j] /= sum_


def inplace_normalize_csr_l1(
    shape: Tuple[int, int],
    floating[:] data,
    integral[:] indices,
    integral[:] indptr
) -> None:
    """
    In-place L1 normalization of CSR sparse matrix rows.

    Each row is normalized by dividing by its L1 norm (Manhattan distance).
    Empty rows are skipped.

    Args:
        shape: Matrix shape (n_rows, n_cols).
        data: CSR data array (will be modified in-place).
        indices: CSR indices array.
        indptr: CSR indptr array.
    """
    cdef integral n_rows = shape[0]
    cdef floating sum_
    cdef integral i, j

    for i in range(n_rows):
        sum_ = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            sum_ += fabs(data[j])
        # handle empty row
        if sum_ == 0.0:
            continue
        for j in range(indptr[i], indptr[i + 1]):
            data[j] /= sum_


def inplace_normalize_csr_max(
    shape: Tuple[int, int],
    floating[:] data,
    integral[:] indices,
    integral[:] indptr
) -> None:
    """
    In-place max normalization of CSR sparse matrix rows.

    Each row is normalized by dividing by its maximum value.
    Rows with max <= 0 are skipped.

    Args:
        shape: Matrix shape (n_rows, n_cols).
        data: CSR data array (will be modified in-place).
        indices: CSR indices array.
        indptr: CSR indptr array.
    """
    cdef integral n_rows = shape[0]
    cdef floating max_
    cdef integral i, j

    for i in range(n_rows):
        if indptr[i] == indptr[i + 1]:
            continue
        max_ = data[indptr[i]]
        for j in range(indptr[i] + 1, indptr[i + 1]):
            if data[j] > max_:
                max_ = data[j]
        # handle zero division and negative values
        if max_ <= 0.0:
            continue
        for j in range(indptr[i], indptr[i + 1]):
            data[j] /= max_


def inplace_normalize_csr_tfidf(
    shape: Tuple[int, int],
    floating[:] data,
    integral[:] indices,
    integral[:] indptr,
    str tf_mode='sqrt',
    str idf_mode='smooth',
    floating logbase=M_E
) -> None:
    """
    In-place TF-IDF normalization of CSR sparse matrix.

    Computes Term Frequency - Inverse Document Frequency weighting
    where rows are documents and columns are terms/words.

    Args:
        shape: Matrix shape (n_docs, n_words).
        data: CSR data array (will be modified in-place).
        indices: CSR indices array.
        indptr: CSR indptr array.
        tf_mode: Term frequency mode ('binary', 'raw', 'sqrt', 'freq', 'log').
        idf_mode: IDF mode ('unary', 'base', 'smooth', 'prob', 'bm25').
        logbase: Logarithm base for TF/IDF computation.
    """
    cdef integral n_docs = shape[0]
    cdef integral n_words = shape[1]
    cdef floating aux
    cdef integral i, j
    cdef char* format_ = 'f' if floating is float else 'd'  # fused type
    cdef floating[:] idf_ = array(shape=(n_words,), itemsize=sizeof(floating), format=format_)
    cdef floating[:] doc_len = array(shape=(n_docs,), itemsize=sizeof(floating), format=format_)

    for i in range(n_words):
        idf_[i] = 0
    for i in range(n_docs):
        doc_len[i] = 0

    # compute idf incrementally and documents length
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            doc_len[i] += data[j]
            if data[j] > 0:
                idf_[indices[j]] += 1

    for i in range(n_words):
        if idf_[i] != 0:
            idf_[i] = idf(inv_freq=idf_[i], n_docs=n_docs, mode=idf_mode, logbase=logbase)

    # compute tf idf
    cdef floating tf_
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            tf_ = tf(freq=data[j], doc_len=doc_len[i], mode=tf_mode, logbase=logbase)
            data[j] = tf_ * idf_[indices[j]]


def inplace_normalize_csr_bm25plus(
    shape: Tuple[int, int],
    floating[:] data,
    integral[:] indices,
    integral[:] indptr,
    floating k1=1.2,
    floating b=0.75,
    floating delta=1.0,
    str tf_mode='raw',
    str idf_mode='bm25',
    floating logbase=M_E
) -> None:
    """
    In-place BM25+ normalization of CSR sparse matrix.

    Computes BM25+ ranking function, an improved version of BM25
    where rows are documents and columns are terms/words.

    Args:
        shape: Matrix shape (n_docs, n_words).
        data: CSR data array (will be modified in-place).
        indices: CSR indices array.
        indptr: CSR indptr array.
        k1: Term frequency saturation parameter (typically 1.2-2.0).
        b: Document length normalization parameter (0-1, typically 0.75).
        delta: Lower-bounding parameter for BM25+ (typically 1.0).
        tf_mode: Term frequency mode ('binary', 'raw', 'sqrt', 'freq', 'log').
        idf_mode: IDF mode ('unary', 'base', 'smooth', 'prob', 'bm25').
        logbase: Logarithm base for TF/IDF computation.
    """
    cdef integral n_docs = shape[0]
    cdef integral n_words = shape[1]
    cdef floating avg_doc_len = 0.0
    cdef floating aux
    cdef integral i, j
    cdef char* format_ = 'f' if floating is float else 'd'  # fused type
    cdef floating[:] idf_ = array(shape=(n_words,), itemsize=sizeof(floating), format=format_)
    cdef floating[:] doc_len = array(shape=(n_docs,), itemsize=sizeof(floating), format=format_)
    cdef floating[:] norm_doc_len = array(shape=(n_docs,), itemsize=sizeof(floating), format=format_)

    for i in range(n_words):
        idf_[i] = 0
    for i in range(n_docs):
        doc_len[i] = 0

    # compute idf and average documents length incrementally
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            doc_len[i] += data[j]
            if data[j] > 0:
                idf_[indices[j]] += 1
        avg_doc_len += doc_len[i]

    for i in range(n_words):
        if idf_[i] != 0:
            idf_[i] = idf(inv_freq=idf_[i], n_docs=n_docs, mode=idf_mode, logbase=logbase)

    if n_docs == 0:
        return
    avg_doc_len = avg_doc_len / n_docs

    # compute documents length normalized
    for i in range(n_docs):
        norm_doc_len[i] = (1.0 - b) + b * doc_len[i] / avg_doc_len

    # weight each term with bm25
    cdef floating tf_
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            tf_ = tf(freq=data[j], doc_len=doc_len[i], mode=tf_mode, logbase=logbase)
            data[j] = idf_[indices[j]] * ((tf_ * (k1 + 1.0) / (tf_ + k1 * norm_doc_len[i])) + delta)
