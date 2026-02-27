# cython: language_level=3
# cython: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport fabs, sqrt, log
from libc.math cimport M_E  # base of natural logarithm

from cython cimport floating, integral, char
from cython.view cimport array
from typing import Tuple

cdef enum TFMode:
    TF_BINARY = 0
    TF_RAW    = 1
    TF_SQRT   = 2
    TF_FREQ   = 3
    TF_LOG    = 4

cdef enum IDFMode:
    IDF_UNARY  = 0
    IDF_BASE   = 1
    IDF_SMOOTH = 2
    IDF_PROB   = 3
    IDF_BM25   = 4


cdef int _resolve_tf_mode(str mode) except -1:
    """Resolve TF mode string to integer constant (called once before loops)."""
    if mode == 'binary': return TF_BINARY
    if mode == 'raw':    return TF_RAW
    if mode == 'sqrt':   return TF_SQRT
    if mode == 'freq':   return TF_FREQ
    if mode == 'log':    return TF_LOG
    raise ValueError(f"Unknown tf_mode '{mode}'. Expected: binary, raw, sqrt, freq, log")


cdef int _resolve_idf_mode(str mode) except -1:
    """Resolve IDF mode string to integer constant (called once before loops)."""
    if mode == 'unary':  return IDF_UNARY
    if mode == 'base':   return IDF_BASE
    if mode == 'smooth': return IDF_SMOOTH
    if mode == 'prob':   return IDF_PROB
    if mode == 'bm25':   return IDF_BM25
    raise ValueError(f"Unknown idf_mode '{mode}'. Expected: unary, base, smooth, prob, bm25")


cdef inline floating tf(floating freq, floating doc_len, int mode, floating log_logbase) noexcept nogil:
    """
    Compute term frequency (TF) using integer-dispatched mode.

    Args:
        freq: Raw frequency of the term.
        doc_len: Document length.
        mode: TF mode constant (TF_BINARY, TF_RAW, TF_SQRT, TF_FREQ, TF_LOG).
        log_logbase: Pre-computed log(logbase) for log mode.

    Returns:
        Computed term frequency value.
    """
    if mode == TF_BINARY:
        return 1 if freq != 0 else 0
    elif mode == TF_RAW:
        return freq
    elif mode == TF_SQRT:
        return sqrt(freq)
    elif mode == TF_FREQ:
        return freq / doc_len
    else:  # TF_LOG
        return log(1 + freq) / log_logbase


cdef inline floating idf(floating inv_freq, floating n_docs, int mode, floating log_logbase) noexcept nogil:
    """
    Compute inverse document frequency (IDF) using integer-dispatched mode.

    Args:
        inv_freq: Document frequency (number of documents containing the term).
        n_docs: Total number of documents.
        mode: IDF mode constant (IDF_UNARY, IDF_BASE, IDF_SMOOTH, IDF_PROB, IDF_BM25).
        log_logbase: Pre-computed log(logbase) for log modes.

    Returns:
        Computed inverse document frequency value.
    """
    if mode == IDF_UNARY:
        return 1
    elif mode == IDF_BASE:
        return log(n_docs / inv_freq) / log_logbase
    elif mode == IDF_SMOOTH:
        return log(n_docs / (1 + inv_freq)) / log_logbase
    elif mode == IDF_PROB:
        return log((n_docs - inv_freq) / inv_freq) / log_logbase
    else:  # IDF_BM25
        return log((n_docs - inv_freq + 0.5) / (inv_freq + 0.5)) / log_logbase


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
    # Resolve string modes to int constants once (avoids Python string ops in loops)
    cdef int tf_mode_id = _resolve_tf_mode(tf_mode)
    cdef int idf_mode_id = _resolve_idf_mode(idf_mode)
    cdef floating log_logbase = log(logbase)

    cdef integral n_docs = shape[0]
    cdef integral n_words = shape[1]
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
            idf_[i] = idf(idf_[i], n_docs, idf_mode_id, log_logbase)

    # compute tf-idf
    cdef floating tf_
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            tf_ = tf(data[j], doc_len[i], tf_mode_id, log_logbase)
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
    # Resolve string modes to int constants once (avoids Python string ops in loops)
    cdef int tf_mode_id = _resolve_tf_mode(tf_mode)
    cdef int idf_mode_id = _resolve_idf_mode(idf_mode)
    cdef floating log_logbase = log(logbase)

    cdef integral n_docs = shape[0]
    cdef integral n_words = shape[1]
    cdef floating avg_doc_len = 0.0
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
            idf_[i] = idf(idf_[i], n_docs, idf_mode_id, log_logbase)

    if n_docs == 0:
        return
    avg_doc_len = avg_doc_len / n_docs

    # compute documents length normalized
    for i in range(n_docs):
        norm_doc_len[i] = (1.0 - b) + b * doc_len[i] / avg_doc_len

    # weight each term with bm25+
    cdef floating tf_
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            tf_ = tf(data[j], doc_len[i], tf_mode_id, log_logbase)
            data[j] = idf_[indices[j]] * ((tf_ * (k1 + 1.0) / (tf_ + k1 * norm_doc_len[i])) + delta)
