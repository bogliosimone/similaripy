"""
    author: Simone Boglio
    mail: bogliosimone@gmail.com
"""

#!python
#distutils: language = c++
#cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport fabs, sqrt, log
from libc.math cimport M_E  # base of natural logarithm

from cython cimport floating, integral, char
from cython.view cimport array


cdef floating tf(floating freq, floating doc_len=1, str mode='raw', floating logbase=M_E):
    if mode == 'binary':
        return 1 if freq!=0 else 0
    elif mode == 'raw':
        return freq
    elif mode == 'sqrt':
        return sqrt(freq)
    elif mode == 'freq':
        return freq / doc_len
    elif mode == 'log':
        return log(1 + freq) / log(logbase)


cdef floating idf(floating inv_freq, floating n_docs=1, str mode='smooth', floating logbase=M_E):
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


def inplace_normalize_csr_l2(shape, floating[:] data, integral[:] indices, integral[:] indptr):
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
            
            
def inplace_normalize_csr_l1(shape, floating[:] data, integral[:] indices, integral[:] indptr):
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

            
def inplace_normalize_csr_max(shape, floating[:] data, integral[:] indices, integral[:] indptr):
    cdef integral n_rows = shape[0]
    cdef floating max_
    cdef integral i, j
    
    for i in range(n_rows):
        max_ = data[indptr[i]]
        for j in range(indptr[i]+1, indptr[i + 1]):
            if data[j] > max_:
                max_ = data[j]
        # handle zero division and negative values
        if max_ <= 0.0: 
            continue
        for j in range(indptr[i], indptr[i + 1]):
            data[j] /= max_


def inplace_normalize_csr_tfidf(shape, floating[:] data, integral[:] indices, integral[:] indptr,
                                str tf_mode='sqrt', str idf_mode='smooth', floating logbase=M_E):
    cdef integral n_docs = shape[0] 
    cdef integral n_words = shape[1]
    cdef floating aux
    cdef integral i,j 
    cdef char* format_ = 'f' if floating is float else 'd' # fused type
    cdef floating [:] idf_ = array(shape=(n_words,), itemsize=sizeof(floating), format=format_)
    cdef floating [:] doc_len = array(shape=(n_docs,), itemsize=sizeof(floating), format=format_)
    
    for i in range(n_words): idf_[i] = 0
    for i in range(n_docs): doc_len[i] = 0

    # compute idf incrementally and documents length
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            doc_len[i] += data[j]
            if data[j]>0:
                idf_[indices[j]] += 1

    for i in range(n_words):
        if idf_[i]!=0:
            idf_[i] = idf(inv_freq=idf_[i], n_docs=n_docs, mode=idf_mode, logbase=logbase)
    
    # compute tf idf
    cdef floating tf_
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            tf_ = tf(freq=data[j], doc_len=doc_len[i], mode=tf_mode, logbase=logbase)
            data[j] = tf_ * idf_[indices[j]]


def inplace_normalize_csr_bm25plus(shape, floating[:] data, integral[:] indices, integral[:] indptr,
                                   floating k1=1.2, floating b=0.75, floating delta=1.0,
                                   str tf_mode='raw', str idf_mode='bm25', floating logbase=M_E):
    
    cdef integral n_docs = shape[0] 
    cdef integral n_words = shape[1]
    cdef floating avg_doc_len = 0.0
    cdef floating aux
    cdef integral i,j 
    cdef char* format_ = 'f' if floating is float else 'd' # fused type
    cdef floating [:] idf_ = array(shape=(n_words,), itemsize=sizeof(floating), format=format_)
    cdef floating [:] doc_len = array(shape=(n_docs,), itemsize=sizeof(floating), format=format_)
    cdef floating [:] norm_doc_len = array(shape=(n_docs,), itemsize=sizeof(floating), format=format_)
    
    for i in range(n_words): idf_[i] = 0
    for i in range(n_docs): doc_len[i] = 0

    # compute idf and average documents length incrementally
    for i in range(n_docs):
        for j in range(indptr[i], indptr[i + 1]):
            doc_len[i] += data[j]
            if data[j]>0:
                idf_[indices[j]] += 1
        avg_doc_len += doc_len[i]

    for i in range(n_words):
        if idf_[i]!=0:
            idf_[i] = idf(inv_freq=idf_[i], n_docs=n_docs, mode=idf_mode, logbase=logbase)
     
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

