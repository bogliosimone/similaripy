"""
Normalization parity: Rust (`_backend='rust'`) vs Cython (`'cython'`), v1 Step 2a.

Every output value must match with rtol=1e-5, atol=1e-7 (NaN/inf must match too);
output dtypes and the sparsity structure must be identical.

Usage:
    python check_normalization_parity.py                        # synthetic: every mode x dtype x axis
    python check_normalization_parity.py --dataset movielens --version 32m   # reduced set
"""

import argparse
import itertools
import sys

import numpy as np
import scipy.sparse as sp

import similaripy as sim
from dataset_loaders import load_URM
from run_small_benchmarks import make_urm

RTOL, ATOL = 1e-5, 1e-7
TF_MODES = ('binary', 'raw', 'sqrt', 'freq', 'log')
IDF_MODES = ('unary', 'base', 'smooth', 'prob', 'bm25')


def cases(full):
    """(name, func, kwargs). full=False: defaults + one mode varied at a time."""
    out = [(f'normalize_{n}', sim.normalize, {'norm': n}) for n in ('l1', 'l2', 'max')]
    if full:
        modes = list(itertools.product(TF_MODES, IDF_MODES))
    else:
        modes = [(tf, 'bm25') for tf in TF_MODES] + [('raw', idf) for idf in IDF_MODES if idf != 'bm25']
    for tf, idf in modes:
        kw = {'tf_mode': tf, 'idf_mode': idf}
        out += [(f'tfidf_{tf}_{idf}', sim.tfidf, kw),
                (f'bm25_{tf}_{idf}', sim.bm25, kw),
                (f'bm25plus_{tf}_{idf}', sim.bm25plus, kw | {'delta': 0.5, 'k1': 1.5, 'b': 0.6})]
    return out


def with_dtypes(X, data_dtype, index_dtype):
    X = sp.csr_array(X, dtype=data_dtype)
    X.indices = X.indices.astype(index_dtype)
    X.indptr = X.indptr.astype(index_dtype)
    return X


def run_case(name, func, X, kw):
    cy = func(X, **kw, _backend='cython')
    rs = func(X, **kw, _backend='rust')
    fail = None
    if cy.data.dtype != rs.data.dtype or cy.indices.dtype != rs.indices.dtype:
        fail = f'dtype {cy.data.dtype}/{cy.indices.dtype} vs {rs.data.dtype}/{rs.indices.dtype}'
    elif not (np.array_equal(cy.indptr, rs.indptr) and np.array_equal(cy.indices, rs.indices)):
        fail = 'structure differs'
    else:
        bad = ~np.isclose(rs.data, cy.data, rtol=RTOL, atol=ATOL, equal_nan=True)
        if bad.any():
            j = np.flatnonzero(bad)[0]
            fail = f'{bad.sum()} values differ, first at {j}: cython {cy.data[j]!r} vs rust {rs.data[j]!r}'
    if fail:
        print(f'{name:<40} FAIL  {fail}')
    return fail is None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='synthetic', choices=['synthetic', 'movielens', 'yambda'])
    p.add_argument('--version', default=None)
    args = p.parse_args()

    if args.dataset == 'synthetic':
        X0 = make_urm()
    else:
        X0, _ = load_URM(args.dataset, version=args.version)
    full = args.dataset == 'synthetic'

    ok, n = True, 0
    for data_dtype, index_dtype, axis in itertools.product((np.float32, np.float64), (np.int32, np.int64), (1, 0)):
        X = with_dtypes(X0, data_dtype, index_dtype)
        tag = f'{np.dtype(data_dtype).name}/{np.dtype(index_dtype).name}/axis={axis}'
        group_ok = all([run_case(f'{name} [{tag}]', func, X, kw | {'axis': axis})
                        for name, func, kw in cases(full)])
        n += len(cases(full))
        print(f'{tag:<28} {"OK" if group_ok else "FAIL"}')
        ok &= group_ok
    print(f'{"PASS" if ok else "FAIL"} ({n} cases)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
