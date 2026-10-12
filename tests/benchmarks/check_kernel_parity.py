"""
R2 parity check: Rust kernel vs C++ kernel (`_backend='rust'` vs `'cython'`).

Row by row, as (col, score) sets:
  - scores of shared columns match with rtol=1e-5, atol=1e-7;
  - a column may be in only one side only if its score is within tolerance of
    that row's k-th score or of the threshold (near-tie). These one-sided
    entries are counted and reported; anything else fails.

Usage:
    python check_kernel_parity.py                       # synthetic 10k, all cases
    python check_kernel_parity.py --block-size 1024     # synthetic, blocked path (10 blocks)
    python check_kernel_parity.py --dataset movielens --version 32m
"""

import argparse
import sys

import numpy as np

import similaripy as sim
from dataset_loaders import load_URM
from run_small_benchmarks import build_cases, make_urm

RTOL, ATOL = 1e-5, 1e-7


def close(a, b):
    return np.abs(a - b) <= ATOL + RTOL * np.abs(b)


def compare(a, b, k, threshold=0.0):
    """Compare two top-k CSR outputs. Returns (n_one_sided, list of failure messages)."""
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices()
    b.sort_indices()
    one_sided, failures = 0, []
    for r in range(a.shape[0]):
        ac, av = a.indices[a.indptr[r]:a.indptr[r + 1]], a.data[a.indptr[r]:a.indptr[r + 1]]
        bc, bv = b.indices[b.indptr[r]:b.indptr[r + 1]], b.data[b.indptr[r]:b.indptr[r + 1]]
        _, ia, ib = np.intersect1d(ac, bc, assume_unique=True, return_indices=True)
        bad = ~close(bv[ib], av[ia])
        if bad.any():
            j = np.flatnonzero(bad)[0]
            failures.append(f"row {r} col {ac[ia[j]]}: cpp {av[ia[j]]!r} vs rust {bv[ib[j]]!r}")
        # k-th score of each side (only meaningful when the row is full)
        kth = [v.min() for v in (av, bv) if v.size == k]
        for cols, vals, idx in ((ac, av, ia), (bc, bv, ib)):
            only = np.setdiff1d(np.arange(cols.size), idx, assume_unique=True)
            for s in vals[only]:
                if any(close(s, t) for t in kth) or close(s, threshold):
                    one_sided += 1
                else:
                    failures.append(f"row {r}: col only on one side with score {s!r} (k-th {kth})")
        if len(failures) >= 10:
            break
    return one_sided, failures


def run_case(name, func, X, kwargs, block_size=0):
    kw = dict(k=100, verbose=False, format_output='csr', block_size=block_size) | kwargs
    rust = func(X, **kw, _backend='rust')
    cpp = func(X, **kw, _backend='cython')
    one_sided, failures = compare(cpp, rust, min(kw['k'], X.shape[0]), kw.get('threshold', 0.0))
    status = 'OK' if not failures else 'FAIL'
    print(f"{name:<28} {status}  nnz={cpp.nnz:<10} one_sided={one_sided}")
    for f in failures:
        print(f"    {f}")
    return not failures


def scoring_cases():
    """Extra cases covering every scoring term and shrink type."""
    return [
        ('jaccard', sim.jaccard, {}),
        ('dice_shrink', sim.dice, {'shrink': 5}),
        ('tversky', sim.tversky, {'alpha': 0.3, 'beta': 0.7}),
        ('asymmetric_cosine', sim.asymmetric_cosine, {'alpha': 0.2}),
        ('cosine_bayesian', sim.cosine, {'shrink': 10, 'shrink_type': 'bayesian'}),
        ('cosine_additive', sim.cosine, {'shrink': 10, 'shrink_type': 'additive'}),
        ('cosine_threshold', sim.cosine, {'threshold': 0.2}),
        ('p3alpha', sim.p3alpha, {'alpha': 0.8}),
        ('splus', sim.s_plus, {'l3': 0.5, 'alpha': 0.9, 'beta1': 0.3, 'beta2': 0.4,
                               'pop1': 'sum', 'pop2': 'sum'}),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='synthetic', choices=['synthetic', 'movielens', 'yambda'])
    p.add_argument('--version', default=None)
    p.add_argument('--similarities', nargs='+', default=['dot_product', 'cosine', 'rp3beta'])
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--block-size', type=int, default=0, help='0 = auto (262144)')
    args = p.parse_args()

    if args.dataset == 'synthetic':
        X = make_urm().T.tocsr()
        cases = build_cases(X) + scoring_cases()
    else:
        URM, _ = load_URM(args.dataset, version=args.version)
        X = URM.T
        cases = [(s, getattr(sim, s), {'k': args.k}) for s in args.similarities]

    ok = all([run_case(name, func, X, kw, args.block_size) for name, func, kw in cases])
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
