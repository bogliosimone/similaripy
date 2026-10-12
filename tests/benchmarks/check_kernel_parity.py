"""
R2 parity check: this install's top-k outputs vs a reference install's.

Two steps, any two installs (e.g. 0.7.0 from PyPI vs this checkout, or two
commits): `--dump` saves the outputs under the reference, `--compare` recomputes
them here and checks, row by row, as (col, score) sets:
  - scores of shared columns match with rtol=1e-5, atol=1e-7;
  - a column may be in only one side only if its score is within tolerance of
    that row's k-th score or of the threshold (near-tie). These one-sided
    entries are counted and reported; anything else fails.

Usage (from the repo root; the same options on both steps):
    # 0.7.0 reference (the last Cython release), outside the project env
    uv run --isolated --no-project --python 3.14 --with similaripy==0.7.0 --with pandas --with datasets \
        python tests/benchmarks/check_kernel_parity.py --dump ref.npz
    python tests/benchmarks/check_kernel_parity.py --compare ref.npz

    Options: --block-size 1024 (synthetic, blocked path, 10 blocks);
    --dataset movielens --version 32m (dump ~0.3 GB).
    Timing references are commits, not dumps: tag `v1-cython-ref` (both
    kernels in one process, `--kernel cpp rust`).
"""

import argparse
import json
import sys

import numpy as np
import scipy.sparse as sp

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
            failures.append(f"row {r} col {ac[ia[j]]}: ref {av[ia[j]]!r} vs new {bv[ib[j]]!r}")
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


def save_outputs(path, outputs, meta):
    """{name: sparse matrix} + the run options -> one .npz (CSR arrays keep their dtypes)."""
    arrays = {'meta': np.array(json.dumps(meta))}
    for name, m in outputs.items():
        m = m.tocsr()
        arrays |= {f'{name}.data': m.data, f'{name}.indices': m.indices,
                   f'{name}.indptr': m.indptr, f'{name}.shape': np.array(m.shape)}
    np.savez(path, **arrays)
    # Run scripts by path: `python -c` from the repo root would import the checkout, not the reference
    print(f"saved {len(outputs)} cases to {path} (similaripy {sim.__version__} from {sim.__file__})")


def load_outputs(path, meta):
    """The .npz of `save_outputs`, checked against this run's options; name -> CSR getter."""
    ref = np.load(path)
    saved = json.loads(str(ref['meta']))
    if saved != meta:
        sys.exit(f"{path} was dumped with {saved}, this run is {meta}")
    return lambda name: sp.csr_array((ref[f'{name}.data'], ref[f'{name}.indices'], ref[f'{name}.indptr']),
                                     shape=tuple(ref[f'{name}.shape']))


def case_kwargs(kwargs, block_size):
    return dict(k=100, verbose=False, format_output='csr', block_size=block_size) | kwargs


def check_case(name, ref, new, kw):
    one_sided, failures = compare(ref, new, min(kw['k'], new.shape[0]), kw.get('threshold', 0.0))
    status = 'OK' if not failures else 'FAIL'
    print(f"{name:<28} {status}  nnz={ref.nnz:<10} one_sided={one_sided}")
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
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dump', metavar='NPZ', help="save this install's outputs (run under the reference)")
    mode.add_argument('--compare', metavar='NPZ', help="check this install's outputs against a --dump file")
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

    meta = {k: v for k, v in vars(args).items() if k not in ('dump', 'compare')}
    if args.dump:
        save_outputs(args.dump, {name: func(X, **case_kwargs(kw, args.block_size)) for name, func, kw in cases}, meta)
        return
    ref = load_outputs(args.compare, meta)
    ok = True
    for name, func, kw in cases:
        kw = case_kwargs(kw, args.block_size)
        ok &= check_case(name, ref(name), func(X, **kw), kw)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
