"""
Parity check vs a previous version: this install's top-k outputs vs a reference install's.

Two steps, any two installs (e.g. 0.7.0 from PyPI vs this checkout, or two
commits): `--dump` saves the outputs under the reference, `--compare` recomputes
them here and checks, row by row, as (col, score) sets:
  - scores of shared columns match with rtol=1e-5 (`--rtol`), atol=1e-7;
  - a column may be in only one side only if its score is within tolerance of
    that row's k-th score or of the threshold (near-tie). These one-sided
    entries are counted and reported; anything else fails.
Each case also reports `exact` (data, indices and indptr identical byte for
byte, dtypes included) and `overlap` (shared entries / reference entries).
`--exact` makes any bit difference a failure (for changes that claim exactness).
A change of summation order (e.g. scores computed from pre-scaled inputs) moves
float32 sums over long rows by ~sqrt(n) eps: judge it with `--rtol 1e-4` plus a
float64 reference.
Run this script's version under both installs: the case lists must match.

Usage (from the repo root; the same options on both steps):
    # 0.7.0 reference (the last Cython release), outside the project env
    uv run --isolated --no-project --python 3.14 --with similaripy==0.7.0 --with pandas --with datasets \
        python tests/benchmarks/check_kernel_parity.py --dump ref.npz
    python tests/benchmarks/check_kernel_parity.py --compare ref.npz [--exact]

    Options: --block-size 1024 (synthetic, blocked path, 10 blocks);
    --dataset movielens --version 32m (dump ~0.3 GB); --similarities takes the
    benchmark harness names (e.g. rp3beta_stabilized, fixed parameters).
    Timing references are commits, not dumps: tag `v1-cython-ref` (both
    kernels in one process, `--kernel cpp rust`).
"""

import argparse
import json
import sys

import numpy as np
import scipy.sparse as sp
from benchmark import _get_similarity_function
from dataset_loaders import load_URM
from run_small_benchmarks import build_cases, make_urm

import similaripy as sim

RTOL, ATOL = 1e-5, 1e-7


def close(a, b):
    return np.abs(a - b) <= ATOL + RTOL * np.abs(b)


def compare(a, b, k, threshold=0.0):
    """Compare two top-k CSR outputs. Returns (n_one_sided, n_shared, list of failure messages)."""
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices()
    b.sort_indices()
    one_sided, shared, failures = 0, 0, []
    for r in range(a.shape[0]):
        ac, av = a.indices[a.indptr[r] : a.indptr[r + 1]], a.data[a.indptr[r] : a.indptr[r + 1]]
        bc, bv = b.indices[b.indptr[r] : b.indptr[r + 1]], b.data[b.indptr[r] : b.indptr[r + 1]]
        _, ia, ib = np.intersect1d(ac, bc, assume_unique=True, return_indices=True)
        shared += ia.size
        bad = ~close(bv[ib], av[ia])
        if bad.any():
            j = np.flatnonzero(bad)[0]
            failures.append(
                f"row {r} col {ac[ia[j]]}: ref {av[ia[j]]!r} vs new {bv[ib[j]]!r} "
                f"(rel {abs(bv[ib[j]] / av[ia[j]] - 1):.1e})"
            )
        # k-th score of each side (only meaningful when the row is full)
        kth = [v.min() for v in (av, bv) if v.size == k]
        for cols, vals, idx in ((ac, av, ia), (bc, bv, ib)):
            only = np.setdiff1d(np.arange(cols.size), idx, assume_unique=True)
            for s in vals[only]:
                if any(close(s, t) for t in kth) or close(s, threshold):
                    one_sided += 1
                else:
                    failures.append(f"row {r}: col only on one side with score {s!r} (k-th {kth})")
    # every row is counted (overlap, one-sided); the report shows the first failures
    return one_sided, shared, failures


def identical(ref_arrays, m):
    """The dumped arrays and m's CSR arrays, byte for byte (dtypes too; bytes also catch -0.0 vs 0.0).

    Kernel output rows are sorted by column, so the raw arrays are canonical."""
    return all(
        x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        for x, y in zip(ref_arrays, (m.data, m.indices, m.indptr))
    )


def save_outputs(path, outputs, meta):
    """{name: sparse matrix} + the run options -> one .npz (CSR arrays keep their dtypes)."""
    arrays = {"meta": np.array(json.dumps(meta))}
    for name, m in outputs.items():
        m = m.tocsr()
        arrays |= {
            f"{name}.data": m.data,
            f"{name}.indices": m.indices,
            f"{name}.indptr": m.indptr,
            f"{name}.shape": np.array(m.shape),
        }
    np.savez(path, **arrays)
    # Run scripts by path: `python -c` from the repo root would import the checkout, not the reference
    print(f"saved {len(outputs)} cases to {path} (similaripy {sim.__version__} from {sim.__file__})")


def load_outputs(path, meta):
    """The .npz of `save_outputs`, checked against this run's options; name -> (data, indices, indptr, shape)."""
    ref = np.load(path)
    saved = json.loads(str(ref["meta"]))
    if saved != meta:
        sys.exit(f"{path} was dumped with {saved}, this run is {meta}")
    return lambda name: tuple(ref[f"{name}.{a}"] for a in ("data", "indices", "indptr", "shape"))


def case_kwargs(kwargs, block_size):
    # auto = the library default, left out: 0.7.0 spells it 0, v1 'auto'
    auto = {} if block_size == "auto" else {"block_size": None if block_size == "none" else int(block_size)}
    return dict(k=100, verbose=False, format_output="csr") | auto | kwargs


def check_case(name, ref_arrays, new, kw, exact=False):
    new = new.tocsr()
    # Raw dumped arrays: rebuilding a scipy CSR may narrow the index dtypes
    same = identical(ref_arrays, new)  # before compare(): it sorts indices in place
    *arrays, shape = ref_arrays
    ref = sp.csr_array(tuple(arrays), shape=tuple(shape))
    one_sided, shared, failures = compare(ref, new, min(kw["k"], new.shape[1]), kw.get("threshold", 0.0))
    if exact and not same:
        failures.append("not bit-identical (--exact)")
    overlap = shared / ref.nnz if ref.nnz else 1.0
    status = "OK" if not failures else "FAIL"
    print(
        f"{name:<28} {status}  exact={'yes' if same else 'no ':<3}  overlap={overlap:.6f}  "
        f"nnz={ref.nnz:<10} one_sided={one_sided}"
    )
    for f in failures[:10]:
        print(f"    {f}")
    if len(failures) > 10:
        print(f"    ... {len(failures) - 10} more")
    return not failures


def scoring_cases():
    """Extra cases covering every scoring term and shrink type."""
    return [
        ("jaccard", sim.jaccard, {}),
        ("dice_shrink", sim.dice, {"shrink": 5}),
        ("tversky", sim.tversky, {"alpha": 0.3, "beta": 0.7}),
        ("asymmetric_cosine", sim.asymmetric_cosine, {"alpha": 0.2}),
        ("asym_bayesian", sim.asymmetric_cosine, {"alpha": 0.2, "shrink": 10, "shrink_type": "bayesian"}),
        ("dot_bayesian", sim.dot_product, {"shrink": 10, "shrink_type": "bayesian"}),
        ("cosine_bayesian", sim.cosine, {"shrink": 10, "shrink_type": "bayesian"}),
        ("cosine_stabilized", sim.cosine, {"shrink": 10, "shrink_type": "stabilized"}),
        ("cosine_additive", sim.cosine, {"shrink": 10, "shrink_type": "additive"}),
        ("cosine_threshold", sim.cosine, {"threshold": 0.2}),
        ("p3alpha", sim.p3alpha, {"alpha": 0.8}),
        ("p3alpha_bayesian", sim.p3alpha, {"alpha": 0.8, "shrink": 10, "shrink_type": "bayesian"}),
        ("rp3beta_alpha_beta", sim.rp3beta, {"alpha": 0.8, "beta": 0.6}),
        ("rp3beta_stabilized", sim.rp3beta, {"alpha": 0.8, "beta": 0.6, "shrink": 10, "shrink_type": "stabilized"}),
        ("rp3beta_bayesian", sim.rp3beta, {"alpha": 0.8, "beta": 0.6, "shrink": 10, "shrink_type": "bayesian"}),
        ("splus", sim.s_plus, {"l3": 0.5, "alpha": 0.9, "beta1": 0.3, "beta2": 0.4, "pop1": "sum", "pop2": "sum"}),
    ]


def main():
    global RTOL
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", metavar="NPZ", help="save this install's outputs (run under the reference)")
    mode.add_argument("--compare", metavar="NPZ", help="check this install's outputs against a --dump file")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "movielens", "yambda"])
    p.add_argument("--version", default=None)
    p.add_argument("--similarities", nargs="+", default=["dot_product", "cosine", "rp3beta"])
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--block-size", default="auto", help="'auto' (262144), 'none' or an int > 0")
    p.add_argument("--exact", action="store_true", help="--compare: fail on any bit difference")
    p.add_argument("--rtol", type=float, default=RTOL, help=f"--compare: score tolerance (default {RTOL})")
    args = p.parse_args()
    RTOL = args.rtol

    if args.dataset == "synthetic":
        X = make_urm().T.tocsr()
        cases = build_cases(X) + scoring_cases()
    else:
        URM, _ = load_URM(args.dataset, version=args.version)
        X = URM.T
        cases = [(s, _get_similarity_function(s), {"k": args.k}) for s in args.similarities]

    meta = {k: v for k, v in vars(args).items() if k not in ("dump", "compare", "exact", "rtol")}
    # harness variants: their fixed parameters come from the running checkout's benchmark.py
    meta["fixed"] = {name: func.fixed for name, func, _ in cases if hasattr(func, "fixed")}
    if args.dump:
        save_outputs(args.dump, {name: func(X, **case_kwargs(kw, args.block_size)) for name, func, kw in cases}, meta)
        return
    ref = load_outputs(args.compare, meta)
    ok = True
    for name, func, kw in cases:
        kw = case_kwargs(kw, args.block_size)
        ok &= check_case(name, ref(name), func(X, **kw), kw, args.exact)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
