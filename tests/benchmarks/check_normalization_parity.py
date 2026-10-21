"""
Normalization parity: this install vs a reference install's `--dump`
(two steps, as check_kernel_parity.py).

Every output value must match with rtol=1e-5, atol=1e-7 (NaN/inf must match too);
the data dtype and the sparsity structure must be identical (index dtypes may
differ: v1 outputs int32 indices for any input, 0.7.0 kept the input's).

Usage (from the repo root; the same options on both steps):
    uv run --isolated --no-project --python 3.14 --with similaripy==0.7.0 --with pandas --with datasets \
        python tests/benchmarks/check_normalization_parity.py --dump ref_norm.npz
    python tests/benchmarks/check_normalization_parity.py --compare ref_norm.npz

    Synthetic: every mode x dtype x axis. --dataset movielens --version 32m:
    reduced set, outputs checked on a fixed 1% row sample (dump ~1 GB).
"""

import argparse
import itertools
import sys

import numpy as np
import scipy.sparse as sp
from check_kernel_parity import load_outputs, save_outputs
from dataset_loaders import load_URM
from run_small_benchmarks import make_urm

import similaripy as sim

RTOL, ATOL = 1e-5, 1e-7
TF_MODES = ("binary", "raw", "sqrt", "freq", "log")
IDF_MODES = ("unary", "base", "smooth", "prob", "bm25")


def cases(full):
    """(name, func, kwargs). full=False: defaults + one mode varied at a time."""
    out = [(f"normalize_{n}", sim.normalize, {"norm": n}) for n in ("l1", "l2", "max")]
    if full:
        modes = list(itertools.product(TF_MODES, IDF_MODES))
    else:
        modes = [(tf, "bm25") for tf in TF_MODES] + [("raw", idf) for idf in IDF_MODES if idf != "bm25"]
    for tf, idf in modes:
        kw = {"tf_mode": tf, "idf_mode": idf}
        out += [
            (f"tfidf_{tf}_{idf}", sim.tfidf, kw),
            (f"bm25_{tf}_{idf}", sim.bm25, kw),
            (f"bm25plus_{tf}_{idf}", sim.bm25plus, kw | {"delta": 0.5, "k1": 1.5, "b": 0.6}),
        ]
    return out


def with_dtypes(X, data_dtype, index_dtype):
    X = sp.csr_array(X, dtype=data_dtype)
    X.indices = X.indices.astype(index_dtype)
    X.indptr = X.indptr.astype(index_dtype)
    return X


def check_case(name, ref, rs):
    """ref: the dumped (data, indices, indptr, shape). Returns (ok, bit-identical)."""
    data, indices, indptr, shape = ref
    fail = None
    if data.dtype != rs.data.dtype:
        fail = f"dtype {data.dtype} vs {rs.data.dtype}"
    elif not (tuple(shape) == rs.shape and np.array_equal(indptr, rs.indptr) and np.array_equal(indices, rs.indices)):
        fail = "structure differs"
    else:
        bad = ~np.isclose(rs.data, data, rtol=RTOL, atol=ATOL, equal_nan=True)
        if bad.any():
            j = np.flatnonzero(bad)[0]
            fail = f"{bad.sum()} values differ, first at {j}: ref {data[j]!r} vs new {rs.data[j]!r}"
    if fail:
        print(f"{name:<40} FAIL  {fail}")
    return fail is None, fail is None and np.array_equal(data, rs.data, equal_nan=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", metavar="NPZ", help="save this install's outputs (run under the reference)")
    mode.add_argument("--compare", metavar="NPZ", help="check this install's outputs against a --dump file")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "movielens", "yambda"])
    p.add_argument("--version", default=None)
    args = p.parse_args()

    if args.dataset == "synthetic":
        X0 = make_urm()
    else:
        X0, _ = load_URM(args.dataset, version=args.version)
    full = args.dataset == "synthetic"
    # Datasets: keep a fixed 1% of output rows, or 240 full outputs take ~90 GB
    rows = None if full else np.sort(np.random.default_rng(0).choice(X0.shape[0], X0.shape[0] // 100, replace=False))
    meta = {k: v for k, v in vars(args).items() if k not in ("dump", "compare")}
    ref = None if args.dump else load_outputs(args.compare, meta)

    ok, n_exact, outputs = True, 0, {}
    for data_dtype, index_dtype, axis in itertools.product((np.float32, np.float64), (np.int32, np.int64), (1, 0)):
        X = with_dtypes(X0, data_dtype, index_dtype)
        tag = f"{np.dtype(data_dtype).name}/{np.dtype(index_dtype).name}/axis={axis}"
        group_ok = True
        for name, func, kw in cases(full):
            key = f"{name} [{tag}]"
            out = func(X, **kw | {"axis": axis})
            if rows is not None:
                out = out.tocsr()[rows]
            if ref is None:
                outputs[key] = out
            else:
                case_ok, exact = check_case(key, ref(key), out)
                group_ok &= case_ok
                n_exact += exact
        if ref is not None:
            print(f"{tag:<28} {'OK' if group_ok else 'FAIL'}")
        ok &= group_ok
    if ref is None:
        save_outputs(args.dump, outputs, meta)
        return
    print(f"{'PASS' if ok else 'FAIL'} ({len(cases(full)) * 8} cases, {n_exact} bit-identical)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
