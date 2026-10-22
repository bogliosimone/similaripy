"""
Serving-engine benchmark (step 4.1 exploration): per-request latency of the
prepared model (`similaripy._serve.Recommender`) and its candidate variants.

S = cosine(URM.T, k=--model-k) (or --model rp3beta), built once, not timed.
Requests = URM rows (user histories), resampled with replacement in random
order (--requests, default 20k); a history cap N keeps N random items of the
row (URM rows carry no time order). Every request excludes the history's own
items. Times are Rust-side ns per request (`ServeModel.bench`: no Python),
warm (back to back) or cold-ish (--evict-mb read between requests), plus the
Python-call latency of `Recommender.recommend` for the default variant.

Modes (--mode, several allowed):
  latency   default variant: p50/p99/mean by history cap, warm + cold, Python call
  variants  V2 seen, V3 mask (10% / 50% hidden), V4 blocked vs one block,
            V5 scratch reuse vs fresh, V8 history canonicalization; rounds in
            ABBA order, median of the per-round p50 / p99 / mean
  lookup    V1: external int64 ids, each lookup alone (--lookup-batch ids per
            call, this dataset's items + a 3M-item catalogue) and inside requests
  gate      V6: batch C (rayon over users, `ServeModel.batch`) vs the kernel
            (`dot_product(URM, S, filter_cols=URM)`, kernel + output time),
            --gate-threads (default 1 and 6), ABBA
--groups limits the variants mode (e.g. --groups V4 for Yambda 500M).

Usage:
    python run_serve_benchmarks.py --dataset movielens --version 32m --mode latency variants
    python run_serve_benchmarks.py --dataset yambda --version 50m --mode gate --rounds 3
"""

import argparse
import json
import resource
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp
from benchmark import get_system_info
from dataset_loaders import load_URM
from run_recommend_benchmarks import run_once

import similaripy as sim
from similaripy import _core
from similaripy._s_plus import DEFAULT_BLOCK_SIZE
from similaripy._serve import Recommender

BASE = {"lookup": "dense", "seen": "neginf", "mask": "after", "block": DEFAULT_BLOCK_SIZE}
VARIANTS = {
    "V2 seen": {"neginf": {}, "bitset": {"seen": "bitset"}, "later": {"seen": "later"}},
    "V4 blocking": {"blocked": {}, "one block": {"block": 0}},
    "V5 scratch": {"reuse": {}, "fresh": {"reuse_scratch": False}},
    "V8 canonicalize": {"on": {}, "off": {"canonicalize": False}},
}
MASK_VARIANTS = {"after": {}, "before": {"mask": "before"}, "later": {"mask": "later"}}
LOOKUPS = ["binary", "eytzinger", "hash", "hash_batched"]


def variant(**kw):
    return _core.ServeVariant(**{**BASE, **kw})


def histories(URM, users, cap, rng, ext=None):
    """CSR (int64 indptr, int64 ids) of the users' rows, each capped to `cap` random items."""
    rows = []
    for u in users:
        row = URM.indices[URM.indptr[u] : URM.indptr[u + 1]]
        if cap is not None and row.size > cap:
            row = rng.choice(row, size=cap, replace=False)
        rows.append(row)
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([r.size for r in rows], out=indptr[1:])
    ids = np.concatenate(rows).astype(np.int64)
    if ext is not None:
        ids = ext[ids]
    return {"indptr": indptr, "ids": ids, "weights": None}


def stats(ns):
    us = np.asarray(ns) / 1e3
    out = {"p50": float(np.percentile(us, 50)), "p99": float(np.percentile(us, 99)), "mean": float(us.mean())}
    if us.size >= 100_000:  # p99.9 only with enough samples (G7)
        out["p99.9"] = float(np.percentile(us, 99.9))
    return out


def fmt(s):
    return "  ".join(f"{k} {v:9.1f}" for k, v in s.items())


def bench(model, hist, k, v, cold=False, allowed=None, evict_mb=48):
    return model.bench(hist, k=k, exclude_seen=True, allowed=allowed, variant=v, cold=cold, evict_mb=evict_mb)


def abba(cases, rounds, run):
    """{name: [result per round]}, case order reversed every other round."""
    names = list(cases)
    out = {n: [] for n in names}
    for r in range(rounds):
        for n in names if r % 2 == 0 else names[::-1]:
            out[n].append(run(cases[n]))
    return out


def median_stats(per_round):
    return {key: statistics.median(s[key] for s in per_round) for key in per_round[0]}


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10  # bytes on macOS, KiB on Linux


def mode_latency(args, URM, rec, users, rng, report):
    model, v = rec._model, variant()
    out = {}
    for cap in args.caps:
        hist = histories(URM, users, cap, rng)
        bench(model, hist, args.k, v)  # warm-up
        warm, cold = (
            stats(bench(model, hist, args.k, v)),
            stats(bench(model, hist, args.k, v, cold=True, evict_mb=args.evict_mb)),
        )
        n_py = min(args.py_requests, len(users))
        ids = [hist["ids"][hist["indptr"][i] : hist["indptr"][i + 1]] for i in range(n_py)]
        py_ns = []
        for h in ids:
            t = time.perf_counter_ns()
            rec.recommend(h, k=args.k)
            py_ns.append(time.perf_counter_ns() - t)
        py = stats(py_ns)
        name = "none" if cap is None else str(cap)
        print(f"cap {name:>5}  warm  {fmt(warm)}\n            cold  {fmt(cold)}\n            python{fmt(py)}  (us)")
        out[name] = {"warm": warm, "cold": cold, "python": py}
    out["rss_mb_after"] = rss_mb()
    report["latency"] = out


def mode_variants(args, URM, rec, users, rng, report):
    model = rec._model
    hist = histories(URM, users, None, rng)
    bench(model, hist, args.k, variant())  # warm-up
    out = {}
    for group, cases in VARIANTS.items():
        if args.groups and group.split()[0] not in args.groups:
            continue
        res = abba(cases, args.rounds, lambda kw: stats(bench(model, hist, args.k, variant(**kw))))
        out[group] = {n: median_stats(r) for n, r in res.items()}
    n = rec.n_items
    for hidden in (0.1, 0.5) if not args.groups or "V3" in args.groups else ():
        allowed_ids = rng.choice(n, size=int(n * (1 - hidden)), replace=False)
        mask = rec.mask(allowed_ids)

        def run(kw, mask=mask):
            return stats(bench(model, hist, args.k, variant(**kw), allowed=mask))

        res = abba(MASK_VARIANTS, args.rounds, run)
        group = f"V3 mask {int(hidden * 100)}% hidden"
        out[group] = {name: median_stats(r) for name, r in res.items()}
        # completeness of the over-fetch stage: requests with < k results
        v_later = variant(mask="later")
        v_after = variant()
        short = 0
        for i in range(min(2000, len(users))):
            h = hist["ids"][hist["indptr"][i] : hist["indptr"][i + 1]]
            a = model.recommend(h, None, k=args.k, exclude_seen=True, allowed=mask, variant=v_after)[0].size
            b = model.recommend(h, None, k=args.k, exclude_seen=True, allowed=mask, variant=v_later)[0].size
            short += b < a
        out[group]["later: requests with fewer results than 'after' (of 2000)"] = short
    for group, res in out.items():
        print(group)
        for name, s in res.items():
            print(f"  {name:<10} {fmt(s) if isinstance(s, dict) else s}")
    report["variants"] = out


def mode_lookup(args, URM, S, users, rng, report):
    out = {}
    n = S.shape[0]
    ext = rng.choice(2**62, size=n, replace=False).astype(np.int64)
    rec = Recommender(S, item_ids=ext)
    # alone: --lookup-batch known ids per call, ns per id; + a 3M-item catalogue (empty S)
    big_keys = rng.choice(2**62, size=3_000_000, replace=False).astype(np.int64)
    big = Recommender(sp.csr_array((3_000_000, 3_000_000), dtype=np.float32), item_ids=big_keys)
    for name, r, keys in (("items", rec, ext), ("3M", big, big_keys)):
        q = rng.choice(keys, size=args.lookup_batch).astype(np.int64)
        cases = {lk: lk for lk in LOOKUPS}
        res = abba(
            cases, args.rounds, lambda lk, r=r, q=q: r._model.bench_lookup(q, variant(lookup=lk), 200) / (200 * q.size)
        )
        out[name] = {lk: statistics.median(v) for lk, v in res.items()}
        print(
            f"lookup alone, {name} ({len(keys):,} keys, {q.size} ids/call): "
            + "  ".join(f"{lk} {v:.1f}" for lk, v in out[name].items())
            + "  ns/id"
        )
    # inside requests (uncapped histories, external ids)
    hist = histories(URM, users, None, rng, ext=ext)
    bench(rec._model, hist, args.k, variant(lookup="binary"))
    res = abba(
        {lk: lk for lk in LOOKUPS}, args.rounds, lambda lk: stats(bench(rec._model, hist, args.k, variant(lookup=lk)))
    )
    out["in requests"] = {lk: median_stats(r) for lk, r in res.items()}
    dense = histories(URM, users, None, rng)
    out["in requests"]["dense (identity ids)"] = stats(bench(rec._model, dense, args.k, variant(lookup="dense")))
    for lk, s in out["in requests"].items():
        print(f"  in requests {lk:<22} {fmt(s)}")
    report["lookup"] = out


def mode_gate(args, URM, S, rec, report):
    URM = URM.tocsr()
    H = {"indptr": URM.indptr.astype(np.int64), "ids": URM.indices.astype(np.int64), "weights": None}
    out = {}
    for threads in args.gate_threads:

        def kernel(_, threads=threads):
            t = run_once(URM, S, args.k, threads, True, "csr")
            return t["kernel"] + t["output"]

        def engine(_, threads=threads):
            return rec._model.batch(H, k=args.k, exclude_seen=True, variant=variant(), num_threads=threads)[3]

        cases = {"kernel": kernel, "C": engine}
        for c in cases.values():  # warm-up
            c(None)
        res = abba(cases, args.rounds, lambda f: f(None))
        med = {n: statistics.median(v) for n, v in res.items()}
        delta = med["C"] / med["kernel"] - 1
        print(f"gate threads {threads}: kernel {med['kernel']:.3f}s  C {med['C']:.3f}s  ({delta:+.1%})  rounds {res}")
        out[str(threads)] = {"median": med, "delta": delta, "rounds": res}
    report["gate"] = out


def main():
    p = argparse.ArgumentParser(description="Serving-engine benchmark for similaripy (step 4.1)")
    p.add_argument("--dataset", default="movielens")
    p.add_argument("--version", default=None)
    p.add_argument("--model", default="cosine", choices=["cosine", "rp3beta"])
    p.add_argument("--model-k", type=int, default=100)
    p.add_argument("--k", type=int, default=100, help="Recommendations per request (default: 100)")
    p.add_argument("--mode", nargs="+", default=["latency"], choices=["latency", "variants", "lookup", "gate"])
    p.add_argument("--requests", type=int, default=20_000)
    p.add_argument("--py-requests", type=int, default=20_000)
    p.add_argument("--caps", nargs="+", default=["50", "200", "1000", "none"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--evict-mb", type=int, default=48)
    p.add_argument("--lookup-batch", type=int, default=1000)
    p.add_argument("--gate-threads", type=int, nargs="+", default=[1, 6])
    p.add_argument("--groups", nargs="+", default=None, help="variants mode: only these (V2 V3 V4 V5 V8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    args.caps = [None if c == "none" else int(c) for c in args.caps]

    URM, _ = load_URM(args.dataset, version=args.version, verbose=False)
    URM = URM.tocsr()
    t = time.perf_counter()
    if args.model == "cosine":
        S = sim.cosine(URM.T, k=args.model_k, verbose=False)
    else:
        S = sim.rp3beta(URM.T, k=args.model_k, verbose=False)
    t_s = time.perf_counter() - t
    rss_s = rss_mb()
    t = time.perf_counter()
    rec = Recommender(S)
    t_build = time.perf_counter() - t
    print(
        f"URM {URM.shape} nnz {URM.nnz:,}; S ({args.model}) nnz {S.nnz:,} in {t_s:.1f}s; "
        f"model build {t_build:.2f}s, n_active {rec._model.n_active:,}; max RSS {rss_s:.0f} -> {rss_mb():.0f} MB"
    )
    rng = np.random.default_rng(args.seed)
    users = rng.choice(URM.shape[0], size=args.requests)  # resampled, random order
    report = {"system": get_system_info(), "args": vars(args), "build_s": t_build, "s_nnz": int(S.nnz)}
    for m in args.mode:
        print(f"\n== {m} ==")
        if m == "latency":
            mode_latency(args, URM, rec, users, rng, report)
        elif m == "variants":
            mode_variants(args, URM, rec, users, rng, report)
        elif m == "lookup":
            mode_lookup(args, URM, S, users, rng, report)
        else:
            mode_gate(args, URM, S, rec, report)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
