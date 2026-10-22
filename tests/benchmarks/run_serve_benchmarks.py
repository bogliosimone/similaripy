"""
Serving-engine benchmark: `similaripy.Recommender` latency, batch vs the
kernel, batch block size, and the model build's memory.

S = cosine(URM.T, k=--model-k) (or --model rp3beta), built once, not timed.
Requests = URM rows (user histories), resampled with replacement in random
order (--requests, default 20k); a history cap N keeps N random items of the
row (URM rows carry no time order). Every request excludes the history's own
items. The step 4.1 variant comparisons (lookup, seen, mask, scratch,
canonicalization) ran on the exploration build, commit c15ee3a.

Modes (--mode, several allowed):
  latency  `recommend` end to end (Python call), µs p50/p99/mean by history
           cap, warm (back to back) and cold-ish (--evict-mb read between calls)
  gate     batch (`recommend_batch`) vs the kernel
           (`dot_product(URM, S, filter_cols=URM)`, kernel + output time), same
           weights, --gate-threads (default 1 and 6), ABBA, + top-k overlap
  block    batch block size: `DEFAULT_BLOCK_SIZE` vs one block per request,
           --gate-threads, ABBA
  build    the model build's time and peak memory in a fresh process (S saved
           to --scratch first): max RSS after loading S vs after the build

Usage:
    python run_serve_benchmarks.py --dataset movielens --version 32m --mode latency gate block
    python run_serve_benchmarks.py --dataset yambda --version 50m --mode build --scratch /tmp/s.npz
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
from benchmark import get_system_info
from dataset_loaders import load_URM
from run_recommend_benchmarks import run_once

import similaripy as sim
from similaripy._s_plus import DEFAULT_BLOCK_SIZE


def stats(ns):
    us = np.asarray(ns) / 1e3
    out = {"p50": float(np.percentile(us, 50)), "p99": float(np.percentile(us, 99)), "mean": float(us.mean())}
    if us.size >= 100_000:  # p99.9 only with enough samples
        out["p99.9"] = float(np.percentile(us, 99.9))
    return out


def fmt(s):
    return "  ".join(f"{k} {v:9.1f}" for k, v in s.items())


def abba(cases, rounds, run):
    """{name: [result per round]}, case order reversed every other round."""
    names = list(cases)
    out = {n: [] for n in names}
    for r in range(rounds):
        for n in names if r % 2 == 0 else names[::-1]:
            out[n].append(run(cases[n]))
    return out


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10  # bytes on macOS, KiB on Linux


def histories(URM, users, cap, rng):
    rows = []
    for u in users:
        row = URM.indices[URM.indptr[u] : URM.indptr[u + 1]]
        if cap is not None and row.size > cap:
            row = rng.choice(row, size=cap, replace=False)
        rows.append(row.astype(np.int64))
    return rows


def mode_latency(args, URM, rec, users, rng, report):
    evict = np.ones(args.evict_mb << 20, dtype=np.uint8)
    out = {}
    for cap in args.caps:
        hists = histories(URM, users, cap, rng)
        for h in hists[:1000]:  # warm-up
            rec.recommend(h, k=args.k)
        res = {}
        for regime in ("warm", "cold"):
            ns = []
            for h in hists:
                if regime == "cold":
                    evict[::64].sum()  # one read per cache line
                t = time.perf_counter_ns()
                rec.recommend(h, k=args.k)
                ns.append(time.perf_counter_ns() - t)
            res[regime] = stats(ns)
        name = "none" if cap is None else str(cap)
        print(f"cap {name:>5}  warm {fmt(res['warm'])}\n            cold {fmt(res['cold'])}  (us)")
        out[name] = res
    report["latency"] = out


def batch_call(rec, URM, k, threads, block=DEFAULT_BLOCK_SIZE):
    """Times the Rust batch call only (H's prep outside, as the kernel side's
    `kernel + output` excludes its prep)."""
    H = (URM.indptr.astype(np.int64), URM.indices.astype(np.int32), URM.data.astype(np.float32))

    def run(threads=threads, block=block):
        t = time.perf_counter()
        rec._model.batch(*H, k=k, exclude_seen=True, allowed=None, block=block, num_threads=threads)
        return time.perf_counter() - t

    return run


def mode_gate(args, URM, S, rec, report):
    out = {}
    K = sim.dot_product(URM, S, k=args.k, filter_cols=URM, verbose=False, format_output="csr")
    C = rec.recommend_batch(URM, k=args.k)
    overlap = (K != 0).multiply(C != 0).nnz / max(K.nnz, 1)
    print(f"top-k overlap batch vs kernel: {overlap:.6f} (kernel nnz {K.nnz:,}, batch nnz {C.nnz:,})")
    out["overlap"] = overlap
    del K, C
    for threads in args.gate_threads:

        def kernel(threads=threads):
            t = run_once(URM, S, args.k, threads, True, "csr")
            return t["kernel"] + t["output"]

        cases = {"kernel": kernel, "batch": batch_call(rec, URM, args.k, threads)}
        for f in cases.values():  # warm-up
            f()
        res = abba(cases, args.rounds, lambda f: f())
        med = {n: statistics.median(v) for n, v in res.items()}
        delta = med["batch"] / med["kernel"] - 1
        print(f"gate threads {threads}: kernel {med['kernel']:.3f}s  batch {med['batch']:.3f}s  ({delta:+.1%})  {res}")
        out[str(threads)] = {"median": med, "delta": delta, "rounds": res}
    report["gate"] = out


def mode_block(args, URM, rec, report):
    out = {}
    for threads in args.gate_threads:
        cases = {
            "blocked": batch_call(rec, URM, args.k, threads, DEFAULT_BLOCK_SIZE),
            "one block": batch_call(rec, URM, args.k, threads, 0),
        }
        for f in cases.values():  # warm-up
            f()
        res = abba(cases, args.rounds, lambda f: f())
        med = {n: statistics.median(v) for n, v in res.items()}
        print(f"batch block threads {threads}: " + "  ".join(f"{n} {v:.3f}s" for n, v in med.items()) + f"  {res}")
        out[str(threads)] = {"median": med, "rounds": res}
    report["block"] = out


BUILD_CHILD = """
import resource, sys, time
import scipy.sparse as sp
import similaripy as sim
def rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10
S = sp.load_npz(sys.argv[1]).tocsr()
before = rss()
t = time.perf_counter()
rec = sim.Recommender(S)
print(f"{time.perf_counter() - t:.3f} {before:.0f} {rss():.0f} {S.nnz}")
"""


def mode_build(args, S, report):
    sp.save_npz(args.scratch, sp.csr_array(S), compressed=False)
    r = subprocess.run([sys.executable, "-c", BUILD_CHILD, args.scratch], capture_output=True, text=True, check=True)
    secs, before, after, nnz = r.stdout.split()
    mb_s = int(nnz) * 8 / 2**20
    peak = float(after) - float(before)
    print(
        f"build {float(secs):.2f}s; max RSS after loading S {before} MB -> after build {after} MB "
        f"(+{peak:.0f} MB; S = {mb_s:.0f} MB of indices + data: {peak / mb_s:.2f}x)"
    )
    report["build"] = {
        "seconds": float(secs),
        "rss_before_mb": float(before),
        "rss_after_mb": float(after),
        "s_mb": mb_s,
    }


def main():
    p = argparse.ArgumentParser(description="Serving-engine benchmark for similaripy")
    p.add_argument("--dataset", default="movielens")
    p.add_argument("--version", default=None)
    p.add_argument("--model", default="cosine", choices=["cosine", "rp3beta"])
    p.add_argument("--model-k", type=int, default=100)
    p.add_argument("--k", type=int, default=100, help="Recommendations per request (default: 100)")
    p.add_argument("--mode", nargs="+", default=["latency"], choices=["latency", "gate", "block", "build"])
    p.add_argument("--requests", type=int, default=20_000)
    p.add_argument("--caps", nargs="+", default=["50", "200", "1000", "none"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--evict-mb", type=int, default=48)
    p.add_argument("--gate-threads", type=int, nargs="+", default=[1, 6])
    p.add_argument("--scratch", default="serve_S.npz", help="build mode: where S is saved for the child process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    args.caps = [None if c == "none" else int(c) for c in args.caps]

    URM, _ = load_URM(args.dataset, version=args.version, verbose=False)
    URM = URM.tocsr()
    t = time.perf_counter()
    fn = sim.cosine if args.model == "cosine" else sim.rp3beta
    S = fn(URM.T, k=args.model_k, verbose=False)
    t_s = time.perf_counter() - t
    report = {"system": get_system_info(), "args": vars(args), "s_nnz": int(S.nnz)}
    if "build" in args.mode:
        mode_build(args, S, report)
    t = time.perf_counter()
    rec = sim.Recommender(S)
    print(
        f"URM {URM.shape} nnz {URM.nnz:,}; S ({args.model}) nnz {S.nnz:,} in {t_s:.1f}s; "
        f"model build {time.perf_counter() - t:.2f}s, {rec.n_items:,} items; max RSS {rss_mb():.0f} MB"
    )
    rng = np.random.default_rng(args.seed)
    users = rng.choice(URM.shape[0], size=args.requests)  # resampled, random order
    for m in args.mode:
        print(f"\n== {m} ==")
        if m == "latency":
            mode_latency(args, URM, rec, users, rng, report)
        elif m == "gate":
            mode_gate(args, URM, S, rec, report)
        elif m == "block":
            mode_block(args, URM, rec, report)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
