"""
Recommendation-step benchmark: user x item scores from a precomputed item model.

The item-item benchmarks (run_benchmarks.py) have no realistic matrix-mode
filter: their output is items x items. The recommendation step is where one
lives: scores = dot_product(URM, S) with filter_cols=URM ("don't recommend
items the user has already seen"), as one library batch call over all users
(not a server's per-request path). S is a cosine item model
(k=--model-k), built once and not timed. S and the scores use
--format-output (default csr).

Cases, interleaved per round (ABAB): 'recommend' (no filter) and
'recommend_filter_seen' (filter_cols=URM). Prints median total / prep /
kernel / output per case; --json writes them.

Usage:
    python run_recommend_benchmarks.py --dataset movielens --version 32m
    python run_recommend_benchmarks.py --dataset yambda --version 50m --rounds 3
"""

import argparse
import json
import statistics
import time

from benchmark import get_system_info
from dataset_loaders import load_URM

import similaripy as sim

PHASES = ("total", "prep", "kernel", "output")


def run_once(URM, S, k, num_threads, filter_seen, format_output):
    kw = {"filter_cols": URM} if filter_seen else {}
    start = time.perf_counter()
    sim.dot_product(URM, S, k=k, num_threads=num_threads, verbose=False, format_output=format_output, **kw)
    total = time.perf_counter() - start
    t = sim._s_plus._last_timings
    return {"total": total, "prep": total - t["kernel"] - t["output"], **t}


def main():
    parser = argparse.ArgumentParser(description="Recommendation-step benchmark for similaripy")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--version", default=None)
    parser.add_argument("--model-k", type=int, default=100, help="Neighbours per item in S (default: 100)")
    parser.add_argument("--k", type=int, default=100, help="Recommendations per user (default: 100)")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--format-output",
        default="csr",
        choices=["csr", "coo"],
        help="Output format of S and of the timed call (default: csr)",
    )
    parser.add_argument("--json", type=str, default=None, help="Write medians to this JSON file")
    args = parser.parse_args()

    URM, _ = load_URM(args.dataset, version=args.version, verbose=False)
    URM = URM.tocsr()
    S = sim.cosine(URM.T, k=args.model_k, verbose=False, format_output=args.format_output)
    print(f"URM {URM.shape} nnz {URM.nnz:,}; S nnz {S.nnz:,}")

    cases = {"recommend": False, "recommend_filter_seen": True}
    times = {c: [] for c in cases}
    for r in range(args.warmup + args.rounds):
        for case, filter_seen in cases.items():
            t = run_once(URM, S, args.k, args.threads, filter_seen, args.format_output)
            label = "warmup" if r < args.warmup else f"round {r - args.warmup + 1}/{args.rounds}"
            print(f"[{label}] {case:<22} " + "  ".join(f"{p} {t[p]:.3f}s" for p in PHASES))
            if r >= args.warmup:
                times[case].append(t)

    medians = {c: {p: statistics.median(t[p] for t in ts) for p in PHASES} for c, ts in times.items()}
    print(f"\n{'case':<22} " + " ".join(f"{p:>8}" for p in PHASES))
    for case, m in medians.items():
        print(f"{case:<22} " + " ".join(f"{m[p]:8.3f}" for p in PHASES))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"system": get_system_info(), "args": vars(args), "medians": medians, "times": times},
                f,
                indent=2,
                default=str,
            )


if __name__ == "__main__":
    main()
