"""
Inputs for the internal server's load generator (server plan S2): a saved model, request bodies, Python's answers.

Usage (from the repo root):
    uv run --group bench python benchmarks/server_prep.py --dataset movielens --out ~/.cache/similaripy-bench/server/ml-32m
    cargo run --release -p similaripy-server -- --model OUT/model
    cargo run --release -p similaripy-server --bin loadgen -- --requests OUT/requests-c200-k100.jsonl \
        --check OUT/expected-c200-k100.jsonl
    cargo run --release -p similaripy-server --bin loadgen -- --requests OUT/requests-c200-k100.jsonl

Writes, in OUT:
  model/          ItemKNN(S).save: S = the serve workload's model (cosine, k=100), from the bench cache (--model-cache)
  requests-c{CAP}-k{K}.jsonl  per --caps x --ks: one /recommend body per line: a random user's URM row, capped like
                  the serve workload (CAP random items; keep it <= the server's --max-history, or the server cuts it),
                  k, exclude_seen; every other request with float64 weights (the same users and weights for every k)
  expected-c{CAP}-k{K}.jsonl  ItemKNN.load(OUT/model).recommend of the first --check requests: ids + score bits
"""

import argparse
import json
from pathlib import Path

import loaders
import numpy as np
import workloads


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="movielens")
    p.add_argument("--version", default=None)
    p.add_argument("--event-type", default="multi_event")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--requests", type=int, default=100_000, help="request bodies (default: 100000)")
    p.add_argument("--caps", type=int, nargs="+", default=[200], help="history caps (default: 200)")
    p.add_argument("--ks", type=int, nargs="+", default=[100], help="k per request (default: 100)")
    p.add_argument("--check", type=int, default=1000, help="expected answers for the first N requests (default: 1000)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from similaripy.serving import ItemKNN

    URM, dataset = loaders.load(args.dataset, args.version, args.event_type)
    opts = {"model": "cosine", "model_k": 100, "model_cache": True, "dataset_key": dataset["key"]}
    S, _ = workloads._serve_model(URM, opts)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "model").exists():  # a rerun keeps it: expected-*.jsonl are computed from what is there
        ItemKNN(S).save(args.out / "model")

    rec = ItemKNN.load(args.out / "model")  # what the server loads
    for cap in args.caps:
        rng = np.random.default_rng(args.seed)
        users = rng.choice(URM.shape[0], size=args.requests)
        hists = workloads._histories(URM, users, cap, rng)
        # Every other request weighted (float64 as a client sends them): the server's JSON → f32 rounding vs numpy's
        weights = [rng.random(h.size) * 2 + 0.05 if i % 2 else None for i, h in enumerate(hists)]
        for k in args.ks:
            with open(args.out / f"requests-c{cap}-k{k}.jsonl", "w") as f:
                for h, w in zip(hists, weights, strict=True):
                    body = {"ids": h.tolist(), "k": k, "exclude_seen": True}
                    if w is not None:
                        body["weights"] = w.tolist()
                    f.write(json.dumps(body) + "\n")
            with open(args.out / f"expected-c{cap}-k{k}.jsonl", "w") as f:
                for h, w in zip(hists[: args.check], weights, strict=False):
                    ids, scores = rec.recommend(h, weights=w, k=k, exclude_seen=True)
                    f.write(json.dumps({"ids": ids.tolist(), "score_bits": scores.view(np.uint32).tolist()}) + "\n")
            print(f"{args.out}: {len(hists)} requests (cap {cap}, k {k}), {min(args.check, len(hists))} expected")


if __name__ == "__main__":
    main()
