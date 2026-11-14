"""
Inputs for the internal server's load generator (server plan S2): a saved model, request bodies, Python's answers.

Usage (from the repo root):
    uv run --group bench python benchmarks/server_prep.py --dataset movielens --out ~/.cache/similaripy-bench/server/ml-32m
    cargo run --release -p similaripy-server -- --model OUT/model
    cargo run --release -p similaripy-server --bin loadgen -- --requests OUT/requests.jsonl --check OUT/expected.jsonl
    cargo run --release -p similaripy-server --bin loadgen -- --requests OUT/requests.jsonl

Writes, in OUT:
  model/          ItemKNN(S).save: S = the serve workload's model (cosine, k=100), from the bench cache (--model-cache)
  requests.jsonl  one /recommend body per line: a random user's URM row, capped like the serve workload (--cap random
                  items; keep it <= the server's --max-history, or the server cuts it), k, exclude_seen; every
                  other request with float64 weights
  expected.jsonl  ItemKNN.load(OUT/model).recommend of the first --check requests: ids + score bits (uint32)
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
    p.add_argument("--cap", type=int, default=200, help="history cap (default: 200)")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--check", type=int, default=1000, help="expected answers for the first N requests (default: 1000)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from similaripy.serving import ItemKNN

    URM, dataset = loaders.load(args.dataset, args.version, args.event_type)
    opts = {"model": "cosine", "model_k": 100, "model_cache": True, "dataset_key": dataset["key"]}
    S, _ = workloads._serve_model(URM, opts)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "model").exists():  # a rerun keeps it: expected.jsonl is computed from what is there
        ItemKNN(S).save(args.out / "model")

    rng = np.random.default_rng(args.seed)
    users = rng.choice(URM.shape[0], size=args.requests)
    hists = workloads._histories(URM, users, args.cap, rng)
    # Every other request weighted (float64 as a client sends them): the server's JSON → f32 rounding vs numpy's
    weights = [rng.random(h.size) * 2 + 0.05 if i % 2 else None for i, h in enumerate(hists)]
    with open(args.out / "requests.jsonl", "w") as f:
        for h, w in zip(hists, weights, strict=True):
            body = {"ids": h.tolist(), "k": args.k, "exclude_seen": True}
            if w is not None:
                body["weights"] = w.tolist()
            f.write(json.dumps(body) + "\n")

    rec = ItemKNN.load(args.out / "model")  # what the server loads
    with open(args.out / "expected.jsonl", "w") as f:
        for h, w in zip(hists[: args.check], weights, strict=False):
            ids, scores = rec.recommend(h, weights=w, k=args.k, exclude_seen=True)
            f.write(json.dumps({"ids": ids.tolist(), "score_bits": scores.view(np.uint32).tolist()}) + "\n")
    print(
        f"{args.out}: model, {len(hists)} requests (cap {args.cap}, k {args.k}), {min(args.check, len(hists))} expected"
    )


if __name__ == "__main__":
    main()
