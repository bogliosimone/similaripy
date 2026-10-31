"""
Benchmark one similaripy build: one workload on one dataset, a JSON report.

Workloads:
  similarity  item-item similarity on URM.T (every similarity + fixed-parameter variants)
  recommend   dot_product(URM, S) with and without filter_cols=URM, S a cosine model
  small       millisecond calls on the synthetic matrix (smoke set, or --cases all)
  serve       similaripy.serving.ItemKNN: latency, batch vs kernel gate, block size, build

Usage (from the repo root; any similaripy install, 0.7.0 included):
    uv run --group bench python benchmarks/run.py similarity --dataset movielens --similarities cosine rp3beta
    uv run --group bench python benchmarks/run.py similarity --dataset yambda --version 500m --rounds 1 --warmup 0
    uv run --group bench python benchmarks/run.py serve --dataset yambda --mode latency gate
    uv run --isolated --no-project --with similaripy==0.7.0 \\
        python benchmarks/run.py similarity --dataset movielens --tag 0.7.0

See benchmarks/README.md for the protocol (which runs for which change).
"""

import argparse
import sys

import loaders
import report
import workloads


def block_size(raw):
    """--block-size: 'auto', 'none' (no blocking) or an int > 0."""
    if raw.lower() in ("auto", "none"):
        return None if raw.lower() == "none" else "auto"
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("block size must be > 0")
    return value


def parser():
    p = argparse.ArgumentParser(
        description="Benchmark one similaripy build", formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="workload", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset",
        default=None,
        choices=["synthetic", "movielens", "yambda", "hm"],
        help="default: movielens (small: synthetic)",
    )
    common.add_argument("--version", default=None, help="dataset version (default: movielens 32m, yambda 50m)")
    common.add_argument("--event-type", default="multi_event", choices=loaders.YAMBDA_EVENTS, help="yambda only")
    common.add_argument(
        "--rounds", type=int, default=None, help="timed rounds (default: similarity 1, small 3, recommend 5, serve 3)"
    )
    common.add_argument("--warmup", type=int, default=1, help="untimed rounds first (default: 1)")
    common.add_argument("--threads", type=int, default=0, help="num_threads of every call, 0 = all (default: 0)")
    common.add_argument("--format-output", default="csr", choices=["csr", "coo"], help="passed to every call")
    common.add_argument("--tag", default=None, help="report name prefix (default: the build's git or version)")
    common.add_argument("--note", default="", help="free text stored in the report")
    common.add_argument("--output-dir", default=str(report.HARNESS_DIR / "bench_results"))
    common.add_argument("--max-load", type=float, default=3.0, help="preflight: 1-min load above = not quiet")
    common.add_argument("--strict", action="store_true", help="refuse to run when not quiet (load, battery)")
    common.add_argument("--quiet", action="store_true", help="no per-round output")
    rss = argparse.ArgumentParser(add_help=False)
    rss.add_argument(
        "--rss",
        action="store_true",
        help="after the timed rounds, peak memory per case: its first call in a fresh process (untimed)",
    )

    s = sub.add_parser("similarity", parents=[common, rss], help="item-item similarity on URM.T")
    s.add_argument("--similarities", nargs="+", default=["dot_product", "cosine", "rp3beta"], metavar="NAME")
    s.add_argument("--k", type=int, default=100)
    s.add_argument("--shrink", type=float, default=None, help="passed only when set")
    s.add_argument("--threshold", type=float, default=None, help="passed only when set")
    s.add_argument("--block-size", type=block_size, default="auto", help="auto, none or an int > 0")
    s.add_argument("--column-order", default=None, choices=["popularity", "original"], help="private switch")

    r = sub.add_parser("recommend", parents=[common, rss], help="dot_product(URM, S) ± filter_cols=URM")
    r.add_argument("--model-k", type=int, default=100, help="neighbours per item in S (default: 100)")
    r.add_argument("--k", type=int, default=100, help="recommendations per user (default: 100)")

    m = sub.add_parser("small", parents=[common], help="millisecond calls on the synthetic matrix")
    m.add_argument("--cases", nargs="+", default=workloads.SMOKE_CASES, help='case names, or "all" (default: smoke)')
    m.add_argument("--reps", type=int, default=21, help="calls per measurement, median taken (default: 21)")

    v = sub.add_parser("serve", parents=[common], help="similaripy.serving.ItemKNN")
    v.add_argument("--mode", nargs="+", default=["latency"], choices=workloads.SERVE_MODES)
    v.add_argument("--model", default="cosine", choices=["cosine", "rp3beta"])
    v.add_argument("--model-k", type=int, default=100)
    v.add_argument("--k", type=int, default=100, help="recommendations per request (default: 100)")
    v.add_argument("--requests", type=int, default=20_000, help="latency: requests per round (default: 20000)")
    v.add_argument("--caps", nargs="+", default=["50", "200", "1000", "none"], help="latency: history caps")
    v.add_argument("--evict-mb", type=int, default=48, help="latency cold: MB read between requests")
    v.add_argument("--gate-threads", type=int, nargs="+", default=None, help="gate/block (default: 1 and all cores)")
    v.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    """Run the benchmark the arguments describe; returns the report path."""
    args = parser().parse_args(argv)
    # Workload-dependent defaults here: set_defaults on one subparser would change the shared parent's actions
    small = args.workload == "small"
    args.dataset = args.dataset or ("synthetic" if small else "movielens")
    args.rounds = args.rounds or {"small": 3, "recommend": 5, "serve": 3}.get(args.workload, 1)
    if small and args.dataset != "synthetic":
        sys.exit("error: the small preset runs on the synthetic dataset only")
    opts = vars(args)
    pre = report.preflight()
    report.check_quiet(pre, args.max_load, args.strict)
    URM, dataset = loaders.load(args.dataset, args.version, args.event_type, verbose=not args.quiet)
    try:
        cases, info = workloads.make_cases(opts, URM)
    except ValueError as e:
        sys.exit(f"error: {e}")

    config = {k: v for k, v in opts.items() if k not in ("workload", "dataset", "version", "event_type")}
    results = report.run_rounds(cases, args.rounds, args.warmup, verbose=not args.quiet)
    rep = report.build_report(args.workload, config, dataset, cases, results, pre, info)
    if opts.get("rss"):
        del URM, cases  # the children load their own copy
        for name, case in rep["cases"].items():
            case["rss_base_mb"], case["rss_peak_mb"] = report.in_fresh_process(workloads.rss_child, opts, name)
            if not args.quiet:
                print(f"[rss] {name:<24} peak +{case['rss_peak_mb']:.0f} MB over {case['rss_base_mb']:.0f} MB")
    path = report.write(rep, args.output_dir, args.tag)
    report.print_summary(rep)
    print(f"\nreport: {path}")
    return path


if __name__ == "__main__":
    main()
