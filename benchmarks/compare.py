"""
Compare benchmark reports: the first is the reference, every other one is
shown as the change against it, next to each run's spread (the noise).

Usage:
    python benchmarks/compare.py REF.json OTHER.json [MORE.json ...] [--metric kernel]

Metrics: total (default), prep, kernel, output, and any numeric extra a
workload records (p50_us, p99_us, gate_s, rss_peak_mb, ...). Reports from
before the harness rebuild (no schema_version) aren't read.
"""

import argparse
import json
import sys
from pathlib import Path

PHASES = ("total", "prep", "kernel", "output")


def load(path):
    """A report, refusing pre-rebuild or newer-schema files."""
    rep = json.loads(Path(path).read_text())
    if rep.get("schema_version") != 1:
        raise ValueError(f"{path}: not a schema 1 report (pre-rebuild reports aren't read)")
    return rep


def value(case, metric):
    """A case's median for a phase, or its extra; None when not recorded (e.g. no phase timer)."""
    v = case["median"].get(metric) if metric in PHASES else case.get(metric)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def change(ref, other):
    """other / ref - 1, or None."""
    return None if ref is None or other is None or ref == 0 else other / ref - 1


def label(rep):
    """Short name of a report: its tag-like build id and dataset."""
    b = rep["build"]
    return f"{b['git'] or b['similaripy_version']} {rep['dataset']['key']}"


def compare(reports, metric):
    """Rows of (case, ref value, [(value, change, spread) per other report]) over the cases in every report."""
    ref, others = reports[0], reports[1:]
    names = [n for n in ref["cases"] if all(n in r["cases"] for r in others)]
    rows = []
    for n in names:
        r = value(ref["cases"][n], metric)
        cols = []
        for o in others:
            c = o["cases"][n]
            v = value(c, metric)
            cols.append((v, change(r, v), c["spread"]))
        rows.append((n, r, ref["cases"][n]["spread"], cols))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare benchmark reports (first = reference)")
    p.add_argument("reports", nargs="+")
    p.add_argument("--metric", default="total")
    args = p.parse_args(argv)
    if len(args.reports) < 2:
        p.error("give a reference and at least one report")
    try:
        reports = [load(f) for f in args.reports]
    except ValueError as e:
        sys.exit(f"error: {e}")
    if len({r["workload"] for r in reports}) > 1 or len({r["dataset"]["key"] for r in reports}) > 1:
        print("WARNING: different workloads or datasets", file=sys.stderr)
    for i, r in enumerate(reports):
        print(f"[{i}] {label(r)}  {r['timestamp']}  {r['config'].get('note') or ''}")
    print(f"\nmetric: {args.metric} (change vs [0]; spread = (max - min) / median over rounds)")
    fmt = lambda v: "-" if v is None else f"{v:.4g}"
    for name, r, spread, cols in compare(reports, args.metric):
        cells = "  ".join(
            f"[{i + 1}] {fmt(v)} ({'-' if d is None else f'{d:+.1%}'}, spread {s:.1%})"
            for i, (v, d, s) in enumerate(cols)
        )
        print(f"{name:<24} [0] {fmt(r)} (spread {spread:.1%})  {cells}")


if __name__ == "__main__":
    main()
