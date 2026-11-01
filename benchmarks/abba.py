"""
A/B timing of two similaripy builds: alternating processes, per-pair changes.

Both sides run this checkout's benchmarks/run.py (the same harness, so a
case means the same thing on both), each with its own interpreter, one
process per (side, pair), order base-cand then cand-base (ABBA). A side is:
  - a Python interpreter, e.g. ~/similaripy-wt/v1-main/.venv/bin/python
    (a worktree with its own venv; refused if its extension is older than its
    Rust sources: .venv/bin/python doesn't rebuild the way `uv run` does);
  - pypi:VERSION, e.g. pypi:0.7.0: `uv run --isolated --no-project` with that
    wheel, numpy/scipy pinned to this interpreter's versions.
The dataset is loaded (and cached) once here first, so every side reads the
same .npz. 0.7.0 has no phase timer: compare it on total time only.

Usage:
    python benchmarks/abba.py --base ~/similaripy-wt/v1-main/.venv/bin/python --cand .venv/bin/python \\
        --tag d75-check --pairs 2 -- similarity --dataset yambda --version 500m --similarities cosine --warmup 0

Reading it (decisions D40 P6 / D57, applied by the reader): a change counts
when every pair shows it with the same sign and it is above the runs' spread
(`compare.py` on the run reports shows each one's spread).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import compare
import loaders
import numpy
import report
import run
import scipy

RUN = Path(run.__file__).resolve()


def side_command(spec):
    """The interpreter command of one side."""
    if spec.startswith("pypi:"):
        withs = [f"similaripy=={spec[5:]}", f"numpy=={numpy.__version__}", f"scipy=={scipy.__version__}"]
        if platform.system() != "Darwin":
            withs.append("psutil")  # --rss off macOS
        return [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            sys.executable,  # the same CPython build (version, free-threaded or not) as this side
            *(a for w in withs for a in ("--with", w)),
            "python",
        ]
    return [os.path.abspath(Path(spec).expanduser())]  # absolute, not resolved: a venv python is a symlink


def check_fresh(cmd, cwd):
    """Refuse a source-tree build older than its Rust sources (a stale editable install)."""
    probe = "import similaripy, pathlib; print(pathlib.Path(similaripy.__file__).resolve().parent)"
    r = subprocess.run([*cmd, "-c", probe], capture_output=True, text=True, cwd=cwd, check=True)
    package = Path(r.stdout.strip().splitlines()[-1])
    root = package.parent
    if not (root / "Cargo.toml").exists():
        return  # a wheel
    built = max((p.stat().st_mtime for p in package.glob("_core*")), default=0)
    sources = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        *root.glob("crates/**/*.rs"),
        *root.glob("crates/**/Cargo.toml"),
    ]
    newest = max(p.stat().st_mtime for p in sources if p.exists())
    if newest > built:
        sys.exit(f"error: {root}: the extension is older than its Rust sources; rebuild it (make install-dev)")


def run_tags(tag, n_pairs):
    """{run tag: (side, pair)} of one ABBA."""
    return {f"{tag}-{side}-{i}": (side, i) for i in range(1, n_pairs + 1) for side in ("base", "cand")}


def pairs_table(out_dir, tag, n_pairs, metric):
    """{case: [(base value, cand value, change) per pair]} from the run reports."""
    reps = {}
    for f in out_dir.glob("*.json"):
        rep = json.loads(f.read_text())
        side_pair = run_tags(tag, n_pairs).get(rep.get("config", {}).get("tag"))
        if side_pair is not None:
            reps[side_pair] = compare.load(f)
    table = {}
    for i in range(1, n_pairs + 1):
        base, cand = reps[("base", i)], reps[("cand", i)]
        for name in base["cases"]:
            if name in cand["cases"]:
                b, c = compare.value(base["cases"][name], metric), compare.value(cand["cases"][name], metric)
                table.setdefault(name, []).append((b, c, compare.change(b, c)))
    return table


def main(argv=None):
    p = argparse.ArgumentParser(
        description="ABBA timing of two similaripy builds", usage="%(prog)s [options] -- RUN_ARGS"
    )
    p.add_argument("--base", required=True, help="interpreter path or pypi:VERSION")
    p.add_argument("--cand", required=True, help="interpreter path or pypi:VERSION")
    p.add_argument("--tag", type=run.tag_name, required=True, help="names the runs: {tag}-{side}-{pair}")
    p.add_argument("--pairs", type=run.positive, default=2)
    p.add_argument("--output-dir", default=None, help="default: bench_results/abba-{tag}")
    p.add_argument("run_args", nargs=argparse.REMAINDER, help="after --: run.py arguments (workload first)")
    args = p.parse_args(argv)
    run_args = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args
    if not run_args:
        p.error("give the run.py arguments after --")
    for forbidden in ("--tag", "--output-dir", "--note"):
        if any(a == forbidden or a.startswith(forbidden + "=") for a in run_args):
            p.error(f"{forbidden} is set by abba.py")

    parsed = run.parser().parse_args(run_args)
    out_dir = Path(args.output_dir or report.HARNESS_DIR / "bench_results" / f"abba-{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = run_tags(args.tag, args.pairs)
    for f in out_dir.glob("*.json"):  # an earlier run with these tags would mix into the pairs
        if json.loads(f.read_text()).get("config", {}).get("tag") in tags:
            p.error(f"{out_dir} already has reports tagged {args.tag}-*: use another --tag or --output-dir")
    if parsed.workload != "small":  # cache the dataset once, here (pandas), for every side
        loaders.load(parsed.dataset or "movielens", parsed.version, parsed.event_type)

    cmds = {side: side_command(spec) for side, spec in (("base", args.base), ("cand", args.cand))}
    for cmd in cmds.values():
        check_fresh(cmd, out_dir)
    print(f"preflight: {report.preflight()}")
    for i in range(1, args.pairs + 1):
        for side in ("base", "cand") if i % 2 else ("cand", "base"):
            tag = f"{args.tag}-{side}-{i}"
            print(f"== pair {i} {side}: {' '.join(cmds[side])}", flush=True)
            subprocess.run(
                [
                    *cmds[side],
                    str(RUN),
                    *run_args,
                    "--quiet",
                    "--tag",
                    tag,
                    "--output-dir",
                    str(out_dir),
                    "--note",
                    f"abba {args.tag} {side} pair {i}",
                ],
                cwd=out_dir,
                check=True,
            )

    summary = {}
    for metric in ("total", "kernel"):
        table = pairs_table(out_dir, args.tag, args.pairs, metric)
        print(f"\n{metric} (cand vs base per pair; same sign in every pair = a candidate change)")
        for name, rows in table.items():
            deltas = [d for _, _, d in rows]
            if any(d is None for d in deltas):
                print(f"{name:<24} -")
                continue
            same = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
            cells = "  ".join(f"{b:.4g} -> {c:.4g} ({d:+.1%})" for b, c, d in rows)
            print(f"{name:<24} {cells}  {'same sign' if same else 'mixed'}")
        summary[metric] = table
    (out_dir / f"abba_{args.tag}.json").write_text(json.dumps({"args": vars(args), "pairs": summary}, indent=2))


if __name__ == "__main__":
    main()
