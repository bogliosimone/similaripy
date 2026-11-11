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

Several candidates against one base: repeat --cand as NAME=SPEC. Each pair runs
the base once and every candidate once, in rotated order (base, c1, .., ck, then
ck, .., c1, base in the next pair), so the base isn't rerun per candidate. One
summary per candidate, abba_{tag}-{NAME}.json; a single unnamed --cand keeps
abba_{tag}.json.

Usage:
    python benchmarks/abba.py --base ~/similaripy-wt/v1-main/.venv/bin/python --cand .venv/bin/python \\
        --tag d75-check --pairs 2 -- similarity --dataset yambda --version 500m --similarities cosine --warmup 0
    python benchmarks/abba.py --base p0/.venv/bin/python --cand p4=p4/.venv/bin/python \\
        --cand p32=p32/.venv/bin/python --tag band -- serve --dataset movielens --mode block

Reading it (decisions D40 P6 / D57, applied by the reader): a change counts
when every pair shows it with the same sign and it is above the runs' spread
(`compare.py` on the run reports shows each one's spread).
"""

import argparse
import json
import os
import platform
import re
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


# Phases, then the extras a verdict also reads (serving: tail latency, CPU per request; D90)
EXTRAS = ("p99_us", "cpu_us_per_request")
METRICS = ("total", "kernel", *EXTRAS)


def cand_sides(specs):
    """[(name, spec)] of the candidates: one plain --cand is 'cand', named ones are NAME=SPEC."""
    if len(specs) == 1 and "=" not in specs[0]:
        return [("cand", specs[0])]
    sides = []
    for s in specs:
        name, sep, spec = s.partition("=")
        if not sep or not re.fullmatch(r"[\w.+]+", name) or name == "base":
            raise ValueError(
                f"--cand {s!r}: several candidates are NAME=SPEC (NAME: letters, digits, '.', '_', '+'; not base)"
            )
        sides.append((name, spec))
    if len({n for n, _ in sides}) < len(sides):
        raise ValueError("candidate names repeat")
    return sides


def run_tags(tag, n_pairs, sides=("base", "cand")):
    """{run tag: (side, pair)} of one run."""
    return {f"{tag}-{side}-{i}": (side, i) for i in range(1, n_pairs + 1) for side in sides}


def pairs_table(out_dir, tag, n_pairs, metric, cand="cand"):
    """{case: [(base value, cand value, change) per pair]} from the run reports."""
    reps = {}
    tags = run_tags(tag, n_pairs, ("base", cand))
    for f in out_dir.glob("*.json"):
        rep = json.loads(f.read_text())
        side_pair = tags.get(rep.get("config", {}).get("tag"))
        if side_pair is not None:
            reps[side_pair] = compare.load(f)
    table = {}
    for i in range(1, n_pairs + 1):
        base, cand_rep = reps[("base", i)], reps[(cand, i)]
        for name in base["cases"]:
            if name in cand_rep["cases"]:
                b, c = compare.value(base["cases"][name], metric), compare.value(cand_rep["cases"][name], metric)
                table.setdefault(name, []).append((b, c, compare.change(b, c)))
    return table


def main(argv=None):
    p = argparse.ArgumentParser(
        description="ABBA timing of two (or more) similaripy builds", usage="%(prog)s [options] -- RUN_ARGS"
    )
    p.add_argument("--base", required=True, help="interpreter path or pypi:VERSION")
    p.add_argument(
        "--cand",
        required=True,
        action="append",
        help="interpreter path or pypi:VERSION; repeat as NAME=SPEC for several candidates against one base",
    )
    p.add_argument("--tag", type=run.tag_name, required=True, help="names the runs: {tag}-{side}-{pair}")
    p.add_argument("--pairs", type=run.positive, default=2)
    p.add_argument("--output-dir", default=None, help="default: bench_results/abba-{tag}")
    p.add_argument("run_args", nargs=argparse.REMAINDER, help="after --: run.py arguments (workload first)")
    args = p.parse_args(argv)
    try:
        cands = cand_sides(args.cand)
    except ValueError as e:
        p.error(str(e))
    sides = ["base", *(n for n, _ in cands)]
    run_args = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args
    if not run_args:
        p.error("give the run.py arguments after --")
    for forbidden in ("--tag", "--output-dir", "--note"):
        if any(a == forbidden or a.startswith(forbidden + "=") for a in run_args):
            p.error(f"{forbidden} is set by abba.py")

    parsed = run.parser().parse_args(run_args)
    out_dir = Path(args.output_dir or report.HARNESS_DIR / "bench_results" / f"abba-{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = run_tags(args.tag, args.pairs, sides)
    for f in out_dir.glob("*.json"):  # an earlier run with these tags would mix into the pairs
        if json.loads(f.read_text()).get("config", {}).get("tag") in tags:
            p.error(f"{out_dir} already has reports tagged {args.tag}-*: use another --tag or --output-dir")
    if parsed.workload != "small":  # cache the dataset once, here (pandas), for every side
        loaders.load(parsed.dataset or "movielens", parsed.version, parsed.event_type)

    cmds = {side: side_command(spec) for side, spec in (("base", args.base), *cands)}
    for cmd in cmds.values():
        check_fresh(cmd, out_dir)
    print(f"preflight: {report.preflight()}")
    for i in range(1, args.pairs + 1):
        for side in sides if i % 2 else sides[::-1]:
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

    for cand, _ in cands:
        summarize(out_dir, args, cand, len(cands) == 1 and cand == "cand")


def summarize(out_dir, args, cand, unnamed):
    """Print the per-pair changes of one candidate and write its summary JSON."""
    summary = {}
    for metric in METRICS:
        table = pairs_table(out_dir, args.tag, args.pairs, metric, cand)
        if metric in EXTRAS and all(d is None for rows in table.values() for _, _, d in rows):
            continue  # an extra this workload doesn't record
        print(f"\n{metric} ({cand} vs base per pair; same sign in every pair = a candidate change)")
        for name, rows in table.items():
            deltas = [d for _, _, d in rows]
            if any(d is None for d in deltas):
                print(f"{name:<24} -")
                continue
            same = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
            cells = "  ".join(f"{b:.4g} -> {c:.4g} ({d:+.1%})" for b, c, d in rows)
            print(f"{name:<24} {cells}  {'same sign' if same else 'mixed'}")
        summary[metric] = table
    name = f"abba_{args.tag}.json" if unnamed else f"abba_{args.tag}-{cand}.json"
    (out_dir / name).write_text(json.dumps({"args": vars(args), "pairs": summary}, indent=2))


if __name__ == "__main__":
    main()
