"""
The core's benchmark gate: two builds, ABBA (abba.py) on a fixed list of steps, one verdict per case.

    python benchmarks/gate.py quick --base OLD/.venv/bin/python --cand NEW/.venv/bin/python --tag my-change

Tiers (the decision rests on Y500M, the library's scale; ML-32M and Y50M are cheap early signals):
  - quick (~30 min): Y50M then Y500M, kernel (cosine, rp3beta) and serving batch, --threads (the P-cores);
  - full (~60 min, once per PR): quick + Y500M serving at 1 thread, ML-32M kernel / serving / latency, build.

Never idle: every step starts at once (no waiting for a quiet machine). A sampler records the 1-min load
and the CPU used by every process outside this gate's own process tree every 10 s (CPU time differences,
the same on macOS and Linux; a VM's steal time isn't seen); a step whose window had > 100% such CPU in
> 10% of its samples is rerun once at the end, while time remains. A step starts only if its estimate
(scaled by --pairs) ends before --until; a failed or skipped step is a row of the verdict; exit 1 on a failure.
An output dir that already holds a run is refused. Output, in --output-dir: one folder per step (abba.py's reports),
timeline.tsv, steps.tsv, verdict.md (win / loss = same sign in every pair, each >= 3%, D40 P6).
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.03  # a change counts at >= 3% in every pair (D40 P6)
BURST_CPU, BURST_SHARE = 100.0, 0.10  # % CPU outside the gate's tree; share of a step's samples
SAMPLE_S = 10

ML, Y50, Y500 = ["--dataset", "movielens"], ["--dataset", "yambda"], ["--dataset", "yambda", "--version", "500m"]
SIMS = ["--similarities", "cosine", "rp3beta"]
BLOCK = ["--mode", "block", "--rounds", "1", "--model-cache"]


def steps(tier, dry=False, threads=6):
    """(estimate in minutes, name, run.py arguments) in run order; threads = the performance cores."""
    t = str(threads)
    ml, y50, y500 = ([["--dataset", "synthetic"]] * 3) if dry else (ML, Y50, Y500)
    quick = [
        (3, "y50m-kernel", ["similarity", *y50, *SIMS, "--threads", t, "--warmup", "0"]),
        (3, "y50m-block", ["serve", *y50, *BLOCK, "--block-threads", t]),
        (15, "y500m-kernel", ["similarity", *y500, *SIMS, "--threads", t, "--warmup", "0"]),
        (8, "y500m-block", ["serve", *y500, *BLOCK, "--block-threads", t]),
    ]
    full = [
        (15, "y500m-block-t1", ["serve", *y500, *BLOCK, "--block-threads", "1"]),
        (3, "ml32m-kernel", ["similarity", *ml, *SIMS, "--threads", t]),
        (3, "ml32m-block", ["serve", *ml, *BLOCK, "--block-threads", "1", t]),
        (5, "ml32m-lat", ["serve", *ml, "--mode", "latency", "--caps", "50", "none", "--requests", "5000",
                          "--rounds", "1", "--model-cache"]),
        (3, "y50m-build", ["serve", *y50, "--mode", "build", "--rounds", "1"]),
    ]  # fmt: skip
    return quick + (full if tier == "full" else [])


def cpu_seconds(text):
    """ps's cumulative CPU time: [dd-]hh:mm:ss (Linux) or mm:ss.ss (macOS) -> seconds."""
    days, _, rest = text.rpartition("-")
    secs = 0.0
    for part in rest.split(":"):
        secs = secs * 60 + float(part)
    return secs + (int(days) * 86400 if days else 0)


def process_cpu():
    """{pid: (ppid, cumulative CPU seconds)} of every process (ps: macOS and Linux)."""
    out = subprocess.run(
        ["ps", "-Ao", "pid=,ppid=,time="], check=True, capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}
    ).stdout
    procs = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 3:
            procs[int(fields[0])] = (int(fields[1]), cpu_seconds(fields[2]))
    return procs


def background_cpu(before, after, seconds):
    """% CPU used between two process_cpu() samples by the processes outside this process's tree
    (CPU time differences: the same meaning on macOS and Linux, unlike ps's pcpu)."""
    children = {}
    for pid, (ppid, _) in after.items():
        children.setdefault(ppid, []).append(pid)
    ours, todo = set(), [os.getpid()]
    while todo:
        p = todo.pop()
        ours.add(p)
        todo += children.get(p, [])
    used = sum(cpu - before.get(pid, (0, 0.0))[1] for pid, (_, cpu) in after.items() if pid not in ours)
    return max(used, 0.0) / seconds * 100


class Sampler(threading.Thread):
    """(time, load, background %CPU) every SAMPLE_S seconds, also appended to timeline.tsv. A sampling
    error is counted (errors), never fatal: steps.tsv and the verdict say when the sampler was down."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.rows, self.errors, self.stop = path, [], 0, threading.Event()
        path.write_text("epoch\tload1\tbackground_cpu\n")

    def run(self):
        prev = None
        while not self.stop.is_set():
            try:
                now, procs = time.time(), process_cpu()
                if prev is not None:
                    row = (now, os.getloadavg()[0], background_cpu(prev[1], procs, now - prev[0]))
                    self.rows.append(row)
                    with self.path.open("a") as f:
                        f.write("\t".join(f"{v:.2f}" for v in row) + "\n")
                prev = (now, procs)
            except Exception as e:  # noqa: BLE001 - the gate runs on without load data
                self.errors += 1
                print(f"{time.strftime('%H:%M')} sampler error: {e!r}", flush=True)
            self.stop.wait(SAMPLE_S)

    def window(self, t0, t1):
        """(mean load, share of samples with a burst, samples) in [t0, t1]."""
        rows = [r for r in self.rows if t0 <= r[0] <= t1]
        if not rows:
            return 0.0, 0.0, 0
        return sum(r[1] for r in rows) / len(rows), sum(r[2] > BURST_CPU for r in rows) / len(rows), len(rows)


def verdict(rels):
    if any(r is None for r in rels):
        return "n/a"  # a missing phase or a zero reference (compare.change)
    if not (all(r < 0 for r in rels) or all(r > 0 for r in rels)):
        return "neutral"
    if min(abs(r) for r in rels) < THRESHOLD:
        return "small"
    return "win" if rels[0] < 0 else "loss"


def verdict_rows(out, results):
    """One markdown row per (step, case); a failed or skipped step gets one row saying so. A rerun stands
    in for its step."""
    rows = []
    for name, s in results.items():  # in run order; a rerun overwrote its step's entry
        if s["status"] != "ok":
            rows.append(f"| {name} | - | - | - | **{s['status']}** | |")
            continue
        pairs = json.loads((out / s["dir"] / f"abba_{s['dir']}.json").read_text())["pairs"]
        has_kernel = any(p[2] is not None for ps in pairs.get("kernel", {}).values() for p in ps)
        metric = "kernel" if has_kernel else "total"  # serving steps have no kernel phase
        for case, ps in pairs[metric].items():
            rels = [p[2] for p in ps]
            cells = " / ".join("-" if r is None else f"{100 * r:+.1f}%" for r in rels)
            notes = [n for n, on in (("noisy", s["burst"] > BURST_SHARE), ("rerun", s["rerun"])) if on]
            rows.append(f"| {name} | {case} | {metric} | {cells} | **{verdict(rels)}** | {' '.join(notes)} |")
    return rows


def until(text):
    """HH:MM or H:MM -> datetime.time (argparse type)."""
    try:
        hh, mm = text.split(":")
        return dt.time(int(hh), int(mm))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected HH:MM, got {text!r}") from None


def deadline(t):
    """The next occurrence of t (today or tomorrow) as an epoch."""
    now = dt.datetime.now()
    at = dt.datetime.combine(now.date(), t)
    return (at if at > now else at + dt.timedelta(days=1)).timestamp()


def main(argv=None):
    p = argparse.ArgumentParser(description="The core's benchmark gate: two builds, ABBA, one verdict per case")
    p.add_argument("tier", choices=["quick", "full"])
    p.add_argument("--base", required=True, help="interpreter path or pypi:VERSION (as abba.py)")
    p.add_argument("--cand", required=True, help="interpreter path or pypi:VERSION (as abba.py)")
    p.add_argument("--tag", required=True, help="prefix of every step's runs")
    p.add_argument("--pairs", type=int, default=2, help="ABBA pairs per step, >= 2 (default: 2)")
    p.add_argument("--until", type=until, default=None, help="HH:MM: a step starts only if its estimate ends before")
    p.add_argument("--output-dir", default=None, help="default: bench_results/gate-{tag}")
    p.add_argument("--threads", type=int, default=6, help="the performance cores (default: 6, the M2 Pro's)")
    p.add_argument("--dry", action="store_true", help="every step on the synthetic dataset (plumbing check)")
    args = p.parse_args(argv)
    if args.pairs < 2 and not args.dry:
        p.error("--pairs must be >= 2: 'same sign in every pair' means nothing with one")

    out = Path(args.output_dir or HERE / "bench_results" / f"gate-{args.tag}").resolve()
    if (out / "steps.tsv").exists():
        p.error(f"{out} already holds a gate run: use another --tag or --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    end = deadline(args.until) if args.until else float("inf")
    plan = {name: (est, run_args) for est, name, run_args in steps(args.tier, args.dry, args.threads)}
    scale = args.pairs / 2  # the estimates are for 2 pairs
    sampler = Sampler(out / "timeline.tsv")
    sampler.start()
    log = (out / "steps.tsv").open("w")
    log.write("step\tstart\tend\trc\tload_mean\tburst_share\tsamples\n")
    results, flagged = {}, []

    def run_step(name, rerun=False):
        est = plan[name][0] * scale
        if time.time() + est * 60 > end:
            print(f"{time.strftime('%H:%M')} SKIP {name} (needs {est:.0f} min)", flush=True)
            if not rerun:
                results[name] = {"status": "skipped (--until)"}
            return
        d = f"{args.tag}-{name}" + ("-rerun" if rerun else "")
        print(f"{time.strftime('%H:%M')} START {d} (est {est:.0f} min)", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(HERE / "abba.py"), "--base", args.base, "--cand", args.cand, "--tag", d,
               "--pairs", str(args.pairs), "--output-dir", str(out / d), "--", *plan[name][1]]  # fmt: skip
        with (out / f"{d}.log").open("w") as f:
            rc = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT).returncode
        t1 = time.time()
        load, burst, n = sampler.window(t0, t1)
        log.write(f"{d}\t{t0:.0f}\t{t1:.0f}\t{rc}\t{load:.2f}\t{burst:.2f}\t{n}\n")
        log.flush()
        status = "ok" if rc == 0 else f"FAILED rc={rc} (see {d}.log)"
        print(f"{time.strftime('%H:%M')} END {d} {status} {(t1 - t0) / 60:.0f} min load {load:.1f} bursts {burst:.2f}",
              flush=True)  # fmt: skip
        if rc == 0 or not rerun:  # a failed rerun keeps the first run's result
            results[name] = {"status": status, "dir": d, "burst": burst, "rerun": rerun}
        if rc == 0 and burst > BURST_SHARE and not rerun:
            flagged.append(name)
            print(f"{time.strftime('%H:%M')} FLAG {d} for rerun (background CPU)", flush=True)

    try:
        for name in plan:
            run_step(name)
        for name in flagged:
            run_step(name, rerun=True)
    finally:  # an interrupt or an error still leaves the verdict of what ran
        sampler.stop.set()
        down = f" **The load sampler failed {sampler.errors} times: bursts may be missed.**" if sampler.errors else ""
        text = [
            f"# Gate {args.tier}: {args.tag}",
            "",
            (
                f"base `{args.base}`, cand `{args.cand}`, {args.pairs} pairs. win / loss = same sign in every pair,"
                f" each >= {THRESHOLD:.0%} (D40 P6); small = same sign, some pair below; neutral = mixed."
                f" noisy = > {BURST_CPU:.0f}% CPU outside the gate in > {BURST_SHARE:.0%} of the step's samples.{down}"
            ),
            "",
            "| Step | Case | Metric | Pairs | Verdict | Notes |",
            "|---|---|---|---|---|---|",
            *verdict_rows(out, results),
        ]
        (out / "verdict.md").write_text("\n".join(text) + "\n")
        print("\n".join(text))
        print(f"{time.strftime('%H:%M')} DONE: {out / 'verdict.md'}", flush=True)
    if any(r["status"].startswith("FAILED") for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
