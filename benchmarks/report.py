"""
Run reports: system info, preflight (is the machine quiet?), the round loop
and the JSON every workload writes (one format, `SCHEMA_VERSION`).
"""

import json
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import compat
import numpy as np
import scipy as sp

SCHEMA_VERSION = 1
HARNESS_DIR = Path(__file__).resolve().parent
PHASES = ("total", "prep", "kernel", "output")


def _cmd(*args):
    """stdout of a command, or None when it is missing or fails (best effort)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def system_info():
    """Versions, machine and threading of this process."""
    darwin = platform.system() == "Darwin"
    cpu = _cmd("sysctl", "-n", "machdep.cpu.brand_string") if darwin else None
    if cpu is None and Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        cpu = next((ln.split(":", 1)[1].strip() for ln in lines if "model name" in ln), None)
    try:
        memory_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        memory_gb = None
    rustc = _cmd("rustc", "--version")
    return {
        "python": platform.python_version(),
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "numpy": np.__version__,
        "scipy": sp.__version__,
        "os": f"macOS {platform.mac_ver()[0]}" if darwin else platform.platform(),
        "arch": platform.machine(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "rustc": rustc.splitlines()[0] if rustc else None,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


def load_1m():
    """1-minute load average, or None (no getloadavg: Windows)."""
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else None


def preflight():
    """Load, power source and the busiest processes before a timed run (best effort, None where unknown)."""
    on_ac = None
    if platform.system() == "Darwin":
        batt = _cmd("pmset", "-g", "batt")
        on_ac = None if batt is None else "AC Power" in batt.splitlines()[0]
    top = (
        _cmd("ps", "-Ao", "pcpu,comm", "-r")
        if platform.system() == "Darwin"
        else _cmd("ps", "-eo", "pcpu,comm", "--sort=-pcpu")
    )
    return {"load_1m": load_1m(), "on_ac": on_ac, "top": top.splitlines()[:6] if top else None}


def check_quiet(pre, max_load, strict):
    """Warn (or with strict, refuse) when the machine is loaded or on battery."""
    problems = []
    if pre["load_1m"] is not None and pre["load_1m"] > max_load:
        problems.append(f"1-min load {pre['load_1m']:.1f} > {max_load}")
    if pre["on_ac"] is False:
        problems.append("on battery")
    if problems:
        msg = "machine not quiet: " + ", ".join(problems)
        if strict:
            raise SystemExit(f"{msg} (--strict)")
        print(f"WARNING: {msg}; numbers are indicative only", file=sys.stderr)


def run_rounds(cases, rounds, warmup, verbose=True):
    """Warm-up + timed rounds, case order reversed every other round (ABBA): {name: [measurement per round]}."""
    results = {c.name: [] for c in cases}
    for r in range(warmup + rounds):
        label = "warmup" if r < warmup else f"round {r - warmup + 1}/{rounds}"
        for case in cases if r % 2 == 0 else cases[::-1]:
            m = case.run()
            if verbose:
                phases = "  ".join(f"{p} {m[p]:.3f}s" for p in PHASES if m.get(p) is not None)
                print(f"[{label}] {case.name:<24} {phases}")
            if r >= warmup:
                results[case.name].append(m)
    return results


def summarize(measurements):
    """Medians per phase (None if untimed), spread of the total, extras (median when numeric)."""
    totals = [m["total"] for m in measurements]
    median = {}
    for p in PHASES:
        values = [m[p] for m in measurements if m.get(p) is not None]
        median[p] = statistics.median(values) if values else None
    # Extras (out_nnz, percentiles, peak MB): median over rounds when numeric
    extras = {}
    for key in measurements[-1].keys() - set(PHASES):
        values = [m[key] for m in measurements]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        extras[key] = statistics.median(values) if numeric else values[-1]
    return {"median": median, "spread": (max(totals) - min(totals)) / median["total"], **extras}


def build_report(workload, config, dataset, cases, results, pre, info=None):
    """The JSON report of one run (one workload, one dataset); info = facts about the workload's setup."""
    return {
        "schema_version": SCHEMA_VERSION,
        "workload": workload,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "harness": {"git": compat.git_describe(HARNESS_DIR)},
        "build": compat.build_info(),
        "system": system_info(),
        "preflight": {**pre, "load_1m_after": load_1m()},
        "config": config,
        "dataset": dataset,
        "info": info or {},
        "cases": {
            c.name: {"params": _plain(c.params), **summarize(results[c.name]), "rounds": results[c.name]} for c in cases
        },
    }


def _plain(params):
    """params as they'll be written: a matrix param becomes its description, so the report holds no data
    (recommend's filter_cols=URM would keep the URM alive through the --rss children)."""
    return json.loads(json.dumps(params, default=_describe))


def write(report, output_dir, tag):
    """Save the report as {tag}_{workload}_{dataset key}_{timestamp}.json; returns the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = tag or report["build"]["git"] or report["build"]["similaripy_version"]
    path = output_dir / f"{tag}_{report['workload']}_{report['dataset']['key']}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=_describe))
    return path


def _describe(value):
    """JSON stand-in for a param that isn't a plain value (selector arrays and matrices)."""
    if hasattr(value, "nnz"):
        return f"<{type(value).__name__} {value.shape} nnz {value.nnz}>"
    if hasattr(value, "shape"):
        return f"<{type(value).__name__} {value.dtype} {value.shape}>"
    return str(value)


def peak_memory(func):
    """Run func() while sampling the process memory every 1 ms: (result, MB before, MB peak above it).

    On macOS this samples phys_footprint (Activity Monitor's "Memory"), not
    RSS: freed malloc pages stay in RSS, so after the dataset load new
    temporaries reuse resident pages and RSS barely moves (+1 MB measured for
    an ML-32M call). Elsewhere: psutil RSS. Spikes shorter than ~1 ms are
    missed; irrelevant for the GB-scale temporaries this is meant to catch.
    Call it on a fresh process's first call (`in_fresh_process`): a repeated
    call reuses the allocator's cached regions without them counting again
    (the second identical ML-32M cosine call stayed flat, the first rose ~490 MB).
    """
    if platform.system() == "Darwin":
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        buf = ctypes.create_string_buffer(256)
        pid = os.getpid()

        def current():
            # rusage_info_v0: uuid[16], then 7 uint64 before ri_phys_footprint
            if libc.proc_pid_rusage(pid, 0, buf) != 0:
                raise OSError("proc_pid_rusage failed")
            return int.from_bytes(buf.raw[72:80], "little")
    else:
        import psutil  # only needed for peak memory off macOS

        proc = psutil.Process()

        def current():
            return proc.memory_info().rss

    base = peak = current()
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.wait(0.001):
            peak = max(peak, current())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        out = func()
    finally:
        done.set()
        sampler.join()
    peak = max(peak, current())
    return out, base / 2**20, (peak - base) / 2**20


def in_fresh_process(func, *args):
    """func(*args) in a new spawned interpreter (no state, allocator caches or threads from this one)."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(func, *args).result()


def print_summary(report):
    """The table printed at the end of a run."""
    b, s, d, pre = report["build"], report["system"], report["dataset"], report["preflight"]
    load = " -> ".join("?" if x is None else f"{x:.2f}" for x in (pre["load_1m"], pre["load_1m_after"]))
    print(f"\n{report['workload']} on {d['key']} {tuple(d['shape'])} nnz {d['nnz']:,}")
    print(
        f"similaripy {b['similaripy_version']} ({b['git'] or b['path']}), python {s['python']}"
        f"{'' if s['gil_enabled'] else 't'}, {s['cpu']}, load {load}"
    )
    header = f"{'case':<24} " + " ".join(f"{p + ' (s)':>11}" for p in PHASES) + f" {'spread':>7} {'rounds':>6}"
    print(header + "\n" + "-" * len(header))
    for name, c in report["cases"].items():
        cells = " ".join("          -" if c["median"][p] is None else f"{c['median'][p]:11.4f}" for p in PHASES)
        extras = "  ".join(  # every numeric extra (output nnz, percentiles, peak MB, ...)
            f"{k} {v:.4g}"
            for k, v in c.items()
            if k != "spread" and isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        print(f"{name:<24} {cells} {c['spread']:7.1%} {len(c['rounds']):6d}  {extras}")
    if report["info"]:
        print("info: " + ", ".join(f"{k} {v}" for k, v in report["info"].items()))
