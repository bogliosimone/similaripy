"""
Benchmark workloads: each case is a name, the exact kwargs of its call, and
`run()`, one timed measurement.

Params recorded = params passed: every case builds its kwargs once, passes
them as they are and the report writes the same dict. An entry's fixed
values and the harness options never overlap silently: a harness option an
entry fixes is an error (the old harness passed shrink=0 to every call,
which overrode a functools.partial's shrink and timed plain cosine).

`make_cases(opts, URM)` builds every workload from the run's options (a plain
dict), the same way in the main process and in a fresh `--rss` child.
"""

import atexit
import os
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import compat
import loaders
import numpy as np
import report
import scipy.sparse as sp

import similaripy as sim

# Placeholder values for the tuned variants, to be replaced by real tuned ones
SHRINK, ALPHA, BETA = 10, 0.8, 0.6

# name -> (function, fixed kwargs)
SIMILARITIES = {
    "dot_product": (sim.dot_product, {}),
    "dot_product_binary": (sim.dot_product, {"binary": True}),  # integer scores: tie-heavy
    "cosine": (sim.cosine, {}),
    "asymmetric_cosine": (sim.asymmetric_cosine, {}),
    "jaccard": (sim.jaccard, {}),
    "dice": (sim.dice, {}),
    "tversky": (sim.tversky, {}),
    "p3alpha": (sim.p3alpha, {}),
    "rp3beta": (sim.rp3beta, {}),
    "splus": (sim.s_plus, {}),
    "cosine_bayesian": (sim.cosine, {"shrink": SHRINK, "shrink_type": "bayesian"}),
    "cosine_stabilized": (sim.cosine, {"shrink": SHRINK, "shrink_type": "stabilized"}),
    "dot_product_bayesian": (sim.dot_product, {"shrink": SHRINK, "shrink_type": "bayesian"}),
    "p3alpha_tuned": (sim.p3alpha, {"alpha": ALPHA}),
    "p3alpha_bayesian": (sim.p3alpha, {"alpha": ALPHA, "shrink": SHRINK, "shrink_type": "bayesian"}),
    "p3alpha_binary": (sim.p3alpha, {"alpha": ALPHA, "binary": True}),
    "rp3beta_tuned": (sim.rp3beta, {"alpha": ALPHA, "beta": BETA}),
    "rp3beta_binary": (sim.rp3beta, {"alpha": ALPHA, "beta": BETA, "binary": True}),
    "rp3beta_stabilized": (sim.rp3beta, {"alpha": ALPHA, "beta": BETA, "shrink": SHRINK, "shrink_type": "stabilized"}),
    "rp3beta_bayesian": (sim.rp3beta, {"alpha": ALPHA, "beta": BETA, "shrink": SHRINK, "shrink_type": "bayesian"}),
}

# Small preset: every case runs sim.<function>(X, k=100, **kwargs) on the synthetic matrix;
# the smoke set catches big (>20%) regressions only, the rest via --cases all
SMOKE_CASES = ["dot_product", "cosine", "cosine_k1000", "cosine_filter_matrix", "cosine_block2048"]
CONCURRENCY_CASES = ["cosine_2x2_serial", "cosine_2x2_concurrent"]

SERVE_MODES = ("latency", "block", "build")


@dataclass
class Case:
    """One benchmark case: `run()` returns {total, kernel, output, prep} seconds (phases None if untimed) + extras."""

    name: str
    params: dict
    run: Callable[[], dict] = field(repr=False)


def merge(name, fixed, options):
    """The kwargs of one call: the entry's fixed values + the harness options; an option the entry fixes raises."""
    clash = sorted(fixed.keys() & options.keys())
    if clash:
        raise ValueError(f"{name} fixes {clash}: drop the option or pick another entry")
    return {**fixed, **options}


def untimed(total, **extras):
    """A measurement with no phase times (no phase timer, or not a similarity call)."""
    return {"total": total, "kernel": None, "output": None, "prep": None, **extras}


def timed(func, *args, **kwargs):
    """Time one call: ({total, kernel, output, prep} seconds, output). Phases are None without a phase timer."""
    start = time.perf_counter()
    out = func(*args, **kwargs)
    total = time.perf_counter() - start
    phases = compat.phase_times()
    if phases is None:
        return untimed(total), out
    return {"total": total, **phases, "prep": total - phases["kernel"] - phases["output"]}, out


def make_cases(opts, URM):
    """(cases, info) of the workload opts["workload"]; info = facts about the setup, stored in the report."""
    workload = opts["workload"]
    if workload == "similarity":
        compat.set_column_order(opts["column_order"])
        cases, info = similarity_cases(URM, opts["similarities"], similarity_options(opts)), {}
    elif workload == "recommend":
        cases, info = recommend_cases(URM, opts)
    elif workload == "small":
        cases, info = small_cases(URM, opts), {}
    elif workload == "serve":
        cases, info = serve_cases(URM, opts)
    else:
        raise ValueError(f"unknown workload {workload!r}")
    names = [c.name for c in cases]
    if len(set(names)) < len(names):  # results are keyed by name: two cases would mix their rounds
        raise ValueError(f"duplicate cases: {sorted({n for n in names if names.count(n) > 1})}")
    return cases, info


def call_options(opts):
    """What every call of a run gets: threads, output format, no progress bar."""
    return {"num_threads": opts["threads"], "format_output": opts["format_output"], "verbose": False}


def similarity_options(opts):
    """The harness kwargs of a similarity run: k + call options, shrink/threshold only when set, block size."""
    options = {"k": opts["k"], **call_options(opts)}
    options |= {k: opts[k] for k in ("shrink", "threshold") if opts[k] is not None}
    return options | compat.block_size_kwargs(opts["block_size"])


def similarity_cases(URM, names, options):
    """Item-item similarity on URM.T (users x items in), one case per name; options = the harness kwargs set."""
    X = URM.T
    cases = []
    for name in names:
        if name not in SIMILARITIES:
            raise ValueError(f"unknown similarity {name!r}; available: {', '.join(SIMILARITIES)}")
        func, fixed = SIMILARITIES[name]
        kwargs = merge(name, fixed, options)

        def run(func=func, kwargs=kwargs):
            t, out = timed(func, X, **kwargs)
            # The output is dropped right away: one per round adds GBs on Yambda
            return {**t, "out_nnz": int(out.nnz)}

        cases.append(Case(name, kwargs, run))
    return cases


def recommend_cases(URM, opts):
    """The recommendation step as one library batch call: dot_product(URM, S) with and without
    filter_cols=URM ("exclude seen items"). S = cosine(URM.T, k=model_k), built once, untimed.
    With --rss the children load the parent's S: building it there would free its temporaries right
    before the measured call, which then reuses them uncounted (see report.peak_memory)."""
    if opts.get("model_path"):
        S = sp.load_npz(opts["model_path"])
    else:
        S = sim.cosine(URM.T, k=opts["model_k"], verbose=False, format_output=opts["format_output"])
        if opts.get("rss"):
            opts["model_path"] = _save_tmp(S)
    base = {"k": opts["k"], **call_options(opts)}
    cases = []
    for name, kwargs in (("recommend", base), ("recommend_filter_seen", {**base, "filter_cols": URM})):

        def run(kwargs=kwargs):
            t, out = timed(sim.dot_product, URM, S, **kwargs)
            return {**t, "out_nnz": int(out.nnz)}

        cases.append(Case(name, kwargs, run))
    return cases, {"model": "cosine", "model_k": opts["model_k"], "s_nnz": int(S.nnz)}


def _random_col_matrix(n_rows, n_cols, per_row, seed):
    """Sparse 0/1 matrix with per_row random columns per row (filter/target selector)."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n_rows), per_row)
    cols = rng.integers(0, n_cols, size=n_rows * per_row)
    m = sp.csr_array((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n_rows, n_cols))
    m.sum_duplicates()
    m.data[:] = 1
    return m


def small_definitions(n):
    """name -> (function, kwargs) of the small preset, for an n x n item matrix."""
    rng = np.random.default_rng(7)
    return {
        "dot_product": (sim.dot_product, {}),
        "cosine": (sim.cosine, {}),
        "rp3beta": (sim.rp3beta, {}),
        "cosine_k10": (sim.cosine, {"k": 10}),
        "cosine_k1000": (sim.cosine, {"k": 1000}),
        "cosine_binary": (sim.cosine, {"binary": True}),
        "cosine_noblock": (sim.cosine, {"block_size": None}),
        "cosine_block2048": (sim.cosine, {"block_size": 2048}),
        "cosine_filter_array": (sim.cosine, {"filter_cols": np.sort(rng.choice(n, n // 10, replace=False))}),
        "cosine_filter_matrix": (sim.cosine, {"filter_cols": _random_col_matrix(n, n, 50, seed=1)}),
        "cosine_target_matrix": (sim.cosine, {"target_cols": _random_col_matrix(n, n, 500, seed=2)}),
    }


def small_cases(URM, opts):
    """Millisecond calls on the synthetic matrix: fixed per-call costs (prep, binding, thread start-up,
    output build) that MovieLens/Yambda hide. One measurement = the median of opts["reps"] calls."""
    X = URM.T.tocsr()  # items x users, int32 indices (0.7.0 array selectors crash on int64)
    defs = small_definitions(X.shape[0])
    names = list(defs) + CONCURRENCY_CASES if opts["cases"] == ["all"] else opts["cases"]
    unknown = set(names) - defs.keys() - set(CONCURRENCY_CASES)
    if unknown:
        raise ValueError(f"unknown small cases {sorted(unknown)}; available: {', '.join([*defs, *CONCURRENCY_CASES])}")
    options, reps = call_options(opts), opts["reps"]
    cases = []
    for name in names:
        if name in CONCURRENCY_CASES:
            kwargs = {"k": 100, **options, "num_threads": 2}
            cases.append(Case(name, kwargs, _concurrency_run(X, kwargs, reps, name.endswith("concurrent"))))
            continue
        func, fixed = defs[name]
        kwargs = merge(name, {"k": 100, **fixed}, options)

        def run(func=func, kwargs=kwargs):
            times = []
            for _ in range(reps):
                t, out = timed(func, X, **kwargs)  # the previous output is freed here
                times.append(t)
            med = {p: _median([t[p] for t in times]) for p in report.PHASES}
            return {**med, "out_nnz": int(out.nnz)}

        cases.append(Case(name, kwargs, run))
    return cases


def _median(values):
    return None if values[0] is None else statistics.median(values)


def _concurrency_run(X, kwargs, reps, concurrent):
    """Two cosine calls (num_threads=2 each), on two threads or one after the other: median wall time.
    Run on a free-threaded build (3.14t) for a meaningful number; phases aren't per call under concurrency."""

    def run():
        times = []
        with ThreadPoolExecutor(max_workers=2) as pool:  # no threads start until the first map
            for _ in range(reps):
                start = time.perf_counter()
                if concurrent:
                    list(pool.map(lambda _: sim.cosine(X, **kwargs), range(2)))
                else:
                    for _ in range(2):
                        sim.cosine(X, **kwargs)
                times.append(time.perf_counter() - start)
        return untimed(statistics.median(times))

    return run


def serve_cases(URM, opts):
    """`similaripy.serving.ItemKNN` on S = cosine or rp3beta(URM.T, k=model_k), built once, untimed.
    Requests = URM rows (user histories), resampled with replacement in random order; a history cap N
    keeps N random items of the row. Every request excludes the history's own items."""
    if not compat.HAS_SERVING:
        raise ValueError("serve: this build has no similaripy.serving")
    from similaripy.serving import ItemKNN

    model = sim.cosine if opts["model"] == "cosine" else sim.rp3beta
    S = model(URM.T, k=opts["model_k"], verbose=False, format_output="csr")
    rec = ItemKNN(S)
    n_active = compat.serving_n_active(rec)
    info = {"model": opts["model"], "model_k": opts["model_k"], "s_nnz": int(S.nnz), "n_active": n_active}
    rng = np.random.default_rng(opts["seed"])
    users = rng.choice(URM.shape[0], size=opts["requests"])
    k, modes, cases = opts["k"], opts["mode"], []

    if "latency" in modes:
        evict = np.ones(opts["evict_mb"] << 20, dtype=np.uint8)
        for cap in opts["caps"]:
            hists = _histories(URM, users, None if cap == "none" else int(cap), rng)
            for regime in ("warm", "cold"):
                kwargs = {"k": k, "exclude_seen": True}
                run = _latency_run(rec, hists, kwargs, evict if regime == "cold" else None)
                cases.append(Case(f"latency_cap{cap}_{regime}", {**kwargs, "cap": cap, "regime": regime}, run))

    if "block" in modes:
        batch_in = compat.serving_batch_inputs(URM)
        threads = list(dict.fromkeys(opts["block_threads"] or [1, os.cpu_count()]))  # no duplicate case names
        info["block_threads"] = threads
        one = ItemKNN(S, block_size=None)
        info["block_control"] = n_active <= compat.DEFAULT_BLOCK_SIZE  # both models run one block
        for t in threads:
            cases.append(Case(f"block_default_t{t}", {"k": k, "num_threads": t}, _batch_run(rec, batch_in, k, t)))
            cases.append(Case(f"block_one_t{t}", {"k": k, "num_threads": t}, _batch_run(one, batch_in, k, t)))

    if "build" in modes:
        path = _save_tmp(S)
        info["s_mb"] = S.nnz * 8 / 2**20  # indices + data

        def build():
            secs, base_mb, peak_mb = report.in_fresh_process(build_child, path)
            return untimed(secs, rss_base_mb=base_mb, rss_peak_mb=peak_mb)

        cases.append(Case("build", {"block_size": "auto"}, build))
    return cases, info


def _save_tmp(S):
    """S saved as an uncompressed .npz in a temp dir removed at exit (S is GBs on Yambda 500M); its path."""
    tmp = tempfile.mkdtemp(prefix="similaripy-bench-")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    path = str(Path(tmp) / "S.npz")
    sp.save_npz(path, S, compressed=False)
    return path


def _histories(URM, users, cap, rng):
    rows = []
    for u in users:
        row = URM.indices[URM.indptr[u] : URM.indptr[u + 1]]
        if cap is not None and row.size > cap:
            row = rng.choice(row, size=cap, replace=False)
        rows.append(row.astype(np.int64))
    return rows


def _latency_run(rec, hists, kwargs, evict):
    """`recommend` per history, end to end: per-request µs percentiles; cold = evict caches before each call."""

    def run():
        ns = np.empty(len(hists), dtype=np.int64)
        for i, h in enumerate(hists):
            if evict is not None:
                evict[::64].sum()  # one read per cache line
            t = time.perf_counter_ns()
            rec.recommend(h, **kwargs)
            ns[i] = time.perf_counter_ns() - t
        us = ns / 1e3
        out = untimed(ns.sum() / 1e9)
        out |= {"p50_us": float(np.percentile(us, 50)), "p99_us": float(np.percentile(us, 99))}
        if us.size >= 100_000:  # p99.9 only with enough samples
            out["p99.9_us"] = float(np.percentile(us, 99.9))
        return out | {"mean_us": float(us.mean())}

    return run


def _batch_run(rec, batch_in, k, threads):
    """The core batch call on URM's rows (their prep outside the timing)."""

    def run():
        start = time.perf_counter()
        compat.serving_batch(rec, batch_in, k, threads)
        return untimed(time.perf_counter() - start)

    return run


def build_child(path):
    """In a fresh process: (seconds, MB before, MB peak above it) of ItemKNN(S) with S loaded from path.
    The seconds include the 1 ms memory sampler running alongside (one syscall per ms; the build releases the GIL)."""
    from similaripy.serving import ItemKNN

    S = sp.csr_array(sp.load_npz(path))
    start = time.perf_counter()
    _, base_mb, peak_mb = report.peak_memory(lambda: ItemKNN(S))
    return time.perf_counter() - start, base_mb, peak_mb


def rss_child(opts, name):
    """In a fresh process: (MB before, MB peak above it) of case `name`'s first call (see report.peak_memory)."""
    URM, _ = loaders.load(opts["dataset"], opts["version"], opts["event_type"], verbose=False)
    cases, _ = make_cases(opts, URM)
    case = next(c for c in cases if c.name == name)
    _, base_mb, peak_mb = report.peak_memory(case.run)
    return base_mb, peak_mb
