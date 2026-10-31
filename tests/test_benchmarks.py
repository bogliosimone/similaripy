"""Smoke test of the benchmark harness (benchmarks/): every workload on the synthetic dataset, one round.

Needs numpy/scipy only (the CI test group). Skipped where benchmarks/ isn't
shipped (sdist, wheel tests). Only benchmarks/ goes on sys.path, never the
repo root: that would import the source tree's similaripy (no _core).
"""

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
pytestmark = pytest.mark.skipif(not BENCH.is_dir(), reason="benchmarks/ not shipped here")


@pytest.fixture
def run(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(BENCH))
    monkeypatch.chdir(tmp_path)  # spawned children must not import the source tree (no _core)
    import run as bench_run

    def call(*args):
        path = bench_run.main(
            [*args, "--dataset", "synthetic", "--warmup", "0", "--quiet", "--output-dir", str(tmp_path)]
        )
        return json.loads(Path(path).read_text())

    return call


def test_similarity_report(run):
    rep = run("similarity", "--similarities", "cosine", "rp3beta_bayesian", "--rounds", "2")
    assert rep["schema_version"] == 1
    assert rep["dataset"]["key"] == "synthetic-10k"
    assert rep["build"]["has_phase_timer"]
    base = {"k": 100, "num_threads": 0, "format_output": "csr", "verbose": False}
    assert rep["cases"]["cosine"]["params"] == base
    assert rep["cases"]["rp3beta_bayesian"]["params"] == {
        **base,
        "alpha": 0.8,
        "beta": 0.6,
        "shrink": 10,
        "shrink_type": "bayesian",
    }
    for case in rep["cases"].values():
        assert len(case["rounds"]) == 2
        assert case["out_nnz"] > 0
        m = case["median"]
        assert m["total"] > 0 and m["kernel"] > 0
        assert m["prep"] == pytest.approx(m["total"] - m["kernel"] - m["output"], abs=1e-3)


def test_params_recorded_are_params_passed(run, monkeypatch):
    import workloads

    import similaripy as sim

    seen = []

    def spy(X, **kwargs):
        seen.append(kwargs)
        return sim.cosine(X, **kwargs)

    monkeypatch.setitem(workloads.SIMILARITIES, "spy", (spy, {"shrink": 10, "shrink_type": "bayesian"}))
    rep = run("similarity", "--similarities", "spy", "--threshold", "0.01", "--block-size", "none")
    assert seen and all(kw == rep["cases"]["spy"]["params"] for kw in seen)
    assert seen[0]["shrink"] == 10 and seen[0]["threshold"] == 0.01 and seen[0]["block_size"] is None


def test_option_an_entry_fixes_is_an_error(run):
    # The old harness passed shrink= to every call and silently overrode the entry's
    with pytest.raises(SystemExit, match="fixes"):
        run("similarity", "--similarities", "cosine_bayesian", "--shrink", "5")
    rep = run("similarity", "--similarities", "cosine", "--shrink", "5")
    assert rep["cases"]["cosine"]["params"]["shrink"] == 5


def _needs_peak_memory():
    if sys.platform != "darwin":
        pytest.importorskip("psutil")  # peak memory off macOS (the bench group, not the test group)


def test_similarity_rss(run):
    _needs_peak_memory()
    rep = run("similarity", "--similarities", "cosine", "--rss")
    assert rep["cases"]["cosine"]["rss_peak_mb"] >= 0 and rep["cases"]["cosine"]["rss_base_mb"] > 0


def test_recommend(run):
    rep = run("recommend", "--model-k", "20", "--k", "10")
    assert set(rep["cases"]) == {"recommend", "recommend_filter_seen"}
    assert rep["cases"]["recommend_filter_seen"]["params"]["filter_cols"].startswith("<csr_array")
    assert rep["info"]["s_nnz"] > 0


def test_small_every_case(run):
    rep = run("small", "--cases", "all", "--reps", "1", "--rounds", "1")
    assert {"cosine_k10", "cosine_filter_array", "cosine_2x2_concurrent"} <= set(rep["cases"])
    assert rep["cases"]["cosine_k10"]["params"]["k"] == 10
    assert rep["cases"]["cosine_noblock"]["params"]["block_size"] is None


def test_serve_modes(run):
    rep = run(
        "serve", "--mode", "latency", "gate", "block", "--requests", "200", "--caps", "50", "none",
        "--gate-threads", "1", "2", "2", "--model-k", "20", "--k", "10", "--rounds", "1",
    )  # fmt: skip
    cases = rep["cases"]
    assert {"latency_cap50_warm", "latency_capnone_cold", "gate_kernel_t2", "gate_batch_t1", "block_one_t1"} <= set(
        cases
    )
    assert rep["info"]["gate_threads"] == [1, 2]  # duplicates dropped
    assert cases["latency_cap50_warm"]["p99_us"] >= cases["latency_cap50_warm"]["p50_us"] > 0
    assert rep["info"]["gate_overlap"] > 0.99  # same weights, same model: only near-ties may differ


def test_serve_build(run):
    _needs_peak_memory()
    rep = run("serve", "--mode", "build", "--model-k", "20", "--rounds", "1")
    assert rep["cases"]["build"]["rss_peak_mb"] >= 0 and rep["cases"]["build"]["median"]["total"] > 0


def test_workload_defaults(monkeypatch):
    # set_defaults on one subparser once changed the shared parent's defaults: recommend ran on synthetic
    monkeypatch.syspath_prepend(str(BENCH))
    import run as bench_run

    p = bench_run.parser()
    assert p.parse_args(["small"]).dataset is None and p.parse_args(["recommend"]).dataset is None


def test_abba_and_compare(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(BENCH))
    monkeypatch.chdir(tmp_path)
    import abba
    import compare

    small = ["small", "--cases", "dot_product", "--reps", "1", "--rounds", "1", "--warmup", "0"]
    abba.main(
        ["--base", sys.executable, "--cand", sys.executable, "--tag", "t", "--output-dir", str(tmp_path), "--", *small]
    )
    out = json.loads((tmp_path / "abba_t.json").read_text())
    assert len(out["pairs"]["total"]["dot_product"]) == 2
    runs = sorted(tmp_path.glob("t-*.json"))
    notes = [json.loads(f.read_text())["config"]["note"] for f in runs]
    assert sorted(notes) == ["abba t base pair 1", "abba t base pair 2", "abba t cand pair 1", "abba t cand pair 2"]
    capsys.readouterr()
    compare.main([str(runs[0]), str(runs[1]), "--metric", "kernel"])
    assert "dot_product" in capsys.readouterr().out
    # The same tag again would mix the earlier reports into the pairs: refused before running
    with pytest.raises(SystemExit):
        abba.main(
            [
                "--base",
                sys.executable,
                "--cand",
                sys.executable,
                "--tag",
                "t",
                "--output-dir",
                str(tmp_path),
                "--",
                *small,
            ]
        )
