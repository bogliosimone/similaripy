"""`time_decay` / `rank_decay` against a per-entry Python reference."""

import datetime
import math

import numpy as np
import pytest
import scipy.sparse as sps

import similaripy as sim
from similaripy.preprocessing import from_interactions


def _weight(age, shape, scale, p):
    x = age / scale
    if shape == "exponential":
        return 0.5**x
    if shape == "power":
        return (1 + (2 ** (1 / p) - 1) * x) ** -p
    if shape == "linear":
        return max(0.0, 1 - x)
    return 1.0 if x < 1 else 0.0


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    users, items = rng.integers(0, 12, n), rng.integers(0, 30, n)
    values, t = rng.choice([0.5, 1.0, 3.0], n), rng.integers(0, 50, n)  # ties
    URM, _, _, T = from_interactions(users, items, values, timestamps=t)
    return URM, T


def _reference(URM, T, by, shape, scale, p, reference, min_weight):
    URM, T = URM.toarray(), T.toarray()
    present = URM != 0
    out = np.zeros_like(URM, dtype=np.float64)
    for u in range(URM.shape[0]):
        cols = np.flatnonzero(present[u])
        if by == "rank":
            ranked = sorted(cols, key=lambda c: (-T[u, c], c))
            age = {c: r for r, c in enumerate(ranked)}
        else:
            ref = {"user": T[u, cols].max() if cols.size else 0, "global": T[present].max()}.get(reference, reference)
            age = {c: ref - T[u, c] for c in cols}
        for c in cols:
            w = _weight(age[c], shape, scale, p)
            if w > 0 and w >= min_weight:
                out[u, c] = URM[u, c] * w
    return out


@pytest.mark.parametrize("shape", ["exponential", "power", "linear", "step"])
@pytest.mark.parametrize("by", ["time", "rank"])
@pytest.mark.parametrize("min_weight", [0.0, 0.3])
def test_matches_reference(shape, by, min_weight):
    URM, T = _data()
    scale = 10.0 if by == "time" else 4.0
    p = 0.7 if shape == "power" else None
    for reference in ["user", "global", 60.0] if by == "time" else [None]:
        kw = {"shape": shape, "scale": scale, "exponent": p, "min_weight": min_weight}
        if by == "time":
            got = sim.time_decay(URM, T, reference=reference, **kw)
        else:
            got = sim.rank_decay(URM, T, **kw)
        ref = _reference(URM, T, by, shape, scale, p or 1.0, reference, min_weight)
        assert got.dtype == np.float32 and got.has_canonical_format
        assert np.array_equal(got.toarray() != 0, ref != 0)
        np.testing.assert_allclose(got.toarray(), ref, rtol=1e-6)


@pytest.mark.parametrize("shape", ["exponential", "power", "linear", "step"])
def test_inf_scale_is_identity(shape):
    URM, T = _data()
    for out in (sim.time_decay(URM, T, shape=shape, scale=np.inf), sim.rank_decay(URM, T, shape=shape, scale=np.inf)):
        assert np.array_equal(out.indptr, URM.indptr) and np.array_equal(out.indices, URM.indices)
        assert np.array_equal(out.data, URM.data)


def test_half_life_at_scale():
    URM, _, _, T = from_interactions([0, 0], [0, 1], timestamps=[10, 0])
    for p in (0.5, 1.0, 3.0):
        assert sim.time_decay(URM, T, shape="power", scale=10, exponent=p).data.tolist() == pytest.approx([1.0, 0.5])
    assert sim.time_decay(URM, T, scale=10).data.tolist() == pytest.approx([1.0, 0.5])
    assert sim.time_decay(URM, T, shape="linear", scale=10).nnz == 1  # 0 at the horizon: dropped


def test_chains_and_runs_after_bm25():
    URM, T = _data()
    a = sim.rank_decay(sim.time_decay(URM, T, scale=10, min_weight=0.3), T, shape="step", scale=3)
    b = sim.time_decay(sim.rank_decay(URM, T, shape="step", scale=3), T, scale=10, min_weight=0.3)
    np.testing.assert_allclose(a.toarray(), b.toarray(), rtol=1e-6)  # rank over T's full row: order-free
    assert a.nnz < URM.nnz
    sim.time_decay(sim.bm25(URM), T, scale=10)  # bm25's pattern is URM's (or a subset)


def test_durations_and_epoch_zero():
    t = np.array(["1970-01-01", "1970-01-31"], dtype="datetime64[s]")  # the first one is T value 0.0
    URM, _, _, T = from_interactions([0, 0], [0, 1], timestamps=t)
    assert T.nnz == 2
    for scale in (np.timedelta64(30, "D"), datetime.timedelta(days=30), 30 * 86400):
        assert sim.time_decay(URM, T, scale=scale).data.tolist() == pytest.approx([0.5, 1.0])


def test_rank_ties_by_column():
    URM, _, _, T = from_interactions([0, 0, 0], [5, 3, 4], timestamps=[1, 1, 0])
    out = sim.rank_decay(URM, T, shape="linear", scale=3)  # ranks: col 3 -> 0, col 5 -> 1, col 4 -> 2
    assert out.toarray()[0].tolist() == pytest.approx([1.0, 1 / 3, 2 / 3])


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda U, T: sim.time_decay(U, T, scale=1, exponent=2.0), ValueError, "shape='power' only"),
        (lambda U, T: sim.time_decay(U, T, shape="power", scale=1, exponent=0), ValueError, "exponent must be > 0"),
        (lambda U, T: sim.time_decay(U, T, shape="gauss", scale=1), ValueError, "shape must be one of"),
        (lambda U, T: sim.time_decay(U, T, scale=0), ValueError, "scale must be > 0"),
        (lambda U, T: sim.time_decay(U, T, scale=math.nan), ValueError, "scale must be > 0"),
        (lambda U, T: sim.time_decay(U, T, scale=1, reference=0.5), ValueError, "after reference"),
        (lambda U, T: sim.time_decay(U, T, scale=1, reference="now"), ValueError, "reference must be"),
        (lambda U, T: sim.rank_decay(U, T, scale=np.timedelta64(1, "D")), TypeError, "number of items"),
        (lambda U, T: sim.time_decay(U, T[:, :1], scale=1), ValueError, "T must have X's shape"),
        (lambda U, T: sim.time_decay(U, T.toarray(), scale=1), TypeError, "T must be a sparse matrix"),
        (lambda U, T: sim.time_decay(U + sps.csr_array(np.ones(U.shape)), T, scale=1), ValueError, "entries T doesn't"),
    ],
)
def test_rejects(call, error, match):
    URM, _, _, T = from_interactions([0, 0, 1], [0, 1, 1], timestamps=[1, 2, 3])
    with pytest.raises(error, match=match):
        call(URM, T)
