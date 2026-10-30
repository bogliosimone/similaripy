"""`time_decay` / `rank_decay` against a plain-Python reference, and the
event-level flow (offline weights into `from_interactions`, online into
`recommend`)."""

import datetime
import math

import numpy as np
import pytest

import similaripy as sim
from similaripy.preprocessing import from_interactions, rank_decay, time_decay

SHAPES = ["exponential", "power", "linear", "step"]


def _weight(age, shape, scale, p):
    x = max(age, 0.0) / scale
    if shape == "exponential":
        return 0.5**x
    if shape == "power":
        return (1 + (2 ** (1 / p) - 1) * x) ** -p
    if shape == "linear":
        return max(0.0, 1 - x)
    return 1.0 if x < 1 else 0.0


def _ranks(users, t):
    """0 = each user's latest (by t, ties: the later row; without t: input order)."""
    out = [0] * len(users)
    for u in set(users):
        rows = [e for e in range(len(users)) if users[e] == u]
        rows.sort(key=lambda e: (t[e] if t is not None else 0, e), reverse=True)
        for r, e in enumerate(rows):
            out[e] = r
    return out


def _events(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([4, 9, 7_000_000_000], n), rng.integers(0, 60, n).astype(float)  # ties in t


@pytest.mark.parametrize("shape", SHAPES)
def test_time_decay_matches_reference(shape):
    users, t = _events()
    p = 0.7 if shape == "power" else None
    for reference in (60.0, 30.0, "user"):  # 30: some events after it (age < 0 -> 0)
        kw = {"users": users} if reference == "user" else {}
        w = time_decay(t, reference=reference, shape=shape, scale=10.0, exponent=p, **kw)
        latest = {u: t[users == u].max() for u in set(users)}
        ref = [
            _weight((latest[u] if reference == "user" else reference) - x, shape, 10.0, p or 1.0)
            for u, x in zip(users, t)
        ]
        assert w.dtype == np.float32 and w.shape == t.shape
        np.testing.assert_allclose(w, np.float32(ref), rtol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_t", [False, True])
def test_rank_decay_matches_reference(shape, with_t):
    users, t = _events()
    p = 0.7 if shape == "power" else None
    w = rank_decay(users, timestamps=t if with_t else None, shape=shape, scale=5.0, exponent=p)
    ref = [_weight(r, shape, 5.0, p or 1.0) for r in _ranks(users.tolist(), t if with_t else None)]
    np.testing.assert_allclose(w, np.float32(ref), rtol=1e-6)


def test_rank_decay_one_history_and_table():
    t = np.array([5.0, 1.0, 9.0, 9.0])
    assert rank_decay(timestamps=t, shape="linear", scale=4).tolist() == pytest.approx([0.5, 0.25, 0.75, 1.0])
    table = rank_decay(np.zeros(10, dtype=int), shape="linear", scale=4)
    assert table[-3:].tolist() == pytest.approx([0.5, 0.75, 1.0])  # the slice for a 3-event history


@pytest.mark.parametrize("shape", SHAPES)
def test_inf_scale_is_identity(shape):
    users, t = _events()
    assert (time_decay(t, reference=60, shape=shape, scale=np.inf) == 1).all()
    assert (rank_decay(users, shape=shape, scale=np.inf) == 1).all()


def test_half_life_and_horizon():
    for p in (0.5, 1.0, 3.0):
        assert time_decay([0.0], reference=10, shape="power", scale=10, exponent=p)[0] == pytest.approx(0.5)
    assert time_decay([0.0], reference=10, scale=10)[0] == 0.5
    assert time_decay([0.0], reference=10, shape="linear", scale=10)[0] == 0.0  # exclusive horizon
    assert rank_decay(np.zeros(3, dtype=int), shape="step", scale=2).tolist() == [0.0, 1.0, 1.0]


def test_datetimes_and_durations():
    t = np.array(["2026-01-01", "2026-01-31"], dtype="datetime64[s]")
    for scale in (np.timedelta64(30, "D"), datetime.timedelta(days=30), 30 * 86400):
        assert time_decay(t, reference=np.datetime64("2026-01-31"), scale=scale).tolist() == [0.5, 1.0]
    assert time_decay(t, reference="user", users=[1, 1], scale=np.timedelta64(30, "D")).tolist() == [0.5, 1.0]


def test_datetime_like_references():
    t = np.array(["2026-01-01", "2026-01-31"], dtype="datetime64[s]")
    for ref in (datetime.datetime(2026, 1, 31), np.datetime64("2026-01-31")):
        assert time_decay(t, reference=ref, scale=np.timedelta64(30, "D")).tolist() == [0.5, 1.0]
    with pytest.raises(TypeError, match="both be datetimes or both be numbers"):
        time_decay(t, reference=30 * 86400, scale=1)
    with pytest.raises(TypeError, match="both be datetimes or both be numbers"):
        time_decay([1.0, 2.0], reference=np.datetime64("2026-01-31"), scale=1)


def test_pandas_timestamp_reference():
    pd = pytest.importorskip("pandas")  # not in the CI test group
    t = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-31"]))
    assert time_decay(t, reference=t.max(), scale=pd.Timedelta("30D")).tolist() == [0.5, 1.0]


def test_event_level_offline_equals_online():
    # repeats decay per event: offline (summed by from_interactions) == online (summed by recommend)
    rng = np.random.default_rng(0)
    users, items, ts = rng.integers(0, 20, 600), rng.integers(0, 40, 600), rng.integers(0, 100, 600)
    URM, user_ids, item_ids = from_interactions(users, items, time_decay(ts, reference=100, scale=25))
    model = sim.serving.ItemKNN(sim.cosine(URM.T, k=10), item_ids=item_ids)
    for u in user_ids[:5]:
        mine = users == u
        live = model.recommend(items[mine], weights=time_decay(ts[mine], reference=100, scale=25), k=5)
        row = URM[[int(np.searchsorted(user_ids, u))]]
        offline = model.recommend(item_ids[row.indices], weights=row.data, k=5)
        assert live[0].tolist() == offline[0].tolist()
        np.testing.assert_allclose(live[1], offline[1], rtol=1e-5)  # sums in another order


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda: time_decay([np.nan], reference=1, scale=1), ValueError, "finite"),
        (lambda: time_decay([1.0], reference="now", scale=1), ValueError, "reference must be"),
        (lambda: time_decay([1.0], reference="user", scale=1), ValueError, "needs users"),
        (lambda: time_decay([1.0], reference=1, users=[1], scale=1), ValueError, "reference='user' only"),
        (lambda: time_decay([1.0, 2.0], reference="user", users=[1], scale=1), ValueError, "one id per event"),
        (lambda: time_decay([1.0], reference=1, scale=1, exponent=2.0), ValueError, "shape='power' only"),
        (lambda: time_decay([1.0], reference=1, shape="power", scale=1, exponent=0), ValueError, "exponent must"),
        (lambda: time_decay([1.0], reference=1, shape="gauss", scale=1), ValueError, "shape must be one of"),
        (lambda: time_decay([1.0], reference=1, scale=0), ValueError, "scale must be > 0"),
        (lambda: time_decay([1.0], reference=1, scale=math.nan), ValueError, "scale must be > 0"),
        (lambda: rank_decay(scale=1), ValueError, "users and/or timestamps"),
        (lambda: rank_decay([1], scale=np.timedelta64(1, "D")), TypeError, "number of events"),
        (lambda: rank_decay([1, 2], timestamps=[1.0], scale=1), ValueError, "one id per event"),
        (lambda: rank_decay(["a"], scale=1), TypeError, "integer ids"),
    ],
)
def test_rejects(call, error, match):
    with pytest.raises(error, match=match):
        call()
