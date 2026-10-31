"""`interactions_to_urm` against a plain-Python reference (a dict per pair)."""

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim
from similaripy.preprocessing import interactions_to_histories, interactions_to_urm, time_decay, urm_to_histories


def _reference(users, items, values, timestamps, duplicates):
    """{(user, item): value} with the documented rules, events in input order."""
    pairs = {}
    for e, (u, i) in enumerate(zip(users, items)):
        v = 1.0 if values is None else float(values[e])
        t = e if timestamps is None else float(timestamps[e])  # input order without t
        pairs.setdefault((u, i), []).append((t, e, v))
    out = {}
    for pair, events in pairs.items():
        events.sort()  # by t, ties by input order
        if duplicates == "sum":
            v = sum(ev[2] for ev in events)
        else:
            v = events[0 if duplicates == "first" else -1][2]
        if np.float32(v) != 0:
            out[pair] = np.float32(v)
    return out


def _as_dict(URM, user_ids, item_ids):
    URM = URM.tocoo()
    d = {}
    for r, c, v in zip(URM.row, URM.col, URM.data):
        d[(int(user_ids[r]), int(item_ids[c]))] = v
    return d


@pytest.mark.parametrize("duplicates", ["sum", "first", "last"])
@pytest.mark.parametrize("with_t", [False, True])
def test_matches_reference(duplicates, with_t):
    rng = np.random.default_rng(0)
    n = 400
    users = rng.choice([3, 10, 11, 500, 7_000_000_000], n)  # gaps, an id past int32
    items = rng.choice([-5, 0, 2, 9, 40, 41], n)
    values = rng.choice([-1.0, 0.5, 1.0, 2.0], n)
    t = rng.integers(0, 20, n) if with_t else None  # many ties
    URM, user_ids, item_ids = interactions_to_urm(users, items, values, timestamps=t, duplicates=duplicates)
    assert user_ids.tolist() == sorted(set(users.tolist())) and item_ids.tolist() == sorted(set(items.tolist()))
    assert URM.dtype == np.float32 and URM.indices.dtype == np.int32 and URM.has_canonical_format
    ref = _reference(users, items, values, t, duplicates)
    got = _as_dict(URM, user_ids, item_ids)
    assert got.keys() == ref.keys()
    for pair, v in ref.items():
        assert got[pair] == pytest.approx(v, rel=1e-6)


def test_small_example():
    users = [7, 3, 7, 7, 3, 9]
    items = [100, 5, 100, 42, 100, 5]
    URM, user_ids, item_ids = interactions_to_urm(users, items)
    assert user_ids.tolist() == [3, 7, 9] and item_ids.tolist() == [5, 42, 100]
    assert URM.toarray().tolist() == [[1, 0, 1], [0, 1, 2], [1, 0, 0]]
    # feeds ItemKNN in the original item ids
    rec = sim.serving.ItemKNN(sim.cosine(URM.T, k=5), item_ids=item_ids)
    assert rec.recommend([100], k=3)[0].tolist() == [42, 5]  # cosine 2/sqrt(5) > 1/sqrt(10)


def test_first_last_follow_time_then_input_order():
    users, items = [1, 1, 1], [2, 2, 2]
    values, t = [10.0, 20.0, 30.0], [5, 1, 5]
    last = interactions_to_urm(users, items, values, timestamps=t, duplicates="last")
    first = interactions_to_urm(users, items, values, timestamps=t, duplicates="first")
    assert last[0].data.tolist() == [30.0] and first[0].data.tolist() == [20.0]
    no_t = interactions_to_urm(users, items, values, duplicates="first")
    assert no_t[0].data.tolist() == [10.0]


def test_zeros_dropped_from_both_matrices():
    # +1/-1 cancel; 1e-50 is 0 in float32
    URM, _, item_ids = interactions_to_urm([1, 1, 1, 2], [5, 5, 6, 7], [1.0, -1.0, 1e-50, 2.0], timestamps=[0, 1, 2, 3])
    assert URM.nnz == 1 and item_ids.tolist() == [5, 6, 7]
    assert URM.toarray().tolist() == [[0, 0, 0], [0, 0, 2]]


def test_datetime64_any_unit_as_seconds():
    for unit in ("s", "ms", "us", "ns"):
        t = np.array(["2026-01-01T00:00:00", "2026-01-02T00:00:00"], dtype=f"datetime64[{unit}]")
        # the later event wins under "last", whatever the unit
        URM, _, _ = interactions_to_urm([1, 1], [1, 1], [1.0, 2.0], timestamps=t[::-1], duplicates="last")
        assert URM.data.tolist() == [1.0]


def test_empty():
    URM, user_ids, _item_ids = interactions_to_urm([], [], timestamps=[], duplicates="last")
    assert URM.shape == (0, 0) and user_ids.dtype == np.int64


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"users": ["a"], "items": [1]}, TypeError, "users must be integer ids"),
        ({"users": [1.0], "items": [1]}, TypeError, "users must be integer ids"),
        ({"users": [1, 2], "items": [1]}, ValueError, "same length"),
        ({"users": [1], "items": [1], "values": [1.0, 2.0]}, ValueError, "one value per interaction"),
        ({"users": [1], "items": [1], "values": [np.nan]}, ValueError, "values must be finite"),
        ({"users": ["a"], "items": [1]}, TypeError, "pd.factorize"),
        ({"users": [1], "items": [1], "values": ["a"]}, TypeError, "values must be numbers"),
        ({"users": [1], "items": [1], "values": ["1.5"]}, TypeError, "values must be numbers"),
        ({"users": [1], "items": [1], "timestamps": ["2026-01-01"]}, TypeError, "pd.to_datetime"),
        ({"users": [1], "items": [1], "timestamps": ["x"]}, TypeError, "tz_convert"),
        ({"users": [1], "items": [1], "timestamps": np.array(["NaT"], dtype="datetime64[s]")}, ValueError, "finite"),
        ({"users": [1], "items": [1], "timestamps": [1, 2]}, ValueError, "one value per interaction"),
        ({"users": [1], "items": [1], "duplicates": "max"}, ValueError, "duplicates must be"),
    ],
)
def test_rejects(kwargs, error, match):
    with pytest.raises(error, match=match):
        interactions_to_urm(**kwargs)


def test_int64_indptr_past_int32_pairs(monkeypatch):
    # past 2^31 - 1 pairs the URM's indptr (and indices: scipy) are int64;
    # forced with a limit of 0. Same matrix, and it feeds the rest as usual.
    from similaripy import _s_plus

    rng = np.random.default_rng(3)
    users, items = rng.integers(0, 30, 500), rng.integers(0, 50, 500)
    ref, _, _ = interactions_to_urm(users, items)
    monkeypatch.setattr(_s_plus, "_MAX_INT32_NNZ", 0)
    URM, _, _ = interactions_to_urm(users, items)
    assert ref.indptr.dtype == np.int32 and URM.indptr.dtype == URM.indices.dtype == np.int64
    for a in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(URM, a), getattr(ref, a))
    assert all(np.array_equal(a, b) for a, b in zip(urm_to_histories(URM)[0], urm_to_histories(ref)[0]))


def test_to_histories_feeds_recommend_batch():
    rng = np.random.default_rng(1)
    users, items = rng.integers(0, 40, 800), rng.choice(np.arange(100, 160) * 7, 800)
    t = rng.integers(0, 99, 800)
    URM, _, item_ids = interactions_to_urm(users, items, rng.random(800) * time_decay(t, reference=99, scale=20))
    H = URM
    model = sim.serving.ItemKNN(sim.cosine(URM.T, k=20), item_ids=item_ids)
    histories, weights = urm_to_histories(H, item_ids)
    assert len(histories) == URM.shape[0]
    ids, scores = model.recommend_batch(histories, weights=weights, k=10)
    for u in range(URM.shape[0]):
        row = H[[u]]
        one = model.recommend(item_ids[row.indices], weights=row.data, k=10)
        assert np.array_equal(ids[u], one[0]) and np.array_equal(scores[u], one[1])  # bit for bit
    # without item_ids: column indices, for ItemKNN(S) without ids
    histories, _ = urm_to_histories(URM)
    assert histories[0].tolist() == URM[[0]].indices.tolist() and histories[0].dtype == np.int64


def test_urm_to_histories_drops_zeros():
    # a stored 0 (and 0 in float32) is no event: with exclude_seen it would hide the item
    X = sp.csr_array((np.array([1.0, 0.0, 1e-50]), np.array([0, 1, 2]), np.array([0, 3])), shape=(1, 3))
    histories, weights = urm_to_histories(X)
    assert histories[0].tolist() == [0] and weights[0].tolist() == [1.0]
    assert X.nnz == 3  # the caller's matrix untouched


def test_to_histories_rejects():
    URM, _, _ = interactions_to_urm([0, 1], [0, 1])
    with pytest.raises(TypeError, match="integer ids"):
        urm_to_histories(URM, np.array(["a", "b"]))
    with pytest.raises(ValueError, match="one id per column"):
        urm_to_histories(URM, [1])


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("spread", [1, 10**8])  # lookup table / sort path
def test_id_dtypes(dtype, spread):
    users = np.array([3, 1, 3], dtype=dtype) * dtype(spread)
    items = np.array([2, 2, 0], dtype=dtype)
    URM, user_ids, item_ids = interactions_to_urm(users, items, values=np.array([1.0, 2.0, 3.0]))
    assert user_ids.dtype == item_ids.dtype == np.int64
    assert user_ids.tolist() == [spread, 3 * spread] and URM.toarray().tolist() == [[0, 2], [3, 1]]
    interactions_to_urm(users, items, values=np.arange(6.0)[::2])  # strided values


def test_small_int_ids_wide_span():
    # int8 ids spanning more than int8's max: the table offset must not wrap
    users = np.array([-100, 100, -100, 0], dtype=np.int8)
    URM, user_ids, _ = interactions_to_urm(users, np.array([1, 2, 3, 4], dtype=np.int16))
    assert user_ids.tolist() == [-100, 0, 100] and URM.toarray().tolist() == [[1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]


def test_interactions_to_histories():
    users = np.array([7, 3, 7, 7, 3, 9])
    items = np.array([100, 5, 100, 42, 100, 5])
    w = np.array([1.0, 2.0, 0.0, 3.0, 4.0, 5.0])  # one weight-0 event: dropped
    user_ids, histories, weights = interactions_to_histories(users, items, w)
    assert user_ids.tolist() == [3, 7, 9]
    assert [h.tolist() for h in histories] == [[5, 100], [100, 42], [5]]  # input order, repeats kept
    assert [x.tolist() for x in weights] == [[2.0, 4.0], [1.0, 3.0], [5.0]]
    assert histories[0].dtype == np.int64 and weights[0].dtype == np.float32
    # same scores as the URM path (recommend sums repeats)
    rng = np.random.default_rng(2)
    users, items, w = rng.integers(0, 30, 900), rng.integers(0, 50, 900), rng.random(900).astype(np.float32)
    URM, _, item_ids = interactions_to_urm(users, items, w)
    model = sim.serving.ItemKNN(sim.cosine(URM.T, k=10).T, item_ids=item_ids)
    _, histories, weights = interactions_to_histories(users, items, w)
    a = model.recommend_batch(histories, weights=weights, k=5)
    hb, wb = urm_to_histories(URM, item_ids)
    b = model.recommend_batch(hb, weights=wb, k=5)
    for u in range(len(histories)):
        assert a[0][u].tolist() == b[0][u].tolist()
        np.testing.assert_allclose(a[1][u], b[1][u], rtol=1e-5)
    empty = interactions_to_histories([], [])
    assert empty[0].size == 0 and empty[1] == [] and empty[2] == []
    assert urm_to_histories(interactions_to_urm([], [])[0]) == ([], [])


def test_interactions_to_histories_rejects():
    with pytest.raises(ValueError, match="same length"):
        interactions_to_histories([1, 2], [1])
    with pytest.raises(ValueError, match="one value per event"):
        interactions_to_histories([1], [1], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        interactions_to_histories([1], [1], [np.nan])
    with pytest.raises(TypeError, match="integer ids"):
        interactions_to_histories([1], ["a"])
    with pytest.raises(TypeError, match="weights must be numbers"):
        interactions_to_histories([1], [1], ["1.0"])
