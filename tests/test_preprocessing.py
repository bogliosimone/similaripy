"""`from_interactions` against a plain-Python reference (a dict per pair)."""

import numpy as np
import pytest

import similaripy as sim
from similaripy.preprocessing import from_interactions, time_decay, to_histories


def _reference(users, items, values, timestamps, duplicates, user_ids, item_ids):
    """{(user, item): value} with the documented rules, events in input order."""
    pairs = {}
    for e, (u, i) in enumerate(zip(users, items)):
        if (user_ids is not None and u not in user_ids) or (item_ids is not None and i not in item_ids):
            continue
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
@pytest.mark.parametrize("vocab", [False, True])
def test_matches_reference(duplicates, with_t, vocab):
    rng = np.random.default_rng(0)
    n = 400
    users = rng.choice([3, 10, 11, 500, 7_000_000_000], n)  # gaps, an id past int32
    items = rng.choice([-5, 0, 2, 9, 40, 41], n)
    values = rng.choice([-1.0, 0.5, 1.0, 2.0], n)
    t = rng.integers(0, 20, n) if with_t else None  # many ties
    uv = np.array([500, 3, 99]) if vocab else None  # unsorted, 99 never seen
    iv = np.array([41, 2, -5, 9, 12]) if vocab else None
    URM, user_ids, item_ids = from_interactions(
        users, items, values, timestamps=t, duplicates=duplicates, user_ids=uv, item_ids=iv
    )
    if vocab:
        assert user_ids.tolist() == uv.tolist() and item_ids.tolist() == iv.tolist()
    else:
        assert user_ids.tolist() == sorted(set(users.tolist())) and item_ids.tolist() == sorted(set(items.tolist()))
    assert URM.dtype == np.float32 and URM.indices.dtype == np.int32 and URM.has_canonical_format
    ref = _reference(users, items, values, t, duplicates, uv, iv)
    got = _as_dict(URM, user_ids, item_ids)
    assert got.keys() == ref.keys()
    for pair, v in ref.items():
        assert got[pair] == pytest.approx(v, rel=1e-6)


def test_small_example():
    users = [7, 3, 7, 7, 3, 9]
    items = [100, 5, 100, 42, 100, 5]
    URM, user_ids, item_ids = from_interactions(users, items)
    assert user_ids.tolist() == [3, 7, 9] and item_ids.tolist() == [5, 42, 100]
    assert URM.toarray().tolist() == [[1, 0, 1], [0, 1, 2], [1, 0, 0]]
    # feeds ItemKNN in the original item ids
    rec = sim.serving.ItemKNN(sim.cosine(URM.T, k=5), item_ids=item_ids)
    assert rec.recommend([100], k=3)[0].tolist() == [42, 5]  # cosine 2/sqrt(5) > 1/sqrt(10)


def test_first_last_follow_time_then_input_order():
    users, items = [1, 1, 1], [2, 2, 2]
    values, t = [10.0, 20.0, 30.0], [5, 1, 5]
    last = from_interactions(users, items, values, timestamps=t, duplicates="last")
    first = from_interactions(users, items, values, timestamps=t, duplicates="first")
    assert last[0].data.tolist() == [30.0] and first[0].data.tolist() == [20.0]
    no_t = from_interactions(users, items, values, duplicates="first")
    assert no_t[0].data.tolist() == [10.0]


def test_zeros_dropped_from_both_matrices():
    # +1/-1 cancel; 1e-50 is 0 in float32
    URM, _, item_ids = from_interactions([1, 1, 1, 2], [5, 5, 6, 7], [1.0, -1.0, 1e-50, 2.0], timestamps=[0, 1, 2, 3])
    assert URM.nnz == 1 and item_ids.tolist() == [5, 6, 7]
    assert URM.toarray().tolist() == [[0, 0, 0], [0, 0, 2]]


def test_datetime64_any_unit_as_seconds():
    for unit in ("s", "ms", "us", "ns"):
        t = np.array(["2026-01-01T00:00:00", "2026-01-02T00:00:00"], dtype=f"datetime64[{unit}]")
        # the later event wins under "last", whatever the unit
        URM, _, _ = from_interactions([1, 1], [1, 1], [1.0, 2.0], timestamps=t[::-1], duplicates="last")
        assert URM.data.tolist() == [1.0]


def test_empty():
    URM, user_ids, _item_ids = from_interactions([], [], timestamps=[], duplicates="last")
    assert URM.shape == (0, 0) and user_ids.dtype == np.int64
    URM, _, _ = from_interactions([], [], item_ids=[4, 5])
    assert URM.shape == (0, 2)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"users": ["a"], "items": [1]}, TypeError, "users must be integer ids"),
        ({"users": [1.0], "items": [1]}, TypeError, "users must be integer ids"),
        ({"users": [1, 2], "items": [1]}, ValueError, "same length"),
        ({"users": [1], "items": [1], "values": [1.0, 2.0]}, ValueError, "one value per interaction"),
        ({"users": [1], "items": [1], "values": [np.nan]}, ValueError, "values must be finite"),
        ({"users": [1], "items": [1], "timestamps": ["x"]}, TypeError, "tz_convert"),
        ({"users": [1], "items": [1], "timestamps": np.array(["NaT"], dtype="datetime64[s]")}, ValueError, "finite"),
        ({"users": [1], "items": [1], "timestamps": [1, 2]}, ValueError, "one value per interaction"),
        ({"users": [1], "items": [1], "duplicates": "max"}, ValueError, "duplicates must be"),
        ({"users": [1], "items": [1], "item_ids": [1, 1]}, ValueError, "item_ids must be unique"),
        ({"users": [1], "items": [1], "item_ids": ["1"]}, TypeError, "item_ids must be integer ids"),
        ({"users": [1], "items": [1], "item_ids": [2]}, ValueError, "no interaction left"),
    ],
)
def test_rejects(kwargs, error, match):
    with pytest.raises(error, match=match):
        from_interactions(**kwargs)


def test_pair_cap(monkeypatch):
    monkeypatch.setattr(sim.preprocessing, "_MAX_PAIRS", 2)
    from_interactions([1, 1, 1, 1], [1, 1, 2, 2])  # 4 events, 2 pairs: fine
    with pytest.raises(ValueError, match="at most 2 distinct"):
        from_interactions([1, 1, 2], [1, 2, 1])


def test_to_histories_feeds_recommend_batch():
    rng = np.random.default_rng(1)
    users, items = rng.integers(0, 40, 800), rng.choice(np.arange(100, 160) * 7, 800)
    t = rng.integers(0, 99, 800)
    H, _, item_ids = from_interactions(users, items, rng.random(800) * time_decay(t, reference=99, scale=20))
    URM = H
    model = sim.serving.ItemKNN(sim.cosine(URM.T, k=20), item_ids=item_ids)
    histories, weights = to_histories(H, item_ids)
    assert len(histories) == URM.shape[0]
    ids, scores = model.recommend_batch(histories, weights=weights, k=10)
    for u in range(URM.shape[0]):
        row = H[[u]]
        one = model.recommend(item_ids[row.indices], weights=row.data, k=10)
        assert np.array_equal(ids[u], one[0]) and np.array_equal(scores[u], one[1])  # bit for bit
    # without item_ids: column indices, for ItemKNN(S) without ids
    histories, _ = to_histories(URM)
    assert histories[0].tolist() == URM[[0]].indices.tolist() and histories[0].dtype == np.int64


def test_to_histories_rejects():
    URM, _, _ = from_interactions([0, 1], [0, 1])
    with pytest.raises(TypeError, match="integer ids"):
        to_histories(URM, np.array(["a", "b"]))
    with pytest.raises(ValueError, match="one id per column"):
        to_histories(URM, [1])


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("spread", [1, 10**8])  # lookup table / sort path
def test_id_dtypes(dtype, spread):
    users = np.array([3, 1, 3], dtype=dtype) * dtype(spread)
    items = np.array([2, 2, 0], dtype=dtype)
    URM, user_ids, item_ids = from_interactions(users, items, values=np.array([1.0, 2.0, 3.0]))
    assert user_ids.dtype == item_ids.dtype == np.int64
    assert user_ids.tolist() == [spread, 3 * spread] and URM.toarray().tolist() == [[0, 2], [3, 1]]
    from_interactions(users, items, values=np.arange(6.0)[::2])  # strided values
