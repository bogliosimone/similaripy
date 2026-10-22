"""Serving engine (step 4.1): `Recommender` against a dense float64 reference."""

import numpy as np
import pytest
import scipy.sparse as sp

from similaripy import _core
from similaripy._serve import Recommender

VARIANTS = [
    {},
    {"block": 3},
    {"seen": "bitset", "mask": "before"},
    {"seen": "later", "mask": "later", "block": 2, "reuse_scratch": False},
    {"canonicalize": False},
]


def _S(n, signed, seed, density=0.3):
    rng = np.random.default_rng(seed)
    S = sp.random_array((n, n), density=density, rng=rng, format="csr", dtype=np.float64)
    S.data = np.round(S.data * 3 + 1)  # small integers: exact sums in float32
    if signed:
        S.data *= rng.choice([-1, 1], size=S.nnz)
    S = sp.csr_array(S.multiply(np.r_[np.ones(n - 2), 0, 0]))  # two items without a column: ids past n_active
    S.eliminate_zeros()
    return S


def _reference(S, ext, history, weights, k, exclude_seen, allowed_ids):
    """Top-k of score = h @ S (> 0), ties: more column entries, then lower column."""
    n = S.shape[0]
    pos = {e: i for i, e in enumerate(ext)}
    h = np.zeros(n)
    for j, e in enumerate(history):
        if e in pos:
            h[pos[e]] += 1.0 if weights is None else weights[j]
    scores = h @ S.toarray()
    counts = np.bincount(S.indices, minlength=n)
    keep = scores > 0
    if exclude_seen:
        keep[[pos[e] for e in history if e in pos]] = False
    if allowed_ids is not None:
        keep &= np.isin(ext, allowed_ids)
    cols = np.flatnonzero(keep)
    order = np.lexsort((cols, -counts[cols], -scores[cols]))[:k]
    return ext[cols[order]], scores[cols[order]]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_matches_dense_reference(variant, signed, external):
    n = 40
    S = _S(n, signed, seed=7)
    rng = np.random.default_rng(1)
    ext = rng.choice(10**12, size=n, replace=False).astype(np.int64) if external else np.arange(n)
    rec = Recommender(S, item_ids=ext if external else None)
    lookup = "eytzinger" if external else "dense"
    v = _core.ServeVariant(lookup=lookup, **variant)
    for trial in range(30):
        hist = rng.choice(ext, size=rng.integers(0, 8))
        hist = np.append(hist, hist[:1])  # a duplicate
        hist = np.append(hist, -5)  # an unknown id
        weights = rng.integers(1, 4, size=hist.size).astype(np.float64) if trial % 2 else None
        allowed_ids = rng.choice(ext, size=n // 2, replace=False) if trial % 3 == 0 else None
        allowed = None if allowed_ids is None else rec.mask(allowed_ids)
        k = int(rng.integers(0, n + 3))
        for exclude_seen in (True, False):
            got = rec.recommend(hist, weights, k=k, exclude_seen=exclude_seen, allowed=allowed, _variant=v)
            want = _reference(S, ext, hist, weights, k, exclude_seen, allowed_ids)
            if variant.get("mask") == "later" and allowed is not None:
                # over-fetch then filter: a prefix of the full answer
                assert np.array_equal(got[0], want[0][: got[0].size])
            elif variant.get("seen") == "later" and exclude_seen:
                assert np.array_equal(got[0], want[0][: got[0].size])
            else:
                np.testing.assert_array_equal(got[0], want[0])
                np.testing.assert_allclose(got[1], want[1], rtol=1e-6)


def test_default_path_and_dtypes():
    S = _S(30, False, seed=3)
    rec = Recommender(S)
    ids, scores = rec.recommend([0, 1, 2], k=5)
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    assert np.all(np.diff(scores) <= 0)
    assert not np.isin(ids, [0, 1, 2]).any()
    assert rec.recommend([], k=5)[0].size == 0
    assert rec.recommend([0], k=0)[0].size == 0


def test_float_bits_independent_of_order_duplicates_and_blocking():
    rng = np.random.default_rng(0)
    S = sp.random_array((200, 200), density=0.1, rng=rng, format="csr")
    rec = Recommender(S)
    hist = rng.choice(200, size=30, replace=False)
    ids, scores = rec.recommend(hist, k=20)
    for other in (
        rec.recommend(hist[::-1], k=20),
        rec.recommend(np.concatenate([hist, hist[:5]]), np.r_[np.full(30, 1.0), np.zeros(5)], k=20),
        rec.recommend(hist, k=20, _variant=_core.ServeVariant(block=7)),
    ):
        assert np.array_equal(other[0], ids)
        assert np.array_equal(other[1].view(np.int32), scores.view(np.int32))
    # [a, a] == [a] with weight 2.0
    two = rec.recommend(np.r_[hist, hist[:1]], k=20)
    w = np.ones(30)
    w[0] = 2.0
    assert np.array_equal(two[1].view(np.int32), rec.recommend(hist, w, k=20)[1].view(np.int32))


def test_rejects_bad_input():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="square"):
        Recommender(sp.random_array((3, 4), density=0.5, format="csr"))
    with pytest.raises(ValueError, match="unique"):
        Recommender(S, item_ids=np.zeros(10))
    with pytest.raises(ValueError, match="one id per item"):
        Recommender(S, item_ids=np.arange(3))
    rec = Recommender(S)
    with pytest.raises(ValueError, match="finite"):
        rec.recommend([0, 1], weights=[1.0, np.nan])
    with pytest.raises(ValueError, match="len"):
        rec.recommend([0, 1], weights=[1.0])
    with pytest.raises(ValueError, match="another model"):
        rec.recommend([0], allowed=Recommender(_S(100, False, seed=1)).mask([0]))
    with pytest.raises(ValueError, match="k must"):
        rec.recommend([0], k=-1)
    with pytest.raises(ValueError, match="unknown lookup"):
        _core.ServeVariant(lookup="nope")


def test_caller_matrix_unchanged():
    S = _S(20, True, seed=5)
    before = (S.data.copy(), S.indices.copy(), S.indptr.copy())
    Recommender(S)
    assert all(np.array_equal(x, y) for x, y in zip(before, (S.data, S.indices, S.indptr)))


def test_concurrent_calls_identical():
    """One model shared by threads (scratch per OS thread, GIL released)."""
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(2)
    rec = Recommender(sp.random_array((500, 500), density=0.05, rng=rng, format="csr"))
    hists = [rng.choice(500, size=20) for _ in range(64)]
    want = [rec.recommend(h, k=10) for h in hists]
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda h: rec.recommend(h, k=10), hists * 4))
    for (gi, gs), (wi, ws) in zip(got, want * 4):
        assert np.array_equal(gi, wi) and np.array_equal(gs, ws)
