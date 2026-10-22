"""Serving engine: `Recommender` against a dense float64 reference."""

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim


def _S(n, signed, seed, density=0.3):
    """Small-integer S (exact sums in float32); the last two items have no
    column, item 0 has neither a row nor a column (dropped at build)."""
    rng = np.random.default_rng(seed)
    S = sp.random_array((n, n), density=density, rng=rng, format="csr", dtype=np.float64)
    S.data = np.round(S.data * 3 + 1)
    if signed:
        S.data *= rng.choice([-1, 1], size=S.nnz)
    keep = np.ones(n)
    keep[[0, n - 2, n - 1]] = 0
    S = sp.csr_array(S.multiply(keep))  # columns
    S = sp.csr_array(sp.diags_array(np.r_[0.0, np.ones(n - 1)]) @ S)  # row 0
    S.eliminate_zeros()
    return S


def _reference(S, ext, history, weights, k, exclude_seen, allowed_ids):
    """Top-k of score = h @ S (> 0), ties: more column entries, then lower column."""
    n = S.shape[0]
    D = S.toarray()
    counts = np.bincount(S.indices, minlength=n)
    kept = (np.diff(S.indptr) > 0) | (counts > 0)
    pos = {e: i for i, e in enumerate(ext) if kept[i]}
    h = np.zeros(n)
    for j, e in enumerate(history):
        if e in pos:
            h[pos[e]] += 1.0 if weights is None else weights[j]
    scores = h @ D
    keep = scores > 0
    if exclude_seen:
        keep[[pos[e] for e in history if e in pos]] = False
    if allowed_ids is not None:
        keep &= np.isin(ext, allowed_ids)
    cols = np.flatnonzero(keep)
    order = np.lexsort((cols, -counts[cols], -scores[cols]))[:k]
    return ext[cols[order]], scores[cols[order]]


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("n", [40, 150])  # 150: bitsets past one word
def test_matches_dense_reference(signed, external, n):
    S = _S(n, signed, seed=7, density=0.3 if n < 100 else 0.08)
    rng = np.random.default_rng(1)
    ext = np.arange(n)
    if external:  # random ids plus the int64 edge keys as real items
        ext = rng.choice(10**12, size=n, replace=False).astype(np.int64)
        ext[1:4] = [np.iinfo(np.int64).min, np.iinfo(np.int64).max, -1]
    rec = sim.Recommender(S, item_ids=ext if external else None)
    assert rec.n_items == n - 1  # item 0 dropped
    for trial in range(30):
        hist = rng.choice(ext, size=rng.integers(0, 8))
        hist = np.append(hist, hist[:1])  # a duplicate
        hist = np.append(hist, [ext[0], -5 if external else n + 3])  # dropped + unknown
        weights = rng.integers(1, 4, size=hist.size).astype(np.float64) if trial % 2 else None
        allowed_ids = rng.choice(ext, size=n // 2, replace=False) if trial % 3 == 0 else None
        allowed = None if allowed_ids is None else rec.mask(allowed_ids)
        k = int(rng.integers(0, n + 3))
        for exclude_seen in (True, False):
            got = rec.recommend(hist, weights, k=k, exclude_seen=exclude_seen, allowed=allowed)
            want = _reference(S, ext, hist, weights, k, exclude_seen, allowed_ids)
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_allclose(got[1], want[1], rtol=1e-6)
            if allowed_ids is not None:  # baked mask == per-call mask, bits
                baked = rec.with_allowed(allowed_ids).recommend(hist, weights, k=k, exclude_seen=exclude_seen)
                assert np.array_equal(baked[0], got[0])
                assert np.array_equal(baked[1].view(np.int32), got[1].view(np.int32))


def test_default_path_and_dtypes():
    rec = sim.Recommender(_S(30, False, seed=3))
    ids, scores = rec.recommend([5, 6, 7], k=5)
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    assert np.all(np.diff(scores) <= 0)
    assert not np.isin(ids, [5, 6, 7]).any()
    assert rec.recommend([], k=5)[0].size == 0
    assert rec.recommend([5], k=0)[0].size == 0
    assert 0 < rec.recommend([5, 6], k=2**62)[0].size <= rec.n_items  # huge k clamped


def test_unknown_ids():
    rec = sim.Recommender(_S(20, False, seed=2), item_ids=np.arange(100, 120))
    assert rec.recommend([105, 7], k=3)[0].size == rec.recommend([105], k=3)[0].size
    with pytest.raises(ValueError, match="unknown item id: 7"):
        rec.recommend([105, 7], unknown="raise")
    with pytest.raises(ValueError, match="unknown item id: 100"):  # dropped at build
        rec.recommend([100], unknown="raise")
    with pytest.raises(ValueError, match="'skip' or 'raise'"):
        rec.recommend([105], unknown="ignore")


def test_float_bits_independent_of_order_and_duplicates():
    rng = np.random.default_rng(0)
    rec = sim.Recommender(sp.random_array((200, 200), density=0.1, rng=rng, format="csr"))
    hist = rng.choice(200, size=30, replace=False)
    ids, scores = rec.recommend(hist, k=20)
    for other in (
        rec.recommend(hist[::-1], k=20),
        rec.recommend(np.concatenate([hist, hist[:5]]), np.r_[np.full(30, 1.0), np.zeros(5)], k=20),
    ):
        assert np.array_equal(other[0], ids)
        assert np.array_equal(other[1].view(np.int32), scores.view(np.int32))
    w = np.ones(30)
    w[0] = 2.0  # [a, a] == [a] with weight 2.0
    a, b = rec.recommend(np.r_[hist, hist[:1]], k=20), rec.recommend(hist, w, k=20)
    assert np.array_equal(a[1].view(np.int32), b[1].view(np.int32))


@pytest.mark.parametrize("block", [7, 0])
def test_batch_matches_single_bits(block):
    rng = np.random.default_rng(4)
    S = sp.random_array((300, 300), density=0.05, rng=rng, format="csr")
    ext = rng.choice(10**9, size=300, replace=False)
    rec = sim.Recommender(S, item_ids=ext, _block_size=block)
    one_block = sim.Recommender(S, item_ids=ext, _block_size=0)
    H = sp.random_array((50, 300), density=0.05, rng=rng, format="csr")
    mask = rec.mask(ext[:150])
    for allowed in (None, mask):
        R = rec.recommend_batch(H, k=10, allowed=allowed, num_threads=2)
        assert R.shape == H.shape and R.indptr.dtype == np.int32 and R.indices.dtype == np.int32
        for u in range(50):
            a, b = H.indptr[u], H.indptr[u + 1]
            ids, scores = rec.recommend(ext[H.indices[a:b]], H.data[a:b], k=10, allowed=allowed)
            if allowed is None:  # blocking never changes bits
                assert np.array_equal(one_block.recommend(ext[H.indices[a:b]], H.data[a:b], k=10)[1], scores)
            row = slice(R.indptr[u], R.indptr[u + 1])
            assert np.array_equal(ext[R.indices[row]], ids)  # best first, not sorted by column
            assert np.array_equal(R.data[row].view(np.int32), scores.view(np.int32))


def test_masks_bound_to_their_build():
    S = _S(10, False, seed=1)
    rec = sim.Recommender(S)
    with pytest.raises(ValueError, match="another model"):
        rec.recommend([5], allowed=sim.Recommender(_S(100, False, seed=1)).mask([5]))
    with pytest.raises(ValueError, match="another model"):  # same S, another build
        rec.recommend([5], allowed=sim.Recommender(S).mask([5]))
    baked = rec.with_allowed([3, 4, 5, 6])
    rec.recommend([5], allowed=baked.mask([3]))  # same build: fine both ways
    baked.recommend([5], allowed=rec.mask([3]))
    assert set(baked.recommend([5, 6, 7], k=10, exclude_seen=False)[0]) <= {3, 4, 5, 6}
    assert set(sim.Recommender(S, allowed=[3, 4]).recommend([5, 6, 7], k=10)[0]) <= {3, 4}


def test_pickle_round_trip_keeps_layout_and_mask():
    rng = np.random.default_rng(5)
    S = _S(80, True, seed=5)
    ext = rng.choice(10**9, size=80, replace=False)
    rec = sim.Recommender(S, item_ids=ext).with_allowed(ext[10:60])
    back = pickle.loads(pickle.dumps(rec))
    assert back.n_items == rec.n_items
    for _ in range(20):
        hist = rng.choice(ext, size=6)
        a, b = rec.recommend(hist, k=15), back.recommend(hist, k=15)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int32), b[1].view(np.int32))


def test_rejects_bad_input():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="square"):
        sim.Recommender(sp.random_array((3, 4), density=0.5, format="csr"))
    with pytest.raises(ValueError, match="unique"):
        sim.Recommender(S, item_ids=np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match="one id per item"):
        sim.Recommender(S, item_ids=np.arange(3))
    with pytest.raises(TypeError, match="integer ids"):
        sim.Recommender(S, item_ids=np.arange(10.0))
    rec = sim.Recommender(S)
    with pytest.raises(TypeError, match="integer ids"):
        rec.recommend([1.9])
    with pytest.raises(ValueError, match="finite"):
        rec.recommend([5, 6], weights=[1.0, np.nan])
    with pytest.raises(ValueError, match="len"):
        rec.recommend([5, 6], weights=[1.0])
    with pytest.raises(ValueError, match="k must"):
        rec.recommend([5], k=-1)
    with pytest.raises(ValueError, match="one column per item"):
        rec.recommend_batch(sp.random_array((2, 9), density=0.5, format="csr"))


def test_caller_matrices_unchanged():
    S = _S(20, True, seed=5)
    H = sp.random_array((5, 20), density=0.3, format="csr", rng=0)
    before = [(m.data.copy(), m.indices.copy(), m.indptr.copy()) for m in (S, H)]
    sim.Recommender(S).recommend_batch(H)
    for m, b in zip((S, H), before):
        assert all(np.array_equal(x, y) for x, y in zip(b, (m.data, m.indices, m.indptr)))


def test_concurrent_calls_identical():
    """One model shared by threads (scratch per OS thread, GIL released)."""
    rng = np.random.default_rng(2)
    rec = sim.Recommender(sp.random_array((500, 500), density=0.05, rng=rng, format="csr"))
    hists = [rng.choice(500, size=20) for _ in range(64)]
    want = [rec.recommend(h, k=10) for h in hists]
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda h: rec.recommend(h, k=10), hists * 4))
    for (gi, gs), (wi, ws) in zip(got, want * 4):
        assert np.array_equal(gi, wi) and np.array_equal(gs, ws)


def test_empty_models_and_masks():
    for S in (sp.csr_array((0, 0)), sp.csr_array((5, 5))):
        rec = sim.Recommender(S)
        assert rec.n_items == 0
        assert rec.recommend([0, 1], k=3)[0].size == 0
        assert rec.recommend_batch(sp.csr_array((2, S.shape[1]))).nnz == 0
        assert pickle.loads(pickle.dumps(rec)).n_items == 0
    rec = sim.Recommender(_S(20, False, seed=6), allowed=[])
    assert rec.recommend([5, 6, 7], k=5)[0].size == 0


def test_zero_as_a_real_external_id():
    S = _S(10, False, seed=8)
    ext = np.arange(10) * 7 - 14  # item 2 has id 0
    rec = sim.Recommender(S, item_ids=ext)
    D = S.toarray()
    hist = [ext[j] for j in range(10) if D[j].any()][:3]
    want = _reference(S, ext, hist, None, 10, True, None)
    assert np.array_equal(rec.recommend(hist, k=10)[0], want[0])
    assert np.array_equal(rec.recommend([0], k=10)[0], _reference(S, ext, [0], None, 10, True, None)[0])


def test_batch_skips_dropped_columns_and_int64_indptr(monkeypatch):
    S = _S(30, False, seed=9)  # item 0 is dropped
    rec = sim.Recommender(S)
    H = sp.csr_array(([1.0, 1.0, 1.0], ([0, 0, 1], [0, 5, 0])), shape=(2, 30))
    R = rec.recommend_batch(H, k=5)
    assert np.array_equal(R.indices[R.indptr[0] : R.indptr[1]], rec.recommend([5], k=5)[0])
    assert R.indptr[2] == R.indptr[1]  # a history of dropped items only
    import similaripy._serve as serve

    monkeypatch.setattr(serve, "_INT32_MAX", 0)
    assert rec.recommend_batch(H, k=5).indptr.dtype == np.int64


def test_reuse_after_raise():
    rec = sim.Recommender(_S(30, False, seed=10))
    want = rec.recommend([5, 6], k=5)
    with pytest.raises(ValueError, match="unknown"):
        rec.recommend([5, 6, 999], k=5, unknown="raise")
    got = rec.recommend([5, 6], k=5)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_boundary_types():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="fit in int64"):
        sim.Recommender(S, item_ids=np.array([2**63] + list(range(9)), dtype=np.uint64))
    rec = sim.Recommender(S)
    with pytest.raises(ValueError, match="fit in int64"):
        rec.recommend([2**63])
    with pytest.raises(ValueError, match="1-d"):
        rec.recommend([[5], [6]])
    with pytest.raises(TypeError, match="Recommender.mask"):
        rec.recommend([5], allowed=[3, 4])
    assert rec.recommend(np.array([5, 6], dtype=np.uint32), k=2**70)[0].size > 0
    bad = sp.csr_array(np.array([[0.0, np.inf], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        sim.Recommender(bad)
    with pytest.raises(ValueError, match="finite"):
        sim.Recommender(sp.csr_array(np.array([[0.0, 1e300], [1.0, 0.0]])))
