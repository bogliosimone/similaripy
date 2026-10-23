"""Serving engine: `ItemKNN` against a dense float64 reference."""

import json
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
    rec = sim.ItemKNN(S, item_ids=ext if external else None)
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
            if allowed_ids is not None:  # trimmed S == view's default mask == per-call mask, bits
                for other in (
                    sim.ItemKNN(S, item_ids=ext if external else None, candidate_items=allowed_ids),
                    rec.view(allowed_items=allowed_ids),
                ):
                    r = other.recommend(hist, weights, k=k, exclude_seen=exclude_seen)
                    assert np.array_equal(r[0], got[0])
                    assert np.array_equal(r[1].view(np.int32), got[1].view(np.int32))


def test_default_path_and_dtypes():
    rec = sim.ItemKNN(_S(30, False, seed=3))
    ids, scores = rec.recommend([5, 6, 7], k=5)
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    assert np.all(np.diff(scores) <= 0)
    assert not np.isin(ids, [5, 6, 7]).any()
    assert rec.recommend([], k=5)[0].size == 0
    assert rec.recommend([5], k=0)[0].size == 0
    assert 0 < rec.recommend([5, 6], k=2**62)[0].size <= rec.n_items  # huge k clamped


def test_unknown_ids():
    rec = sim.ItemKNN(_S(20, False, seed=2), item_ids=np.arange(100, 120))
    assert rec.recommend([105, 7], k=3)[0].size == rec.recommend([105], k=3)[0].size
    with pytest.raises(ValueError, match="unknown item id: 7"):
        rec.recommend([105, 7], unknown="raise")
    with pytest.raises(ValueError, match="unknown item id: 100"):  # dropped at build
        rec.recommend([100], unknown="raise")
    with pytest.raises(ValueError, match="'skip' or 'raise'"):
        rec.recommend([105], unknown="ignore")


def test_float_bits_independent_of_order_and_duplicates():
    rng = np.random.default_rng(0)
    rec = sim.ItemKNN(sp.random_array((200, 200), density=0.1, rng=rng, format="csr"))
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


@pytest.mark.parametrize("block", [7, None])
def test_batch_matches_single_bits(block):
    rng = np.random.default_rng(4)
    S = sp.random_array((300, 300), density=0.05, rng=rng, format="csr")
    ext = rng.choice(10**9, size=300, replace=False)
    rec = sim.ItemKNN(S, item_ids=ext, block_size=block)
    one_block = sim.ItemKNN(S, item_ids=ext, block_size=None)
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
    rec = sim.ItemKNN(S)
    with pytest.raises(ValueError, match="another model"):
        rec.recommend([5], allowed=sim.ItemKNN(_S(100, False, seed=1)).mask([5]))
    with pytest.raises(ValueError, match="another model"):  # same S, another build
        rec.recommend([5], allowed=sim.ItemKNN(S).mask([5]))
    view = rec.view(allowed_items=[3, 4, 5, 6])
    rec.recommend([5], allowed=view.mask([3]))  # same build: fine both ways
    view.recommend([5], allowed=rec.mask([3]))
    assert set(view.recommend([5, 6, 7], k=10, exclude_seen=False)[0]) <= {3, 4, 5, 6}
    assert set(sim.ItemKNN(S, candidate_items=[3, 4]).recommend([5, 6, 7], k=10)[0]) <= {3, 4}


def test_view_shares_s_and_combines_masks():
    S = _S(60, False, seed=12)
    rec = sim.ItemKNN(S)
    hist = [5, 6, 7, 8]
    full = rec.recommend(hist, k=60, exclude_seen=False)[0]
    a, b = list(range(0, 60, 2)), list(range(0, 60, 3))
    va = rec.view(allowed_items=a)
    # default AND per call
    got = va.recommend(hist, k=60, exclude_seen=False, allowed=va.mask(b))[0]
    assert list(got) == [i for i in full if i % 6 == 0]
    # a view of a view replaces the default; None widens back to all
    assert list(va.view(allowed_items=b).recommend(hist, k=60, exclude_seen=False)[0]) == [
        i for i in full if i % 3 == 0
    ]
    assert np.array_equal(va.view(allowed_items=None).recommend(hist, k=60, exclude_seen=False)[0], full)
    # candidate_items trims for good: a view can't widen past it
    trimmed = sim.ItemKNN(S, candidate_items=a)
    assert set(trimmed.view(allowed_items=None).recommend(hist, k=60, exclude_seen=False)[0]) <= set(a)


def test_block_size_validation():
    S = _S(10, False, seed=1)
    for ok in ("auto", None, 1, 1000):
        sim.ItemKNN(S, block_size=ok)
    for bad in (0, -1, "big", 1.5, True):
        with pytest.raises(ValueError, match="block_size"):
            sim.ItemKNN(S, block_size=bad)


def test_pickle_round_trip_keeps_layout_and_mask():
    rng = np.random.default_rng(5)
    S = _S(80, True, seed=5)
    ext = rng.choice(10**9, size=80, replace=False)
    rec = sim.ItemKNN(S, item_ids=ext, candidate_items=ext[5:70], block_size=None).view(allowed_items=ext[10:60])
    back = pickle.loads(pickle.dumps(rec))
    assert back.n_items == rec.n_items and back._model.parts()[-1] is None  # block size as given
    for _ in range(20):
        hist = rng.choice(ext, size=6)
        a, b = rec.recommend(hist, k=15), back.recommend(hist, k=15)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int32), b[1].view(np.int32))


def test_rejects_bad_input():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="square"):
        sim.ItemKNN(sp.random_array((3, 4), density=0.5, format="csr"))
    with pytest.raises(ValueError, match="unique"):
        sim.ItemKNN(S, item_ids=np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match="one id per item"):
        sim.ItemKNN(S, item_ids=np.arange(3))
    with pytest.raises(TypeError, match="integer ids"):
        sim.ItemKNN(S, item_ids=np.arange(10.0))
    rec = sim.ItemKNN(S)
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
    sim.ItemKNN(S).recommend_batch(H)
    for m, b in zip((S, H), before):
        assert all(np.array_equal(x, y) for x, y in zip(b, (m.data, m.indices, m.indptr)))


def test_any_input_form_same_model():
    """Duplicates, explicit zeros, unsorted rows, other formats and dtypes:
    the same model as the canonical float32 CSR, array for array."""
    S = _S(40, True, seed=9)
    want = sim.ItemKNN(S)._model.parts()
    coo = S.tocoo()
    # each entry as two exact halves, shuffled, plus explicit zeros
    rows = np.concatenate([coo.row, coo.row, [0, 3]])
    cols = np.concatenate([coo.col, coo.col, [1, 2]])
    vals = np.concatenate([coo.data / 2, coo.data / 2, [0.0, 0.0]])
    order = np.random.default_rng(1).permutation(rows.size)
    order = order[np.argsort(rows[order], kind="stable")]  # rows grouped, columns shuffled
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=40))])
    raw_csr = sp.csr_array((vals[order], cols[order], indptr), shape=S.shape)  # scipy keeps duplicates as given
    assert not raw_csr.has_canonical_format
    raw_coo = sp.coo_array((vals, (rows, cols)), shape=S.shape)
    for m in (raw_csr, raw_coo, S.tocsc(), S.astype(np.float64), sp.csr_matrix(S)):
        got = sim.ItemKNN(m)._model.parts()
        for a, b in zip(got[:5], want[:5]):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("mapped", [False, True])
def test_concurrent_calls_identical(tmp_path, mapped):
    """One model shared by threads (scratch per OS thread, GIL released), built or mapped from disk."""
    rng = np.random.default_rng(2)
    rec = sim.ItemKNN(sp.random_array((500, 500), density=0.05, rng=rng, format="csr"))
    hists = [rng.choice(500, size=20) for _ in range(64)]
    want = [rec.recommend(h, k=10) for h in hists]
    if mapped:
        rec.save(tmp_path / "m")
        rec = sim.ItemKNN.load(tmp_path / "m")
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda h: rec.recommend(h, k=10), hists * 4))
    for (gi, gs), (wi, ws) in zip(got, want * 4):
        assert np.array_equal(gi, wi) and np.array_equal(gs, ws)


def test_empty_models_and_masks():
    for S in (sp.csr_array((0, 0)), sp.csr_array((5, 5))):
        rec = sim.ItemKNN(S)
        assert rec.n_items == 0
        assert rec.recommend([0, 1], k=3)[0].size == 0
        assert rec.recommend_batch(sp.csr_array((2, S.shape[1]))).nnz == 0
        assert pickle.loads(pickle.dumps(rec)).n_items == 0
    rec = sim.ItemKNN(_S(20, False, seed=6), candidate_items=[])
    assert rec.recommend([5, 6, 7], k=5)[0].size == 0


def test_zero_as_a_real_external_id():
    S = _S(10, False, seed=8)
    ext = np.arange(10) * 7 - 14  # item 2 has id 0
    rec = sim.ItemKNN(S, item_ids=ext)
    D = S.toarray()
    hist = [ext[j] for j in range(10) if D[j].any()][:3]
    want = _reference(S, ext, hist, None, 10, True, None)
    assert np.array_equal(rec.recommend(hist, k=10)[0], want[0])
    assert np.array_equal(rec.recommend([0], k=10)[0], _reference(S, ext, [0], None, 10, True, None)[0])


def test_batch_skips_dropped_columns_and_int64_indptr(monkeypatch):
    S = _S(30, False, seed=9)  # item 0 is dropped
    rec = sim.ItemKNN(S)
    H = sp.csr_array(([1.0, 1.0, 1.0], ([0, 0, 1], [0, 5, 0])), shape=(2, 30))
    R = rec.recommend_batch(H, k=5)
    assert np.array_equal(R.indices[R.indptr[0] : R.indptr[1]], rec.recommend([5], k=5)[0])
    assert R.indptr[2] == R.indptr[1]  # a history of dropped items only
    import similaripy._serve as serve

    monkeypatch.setattr(serve, "_INT32_MAX", 0)
    assert rec.recommend_batch(H, k=5).indptr.dtype == np.int64


def test_reuse_after_raise():
    rec = sim.ItemKNN(_S(30, False, seed=10))
    want = rec.recommend([5, 6], k=5)
    with pytest.raises(ValueError, match="unknown"):
        rec.recommend([5, 6, 999], k=5, unknown="raise")
    got = rec.recommend([5, 6], k=5)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_boundary_types():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="fit in int64"):
        sim.ItemKNN(S, item_ids=np.array([2**63] + list(range(9)), dtype=np.uint64))
    rec = sim.ItemKNN(S)
    with pytest.raises(ValueError, match="fit in int64"):
        rec.recommend([2**63])
    with pytest.raises(ValueError, match="1-d"):
        rec.recommend([[5], [6]])
    with pytest.raises(TypeError, match="ItemKNN.mask"):
        rec.recommend([5], allowed=[3, 4])
    assert rec.recommend(np.array([5, 6], dtype=np.uint32), k=2**70)[0].size > 0
    bad = sp.csr_array(np.array([[0.0, np.inf], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        sim.ItemKNN(bad)
    with pytest.raises(ValueError, match="finite"):
        sim.ItemKNN(sp.csr_array(np.array([[0.0, 1e300], [1.0, 0.0]])))


def _bits(a):
    return a.view(np.int32) if a.dtype == np.float32 else a


@pytest.mark.parametrize("mmap", [True, False])
@pytest.mark.parametrize("block_size", ["auto", None, 7])
def test_save_load_round_trip(tmp_path, mmap, block_size):
    rng = np.random.default_rng(11)
    keep = np.ones(80)
    keep[[3, 4, 5]] = 0
    S = _S(80, True, seed=11) @ sp.diags_array(keep)  # items 3-5: no column
    S = sp.diags_array(np.r_[np.ones(5), 0, np.ones(74)]) @ S  # item 5: no row either (dropped)
    ext = rng.choice(10**12, size=80, replace=False)
    rec = sim.ItemKNN(S, item_ids=ext, candidate_items=ext[2:75], block_size=block_size).view(allowed_items=ext[5:60])
    assert rec.n_items <= 79  # item 5 dropped
    rec.save(tmp_path / "m")
    back = sim.ItemKNN.load(tmp_path / "m", mmap=mmap)
    for a, b in zip(rec._model.parts(), back._model.parts()):
        assert a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
    assert back._model.parts()[-1] == block_size
    for _ in range(20):
        hist = rng.choice(ext, size=8)
        a, b = rec.recommend(hist, k=15), back.recommend(hist, k=15)
        assert np.array_equal(a[0], b[0]) and np.array_equal(_bits(a[1]), _bits(b[1]))
    H = sp.random_array((6, 80), density=0.1, format="csr", rng=1)
    assert (rec.recommend_batch(H, k=5) != back.recommend_batch(H, k=5)).nnz == 0
    with pytest.raises(ValueError, match="another model"):
        back.recommend(ext[:3], allowed=rec.mask(ext[:10]))
    # plain .npy + JSON: inspectable without similaripy
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["format"] == 1 and meta["n_items"] == back.n_items and meta["block_size"] == block_size
    assert np.array_equal(np.load(tmp_path / "m" / "item_ids.npy"), ext)
    # pickling a loaded (mapped) model copies its arrays
    assert (
        pickle.loads(pickle.dumps(back)).recommend(ext[:4], k=3)[0].tolist() == back.recommend(ext[:4], k=3)[0].tolist()
    )


def test_load_files_written_by_numpy(tmp_path):
    """Headers as numpy writes them (padding for growth): load reads them."""
    S = _S(30, False, seed=12)
    rec = sim.ItemKNN(S)
    rec.save(tmp_path / "m")
    for f in (tmp_path / "m").glob("*.npy"):
        np.save(f, np.load(f))
    back = sim.ItemKNN.load(tmp_path / "m")
    assert np.array_equal(back.recommend([1, 2, 3])[0], rec.recommend([1, 2, 3])[0])


def test_save_never_overwrites_by_default(tmp_path):
    a, b = sim.ItemKNN(_S(20, False, seed=1)), sim.ItemKNN(_S(20, False, seed=2))
    a.save(tmp_path / "m")
    with pytest.raises(FileExistsError, match="overwrite"):
        b.save(tmp_path / "m")
    mapped = sim.ItemKNN.load(tmp_path / "m")
    want = mapped.recommend([1, 2], k=5)
    b.save(tmp_path / "m", overwrite=True)  # new files renamed into place
    assert np.array_equal(mapped.recommend([1, 2], k=5)[0], want[0])  # the old mapping still reads
    assert np.array_equal(sim.ItemKNN.load(tmp_path / "m").recommend([1, 2], k=5)[0], b.recommend([1, 2], k=5)[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m"]  # no temp dirs left
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not a saved model"):
        a.save(tmp_path / "other", overwrite=True)
    assert (tmp_path / "other" / "keep.txt").exists()
    (tmp_path / "empty").mkdir()
    a.save(tmp_path / "empty")  # an empty directory is fine


def _corrupt(d, name, fn):
    np.save(d / f"{name}.npy", fn(np.load(d / f"{name}.npy")))


def _meta(d, **kw):
    meta = json.loads((d / "meta.json").read_text())
    meta.update(kw)
    (d / "meta.json").write_text(json.dumps(meta))


def _swap_first_row(a):
    a = a.copy()
    a[[0, 1]] = a[[1, 0]]
    return a


@pytest.mark.parametrize(
    "corrupt, error",
    [
        (lambda d: (d / "indices.npy").unlink(), FileNotFoundError),
        (lambda d: (d / "meta.json").unlink(), FileNotFoundError),
        (lambda d: (d / "data.npy").write_bytes((d / "data.npy").read_bytes()[:-4]), "file size"),
        (lambda d: (d / "data.npy").write_bytes((d / "data.npy").read_bytes() + b"\0" * 4), "file size"),
        (lambda d: (d / "data.npy").write_bytes(b"not numpy at all"), "npy"),
        (lambda d: _corrupt(d, "indices", lambda a: a.astype(np.int64)), "<i4"),
        (lambda d: _corrupt(d, "data", lambda a: a.astype(">f4")), "<f4"),
        (lambda d: _corrupt(d, "data", lambda a: np.asfortranarray(np.stack([a, a], axis=1))), "<f4"),
        (lambda d: np.save(d / "data.npy", np.array([1.0] * 3, dtype=object), allow_pickle=True), "<f4"),
        (lambda d: _corrupt(d, "indptr", lambda a: np.r_[a[0], a[2], a[1], a[3:]]), "monotone"),
        (lambda d: _corrupt(d, "indptr", lambda a: a[:-1]), "indptr"),
        (lambda d: _corrupt(d, "indices", lambda a: np.r_[999, a[1:]].astype(np.int32)), "range"),
        (lambda d: _corrupt(d, "indices", _swap_first_row), "sorted"),
        (lambda d: _corrupt(d, "indices", lambda a: np.r_[a[1], a[1:]].astype(np.int32)), "sorted"),
        (lambda d: _corrupt(d, "data", lambda a: np.r_[np.float32(np.nan), a[1:]]), "finite"),
        (lambda d: _corrupt(d, "new_to_old", lambda a: np.r_[a[1], a[1:]].astype(np.int32)), "distinct"),
        (lambda d: _corrupt(d, "new_to_old", lambda a: np.r_[-1, a[1:]].astype(np.int32)), "distinct"),
        (lambda d: _corrupt(d, "item_ids", lambda a: np.r_[a[1], a[1:]]), "unique"),
        (lambda d: _corrupt(d, "item_ids", lambda a: a[:-1]), "one id per item"),
        (lambda d: _corrupt(d, "allowed", lambda a: np.r_[a, a]), "one bit per item"),
        (lambda d: (d / "meta.json").write_text("{not json"), "meta.json"),
        (lambda d: _meta(d, format=2), "newer"),
        (lambda d: _meta(d, format="1"), "integer"),
        (lambda d: _meta(d, kind="UserKNN"), "not a saved"),
        (lambda d: _meta(d, value_dtype="bfloat16"), "value dtype"),
        (lambda d: _meta(d, n_items=3), "match"),
        (lambda d: _meta(d, n_cols=10), "distinct"),
        (lambda d: _meta(d, block_size=0), "block_size"),
        (lambda d: _meta(d, allowed=False, item_ids=None), "true or false"),
    ],
)
def test_load_rejects_corrupted_files(tmp_path, corrupt, error):
    S = _S(40, False, seed=13)
    rec = sim.ItemKNN(S, item_ids=np.arange(40) * 3 + 1).view(allowed_items=np.arange(10) * 3 + 1)
    d = tmp_path / "m"
    rec.save(d)
    corrupt(d)
    for mmap in (True, False):
        if isinstance(error, str):
            with pytest.raises(ValueError, match=error):
                sim.ItemKNN.load(d, mmap=mmap)
        else:
            with pytest.raises(error):
                sim.ItemKNN.load(d, mmap=mmap)
