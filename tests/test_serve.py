"""Serving engine: `ItemKNN` against a dense float64 reference."""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from similaripy.serving import ItemKNN


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


def _reference(S, ext, history, weights, k, exclude_seen, include=None, exclude=None):
    """Top-k of score = h @ S (> 0), ties: more column entries, then lower column."""
    n = S.shape[0]
    D = S.toarray()
    counts = np.bincount(S.indices, minlength=n)
    pos = {e: i for i, e in enumerate(ext)}
    h = np.zeros(n)
    for j, e in enumerate(history):
        if e in pos:
            h[pos[e]] += 1.0 if weights is None else weights[j]
    scores = h @ D
    keep = scores > 0
    if exclude_seen:
        keep[[pos[e] for e in history if e in pos]] = False
    if include is not None:
        keep &= np.isin(ext, include)
    if exclude is not None:
        keep &= ~np.isin(ext, exclude)
    cols = np.flatnonzero(keep)
    order = np.lexsort((cols, -counts[cols], -scores[cols]))[:k]
    return ext[cols[order]], scores[cols[order]]


def _same(a, b):
    """Same ids and the same score bits."""
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int32), b[1].view(np.int32))


def _same_set(a, b):
    """`_same` up to order (sorted=False: the order can differ between calls)."""
    ia, ib = np.argsort(a[0], kind="stable"), np.argsort(b[0], kind="stable")
    return _same((a[0][ia], a[1][ia]), (b[0][ib], b[1][ib]))


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
    rec = ItemKNN(S, item_ids=ext if external else None)
    assert np.array_equal(rec.item_ids, ext)  # item 0 (no similarities) still listed
    unknown = -5 if external else n + 3
    for trial in range(30):
        hist = rng.choice(ext, size=rng.integers(0, 8))
        hist = np.append(hist, hist[:1])  # a duplicate
        hist = np.append(hist, [ext[0], unknown])  # no similarities + unknown: ignored
        weights = rng.integers(1, 4, size=hist.size).astype(np.float64) if trial % 2 else None
        include = np.r_[rng.choice(ext, size=n // 2, replace=False), unknown] if trial % 3 == 0 else None
        exclude = np.r_[rng.choice(ext, size=n // 5, replace=False), unknown] if trial % 4 == 0 else None
        k = int(rng.integers(0, n + 3))
        for exclude_seen in (True, False):
            opts = {"k": k, "exclude_seen": exclude_seen, "exclude": exclude}
            got = rec.recommend(hist, weights=weights, include=include, **opts)
            want = _reference(S, ext, hist, weights, k, exclude_seen, include, exclude)
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_allclose(got[1], want[1], rtol=1e-6)
            unsorted = rec.recommend(hist, weights=weights, include=include, sorted=False, **opts)
            assert sorted(zip(unsorted[0], unsorted[1].view(np.int32))) == sorted(zip(got[0], got[1].view(np.int32)))
            if include is not None:  # trimmed S == a view == the per-call include (small integers under the head: bits)
                for other in (
                    ItemKNN(S, item_ids=ext if external else None, candidate_items=include),
                    rec.view(include=include),
                ):
                    assert _same(other.recommend(hist, weights=weights, **opts), got)


def test_default_path_and_dtypes():
    rec = ItemKNN(_S(30, False, seed=3))
    ids, scores = rec.recommend([5, 6, 7], k=5)
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    assert np.all(np.diff(scores) <= 0)
    assert not np.isin(ids, [5, 6, 7]).any()
    assert rec.recommend([], k=5)[0].size == 0
    assert rec.recommend([5], k=0)[0].size == 0
    assert 0 < rec.recommend([5, 6], k=2**62)[0].size < 30  # huge k clamped


def test_keyword_only_options():
    rec = ItemKNN(_S(30, False, seed=3))
    with pytest.raises(TypeError):
        rec.recommend([5, 6], None, 5)  # weights, k: by name
    with pytest.raises(TypeError):
        rec.recommend_batch([[5]], None, 5)
    with pytest.raises(TypeError):
        ItemKNN(_S(30, False, seed=3), np.arange(30))


def test_item_ids_property():
    S = _S(20, False, seed=2)
    assert np.array_equal(ItemKNN(S).item_ids, np.arange(20))
    rec = ItemKNN(S, item_ids=np.arange(100, 120))
    ids = rec.item_ids
    assert np.array_equal(ids, np.arange(100, 120)) and ids.dtype == np.int64
    with pytest.raises(ValueError):
        ids[0] = 5  # read-only
    assert np.array_equal(rec.view(include=[105]).item_ids, ids)


def test_unknown_ids_are_ignored():
    # 100 has no similarities (dropped internally), 7 is no item: both ignored, never an error
    rec = ItemKNN(_S(20, False, seed=2), item_ids=np.arange(100, 120))
    assert _same(rec.recommend([105, 7, 100], k=3), rec.recommend([105], k=3))
    assert _same(rec.recommend([105], include=[7, 100], k=3), rec.recommend([105], include=[], k=3))
    assert _same(rec.recommend([105], exclude=[7, 100], k=3), rec.recommend([105], k=3))
    assert _same(rec.view(include=np.arange(100, 130)).recommend([105], k=3), rec.recommend([105], k=3))
    assert rec.recommend([105], include=[], k=3)[0].size == 0


def test_float_bits_independent_of_order_and_duplicates():
    rng = np.random.default_rng(0)
    rec = ItemKNN(sp.random_array((200, 200), density=0.1, rng=rng, format="csr"))
    hist = rng.choice(200, size=30, replace=False)
    ids, scores = rec.recommend(hist, k=20)
    for other in (
        rec.recommend(hist[::-1], k=20),
        rec.recommend(np.concatenate([hist, hist[:5]]), weights=np.r_[np.full(30, 1.0), np.zeros(5)], k=20),
    ):
        assert _same(other, (ids, scores))
    w = np.ones(30)
    w[0] = 2.0  # [a, a] == [a] with weight 2.0
    assert _same(rec.recommend(np.r_[hist, hist[:1]], k=20), rec.recommend(hist, weights=w, k=20))


@pytest.mark.parametrize("block", [7, None])
def test_batch_matches_single_bits(block):
    rng = np.random.default_rng(4)
    S = sp.random_array((300, 300), density=0.05, rng=rng, format="csr")
    ext = rng.choice(10**9, size=300, replace=False)
    rec = ItemKNN(S, item_ids=ext, block_size=block)
    one_block = ItemKNN(S, item_ids=ext, block_size=None)
    users = 50
    hists = [rng.choice(ext, size=rng.integers(0, 15)) for _ in range(users)]
    hists[3] = np.r_[hists[3], -7]  # an unknown id
    weights = [None if u % 5 == 0 else rng.random(len(h)) * 3 for u, h in enumerate(hists)]
    weights[1] = np.zeros(len(hists[1]))  # zero weights: still history (seen)
    include = [None if u % 3 else rng.choice(ext, 150, replace=False) for u in range(users)]
    exclude = [None if u % 4 else rng.choice(ext, 30, replace=False) for u in range(users)]
    for opts in (
        {},
        {"weights": weights},
        {"include": include, "exclude": exclude, "weights": weights},
        {"sorted": False, "exclude_seen": False},
    ):
        got_ids, got_scores = rec.recommend_batch(hists, k=10, num_threads=2, **opts)
        assert len(got_ids) == len(got_scores) == users
        for u in range(users):
            single = {key: (v[u] if isinstance(v, list) else v) for key, v in opts.items()}
            want = rec.recommend(hists[u], k=10, **single)
            assert (_same if opts.get("sorted", True) else _same_set)((got_ids[u], got_scores[u]), want)
            if not opts:  # blocking never changes bits
                assert _same(one_block.recommend(hists[u], k=10), want)


def test_batch_from_a_user_item_matrix():
    """The docs' recipe: URM rows -> histories; the same as dot_product(URM, S)."""
    import similaripy as sim

    rng = np.random.default_rng(6)
    URM = sp.random_array((40, 60), density=0.1, rng=rng, format="csr")
    S = sim.cosine(URM.T, k=20)
    item_ids = rng.choice(10**6, size=60, replace=False)
    rec = ItemKNN(S, item_ids=item_ids)
    histories = np.split(item_ids[URM.indices], URM.indptr[1:-1])
    weights = np.split(URM.data, URM.indptr[1:-1])
    ids, scores = rec.recommend_batch(histories, weights=weights, k=5)
    D = sim.dot_product(URM, S, k=5, filter_cols=URM)
    for u in range(40):
        row = D[[u]]
        assert set(ids[u]) == set(item_ids[row.indices])
        np.testing.assert_allclose(np.sort(scores[u]), np.sort(row.data), rtol=1e-5)


def test_view_include_exclude_and_per_call():
    S = _S(60, False, seed=12)
    rec = ItemKNN(S)
    hist = [5, 6, 7, 8]
    full = list(rec.recommend(hist, k=60, exclude_seen=False)[0])
    a, b = list(range(0, 60, 2)), list(range(0, 60, 3))
    va = rec.view(include=a)
    # view include AND per-call include, minus per-call exclude
    got = va.recommend(hist, k=60, exclude_seen=False, include=b, exclude=[12])[0]
    assert list(got) == [i for i in full if i % 6 == 0 and i != 12]
    # a view's exclude
    assert list(rec.view(exclude=b).recommend(hist, k=60, exclude_seen=False)[0]) == [i for i in full if i % 3]
    # a view of a view narrows (the parent's rules AND the new ones), view() keeps them
    country = rec.view(include=range(30), exclude=[4])
    package = country.view(include=range(10, 60, 2))
    assert list(package.recommend(hist, k=60, exclude_seen=False)[0]) == [
        i for i in full if 10 <= i < 30 and i % 2 == 0
    ]
    assert list(package.view(exclude=[12]).recommend(hist, k=60, exclude_seen=False)[0]) == [
        i for i in full if 10 <= i < 30 and i % 2 == 0 and i != 12
    ]
    assert list(country.view().recommend(hist, k=60, exclude_seen=False)[0]) == [i for i in full if i < 30 and i != 4]
    # to widen, start from the base model
    assert list(rec.view().recommend(hist, k=60, exclude_seen=False)[0]) == full
    # candidate_items trims for good: a view can't widen past it
    trimmed = ItemKNN(S, candidate_items=a)
    assert set(trimmed.view().recommend(hist, k=60, exclude_seen=False)[0]) <= set(a)


def test_block_size_validation():
    S = _S(10, False, seed=1)
    for ok in ("auto", None, 1, 1000):
        ItemKNN(S, block_size=ok)
    for bad in (0, -1, "big", 1.5, True):
        with pytest.raises(ValueError, match="block_size"):
            ItemKNN(S, block_size=bad)


def test_item_order_validation():
    S = _S(10, False, seed=1)
    for ok in ("auto", "clustered", "popularity", "original"):
        ItemKNN(S, item_order=ok)
    for bad in ("fast", "Auto", None, 1):
        with pytest.raises((ValueError, TypeError), match="item_order"):
            ItemKNN(S, item_order=bad)


def test_item_orders_same_scores_ties_by_internal_order(tmp_path):
    """Past the clustered head (16384 items): every order gives the same
    scores; ties go to the lower internal id (the documented tie rule). A
    trimmed model (`candidate_items`) is ordered after the trim: its
    candidates with a column take the first internal ids (`n_active`), and it
    answers as the full S with the candidates as include (D116). A clustered
    tail survives pickle and save/load and doesn't depend on the threads."""
    n, k = 20_000, 50
    rng = np.random.default_rng(7)
    S = sp.random_array((n, n), density=4 / n, rng=rng, format="csr", dtype=np.float64)
    S.data = np.round(S.data * 3 + 1)  # small integers: exact sums in any order
    counts = np.bincount(S.indices, minlength=n)
    cand = rng.choice(n, size=n // 2, replace=False)
    hists = [rng.choice(n, size=30) for _ in range(20)]
    layout = {}
    for order in ("auto", "clustered", "popularity", "original"):
        for trim in (False, True):
            opts = {"item_order": order, "candidate_items": cand if trim else None}
            rec = ItemKNN(S, **opts)
            new_to_old = rec._model.parts()[3]
            if order == "clustered":
                rec.save(tmp_path / str(trim))
                for back in (pickle.loads(pickle.dumps(rec)), ItemKNN.load(tmp_path / str(trim))):
                    for a, b in zip(rec._model.parts(), back._model.parts()):
                        assert a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
                assert np.array_equal(ItemKNN(S, **opts, num_threads=1)._model.parts()[3], new_to_old)
            if trim:
                n_cand = np.count_nonzero(counts[cand])
                assert rec._model.n_active == n_cand
                assert set(new_to_old[:n_cand]) == set(cand[counts[cand] > 0])
            else:
                layout[order] = new_to_old
            internal = np.full(n, -1)
            internal[new_to_old] = np.arange(len(new_to_old))
            for h in hists:
                scores = np.asarray(S[h].sum(axis=0)).ravel()
                keep = scores > 0
                keep[h] = False
                if trim:
                    keep &= np.isin(np.arange(n), cand)
                cols = np.flatnonzero(keep)
                best = cols[np.lexsort((internal[cols], -scores[cols]))[:k]]
                assert _same(rec.recommend(h, k=k), (best, scores[best].astype(np.float32))), (order, trim)
    pop, orig = layout["popularity"], layout["original"]
    assert np.all(np.diff(counts[pop]) <= 0)
    n_col = np.count_nonzero(counts)  # 'original': S's order, items with no column last
    assert np.all(counts[orig[:n_col]] > 0) and np.all(np.diff(orig[:n_col]) > 0) and np.all(np.diff(orig[n_col:]) > 0)
    assert np.array_equal(layout["auto"], layout["clustered"])
    assert np.array_equal(layout["clustered"][:16384], pop[:16384])
    assert not np.array_equal(layout["clustered"], pop)


def test_pickle_round_trip_keeps_layout_and_mask():
    rng = np.random.default_rng(5)
    S = _S(80, True, seed=5)
    ext = rng.choice(10**9, size=80, replace=False)
    rec = ItemKNN(S, item_ids=ext, candidate_items=ext[5:70], block_size=None).view(include=ext[10:60])
    back = pickle.loads(pickle.dumps(rec))
    assert np.array_equal(back.item_ids, ext) and back._model.parts()[-1] is None  # block size as given
    for _ in range(20):
        hist = rng.choice(ext, size=6)
        assert _same(rec.recommend(hist, k=15), back.recommend(hist, k=15))


def test_pickle_from_another_version_is_refused():
    func, (cls, fmt, *parts) = ItemKNN(_S(10, False, seed=1)).__reduce__()
    assert func(cls, fmt, *parts).item_ids.size == 10
    for other in (fmt + 1, parts[0]):  # a newer tag; an untagged tuple (indptr first)
        with pytest.raises(ValueError, match="another similaripy version"):
            func(cls, other, *parts)


def test_rejects_bad_input():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="square"):
        ItemKNN(sp.random_array((3, 4), density=0.5, format="csr"))
    with pytest.raises(ValueError, match="unique"):
        ItemKNN(S, item_ids=np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match="one id per item"):
        ItemKNN(S, item_ids=np.arange(3))
    with pytest.raises(TypeError, match="integer ids"):
        ItemKNN(S, item_ids=np.arange(10.0))
    rec = ItemKNN(S)
    with pytest.raises(TypeError, match="integer ids"):
        rec.recommend([1.9])
    with pytest.raises(TypeError, match="integer ids"):
        rec.recommend([1], include=[1.5])
    with pytest.raises(ValueError, match="finite"):
        rec.recommend([5, 6], weights=[1.0, np.nan])
    with pytest.raises(ValueError, match="len"):
        rec.recommend([5, 6], weights=[1.0])
    with pytest.raises(ValueError, match="k must"):
        rec.recommend([5], k=-1)
    # batch: lists with one entry per user, weights aligned with each history
    with pytest.raises(TypeError, match="one entry per user"):
        rec.recommend_batch(np.array([[5, 6]]))
    with pytest.raises(TypeError, match="view"):
        rec.recommend_batch([[5], [6]], include=np.array([1, 2]))
    with pytest.raises(ValueError, match="one entry per user"):
        rec.recommend_batch([[5], [6]], exclude=[[1]])
    with pytest.raises(ValueError, match="one weight per history id"):
        rec.recommend_batch([[5, 6], [6]], weights=[[1.0], None])
    with pytest.raises(ValueError, match="finite"):
        rec.recommend_batch([[5]], weights=[[np.inf]])
    # common mistakes get a message naming the argument
    with pytest.raises(TypeError, match="sparse"):
        ItemKNN(S.toarray())
    with pytest.raises(ValueError, match="history must be 1-d, got a scalar"):
        rec.recommend(5)
    with pytest.raises(ValueError, match="weights must be 1-d, got a scalar"):
        rec.recommend([5], weights=1.0)
    with pytest.raises(ValueError, match=r"histories\[0\] must be 1-d, got a scalar"):
        rec.recommend_batch([5, 6])
    with pytest.raises(ValueError, match=r"weights\[0\] must be 1-d, got a scalar"):
        rec.recommend_batch([[5]], weights=[1.0])
    with pytest.raises(TypeError, match=r"histories\[1\] is None"):
        rec.recommend_batch([[5], None])


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("block", [7, None])
def test_score_dense_reference_recommend_bits_and_masks(external, block):
    n = 60
    S = _S(n, True, seed=21)
    rng = np.random.default_rng(2)
    ext = rng.choice(10**9, size=n, replace=False) if external else np.arange(n)
    rec = ItemKNN(S, item_ids=ext if external else None, block_size=block)
    D = S.toarray()
    pos = {e: i for i, e in enumerate(ext)}
    unknown = -5 if external else n + 3
    items = np.r_[ext[::-1], ext[5], unknown]  # every item, a duplicate, an unknown id
    for trial in range(20):
        hist = rng.choice(ext, size=rng.integers(0, 10))
        w = rng.integers(-2, 4, size=hist.size).astype(np.float64) if trial % 2 else None
        h = np.zeros(n)
        for j, e in enumerate(hist):
            h[pos[e]] += 1.0 if w is None else w[j]
        want = np.r_[(h @ D)[::-1], (h @ D)[5], 0.0]
        got = rec.score(hist, items, weights=w)
        assert got.dtype == np.float32 and got.shape == items.shape
        np.testing.assert_array_equal(got, want)  # small integers: exact, negatives and zeros kept
        top = rec.recommend(hist, weights=w, k=n, exclude_seen=False)
        assert np.array_equal(rec.score(hist, top[0], weights=w).view(np.int32), top[1].view(np.int32))
        # model-level availability holds (a view, a trimmed build): 0.0 outside it
        a = rng.choice(ext, size=n // 2, replace=False)
        for other in (
            rec.view(include=a),
            rec.view(exclude=np.setdiff1d(ext, a)),
            ItemKNN(S, item_ids=ext if external else None, candidate_items=a),
        ):
            np.testing.assert_array_equal(other.score(hist, items, weights=w), np.where(np.isin(items, a), want, 0.0))
    assert rec.score([], items).tolist() == [0.0] * items.size
    assert rec.score([ext[5]], []).size == 0


def test_score_batch_matches_single_bits():
    rng = np.random.default_rng(8)
    S = sp.random_array((300, 300), density=0.05, rng=rng, format="csr")
    ext = rng.choice(10**9, size=300, replace=False)
    rec = ItemKNN(S, item_ids=ext, block_size=7).view(exclude=ext[:30])
    users = 40
    hists = [rng.choice(ext, size=rng.integers(0, 15)) for _ in range(users)]
    items = [rng.choice(ext, size=rng.integers(0, 50)) for _ in range(users)]
    weights = [None if u % 5 == 0 else rng.random(len(h)) * 3 for u, h in enumerate(hists)]
    for w in (None, weights):
        got = rec.score_batch(hists, items, weights=w, num_threads=2)
        assert len(got) == users
        for u in range(users):
            want = rec.score(hists[u], items[u], weights=None if w is None else w[u])
            assert np.array_equal(got[u].view(np.int32), want.view(np.int32))
    assert rec.score_batch([], []) == []


def test_score_rejects_bad_input():
    rec = ItemKNN(_S(10, False, seed=1))
    with pytest.raises(TypeError, match="items must be integer ids"):
        rec.score([5], [1.5])
    with pytest.raises(ValueError, match="len"):
        rec.score([5, 6], [1], weights=[1.0])
    with pytest.raises(ValueError, match="finite"):
        rec.score([5], [1], weights=[np.nan])
    with pytest.raises(TypeError):
        rec.score([5], [1], [1.0])  # weights: by name
    with pytest.raises(ValueError, match="items must have one entry per user"):
        rec.score_batch([[5], [6]], [[1]])
    with pytest.raises(TypeError, match=r"items\[1\] is None"):
        rec.score_batch([[5], [6]], [[1], None])
    with pytest.raises(ValueError, match="one weight per history id"):
        rec.score_batch([[5, 6]], [[1]], weights=[[1.0]])


def test_caller_arrays_unchanged():
    S = _S(20, True, seed=5)
    hists = [np.array([3, 4, 5]), np.array([6])]
    before = [S.data.copy(), S.indices.copy(), S.indptr.copy()] + [h.copy() for h in hists]
    rec = ItemKNN(S)
    rec.recommend_batch(hists, weights=[np.ones(3), None])
    rec.recommend(hists[0], include=hists[1])
    after = [S.data, S.indices, S.indptr] + hists
    assert all(np.array_equal(x, y) for x, y in zip(before, after))


def test_any_input_form_same_model():
    """Duplicates, explicit zeros, unsorted rows, other formats and dtypes:
    the same model as the canonical float32 CSR, array for array."""
    S = _S(40, True, seed=9)
    want = ItemKNN(S)._model.parts()
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
        got = ItemKNN(m)._model.parts()
        for a, b in zip(got[:5], want[:5]):
            assert np.array_equal(a, b)


def test_int64_indptr_input_same_model(monkeypatch):
    """S past 2^31 - 1 entries comes with an int64 indptr (forced with a limit
    of 0, CSR and COO routes): the same model; its indptr is int64 either way."""
    from similaripy import _s_plus

    S = _S(40, True, seed=9)
    want = ItemKNN(S)._model.parts()
    assert want[0].dtype == np.int64
    monkeypatch.setattr(_s_plus, "_MAX_INT32_NNZ", 0)
    S64 = S.copy()
    S64.indptr, S64.indices = S.indptr.astype(np.int64), S.indices.astype(np.int64)
    for m in (S64, S.tocoo()):
        for a, b in zip(ItemKNN(m)._model.parts()[:5], want[:5]):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("mapped", [False, True])
def test_concurrent_calls_identical(tmp_path, mapped):
    """One model shared by threads (scratch per OS thread, GIL released), built or mapped from disk."""
    rng = np.random.default_rng(2)
    rec = ItemKNN(sp.random_array((500, 500), density=0.05, rng=rng, format="csr"))
    hists = [rng.choice(500, size=20) for _ in range(64)]
    incs = [None if i % 2 else rng.choice(500, size=200) for i in range(64)]
    want = [rec.recommend(h, k=10, include=i) for h, i in zip(hists, incs)]
    if mapped:
        rec.save(tmp_path / "m")
        rec = ItemKNN.load(tmp_path / "m")
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda hi: rec.recommend(hi[0], k=10, include=hi[1]), list(zip(hists, incs)) * 4))
    for g, w in zip(got, want * 4):
        assert _same(g, w)


def test_empty_models():
    for S in (sp.csr_array((0, 0)), sp.csr_array((5, 5))):
        rec = ItemKNN(S)
        assert rec.item_ids.size == S.shape[0]
        assert rec.recommend([0, 1], k=3)[0].size == 0
        ids, scores = rec.recommend_batch([[0], []])
        assert [a.size for a in ids] == [0, 0] and [a.size for a in scores] == [0, 0]
        assert rec.recommend_batch([]) == ([], [])
        assert pickle.loads(pickle.dumps(rec)).item_ids.size == S.shape[0]
    rec = ItemKNN(_S(20, False, seed=6), candidate_items=[])
    assert rec.recommend([5, 6, 7], k=5)[0].size == 0


def test_zero_as_a_real_external_id():
    S = _S(10, False, seed=8)
    ext = np.arange(10) * 7 - 14  # item 2 has id 0
    rec = ItemKNN(S, item_ids=ext)
    D = S.toarray()
    hist = [ext[j] for j in range(10) if D[j].any()][:3]
    want = _reference(S, ext, hist, None, 10, True)
    assert np.array_equal(rec.recommend(hist, k=10)[0], want[0])
    assert np.array_equal(rec.recommend([0], k=10)[0], _reference(S, ext, [0], None, 10, True)[0])


def test_reuse_after_an_error():
    rec = ItemKNN(_S(30, False, seed=10))
    want = rec.recommend([5, 6], k=5, include=list(range(30)))
    with pytest.raises(ValueError, match="finite"):
        rec.recommend([5, 6], weights=[1.0, np.nan], k=5, include=[3])
    assert _same(rec.recommend([5, 6], k=5, include=list(range(30))), want)


def test_boundary_types():
    S = _S(10, False, seed=1)
    with pytest.raises(ValueError, match="fit in int64"):
        ItemKNN(S, item_ids=np.array([2**63] + list(range(9)), dtype=np.uint64))
    rec = ItemKNN(S)
    with pytest.raises(ValueError, match="fit in int64"):
        rec.recommend([2**63])
    with pytest.raises(ValueError, match="1-d"):
        rec.recommend([[5], [6]])
    assert rec.recommend(np.array([5, 6], dtype=np.uint32), k=2**70)[0].size > 0
    bad = sp.csr_array(np.array([[0.0, np.inf], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        ItemKNN(bad)
    with pytest.raises(ValueError, match="finite"):
        ItemKNN(sp.csr_array(np.array([[0.0, 1e300], [1.0, 0.0]])))


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
    rec = ItemKNN(S, item_ids=ext, candidate_items=ext[2:75], block_size=block_size)
    with pytest.raises(ValueError, match="save the base model"):
        rec.view(include=ext[5:60]).save(tmp_path / "v")
    rec.save(tmp_path / "m")
    back = ItemKNN.load(tmp_path / "m", mmap=mmap, block_size=block_size)
    for a, b in zip(rec._model.parts(), back._model.parts()):
        assert a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
    assert back._model.parts()[-1] == block_size
    # The block size is a runtime setting, not saved (W1): `load` takes its own.
    assert ItemKNN.load(tmp_path / "m", mmap=mmap)._model.parts()[-1] == "auto"
    assert ItemKNN.load(tmp_path / "m", mmap=mmap, block_size=3)._model.parts()[-1] == 3
    with pytest.raises(ValueError, match="block_size"):
        ItemKNN.load(tmp_path / "m", mmap=mmap, block_size=0)
    for _ in range(20):
        hist = rng.choice(ext, size=8)
        a, b = rec.recommend(hist, k=15), back.recommend(hist, k=15)
        assert np.array_equal(a[0], b[0]) and np.array_equal(_bits(a[1]), _bits(b[1]))
    hists = [rng.choice(ext, size=8) for _ in range(6)]
    got, want = back.recommend_batch(hists, k=5), rec.recommend_batch(hists, k=5)
    assert all(_same((gi, gs), (wi, ws)) for gi, gs, wi, ws in zip(*got, *want))
    assert np.array_equal(back.item_ids, ext)
    # plain .npy + JSON: inspectable without similaripy
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["format"] == 1 and meta["n_cols"] == 80 and "block_size" not in meta
    assert np.array_equal(np.load(tmp_path / "m" / "item_ids.npy"), ext)
    assert np.load(tmp_path / "m" / "indptr.npy").dtype == np.int64  # one dtype, whatever S's nnz
    pairs = np.load(tmp_path / "m" / "pairs.npy")  # S's entries: a record array, (col, val) side by side
    assert pairs.dtype == np.dtype([("col", "<i4"), ("val", "<f4")])
    assert np.array_equal(pairs["col"], rec._model.parts()[1]) and np.array_equal(pairs["val"], rec._model.parts()[2])
    # pickling a loaded (mapped) model copies its arrays
    assert (
        pickle.loads(pickle.dumps(back)).recommend(ext[:4], k=3)[0].tolist() == back.recommend(ext[:4], k=3)[0].tolist()
    )


def test_load_files_written_by_numpy(tmp_path):
    """Headers as numpy writes them (padding for growth): load reads them."""
    S = _S(30, False, seed=12)
    rec = ItemKNN(S)
    rec.save(tmp_path / "m")
    for f in (tmp_path / "m").glob("*.npy"):
        np.save(f, np.load(f))
    back = ItemKNN.load(tmp_path / "m")
    assert np.array_equal(back.recommend([1, 2, 3])[0], rec.recommend([1, 2, 3])[0])


def test_save_only_into_a_new_or_empty_directory(tmp_path):
    a, b = ItemKNN(_S(20, False, seed=1)), ItemKNN(_S(20, False, seed=2))
    a.save(tmp_path / "m")
    mapped = ItemKNN.load(tmp_path / "m")
    want = mapped.recommend([1, 2], k=5)
    with pytest.raises(FileExistsError, match="new directory"):
        b.save(tmp_path / "m")  # a saved model is never replaced
    assert _same(mapped.recommend([1, 2], k=5), want)
    assert _same(ItemKNN.load(tmp_path / "m").recommend([1, 2], k=5), want)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="new directory"):
        a.save(tmp_path / "other")
    assert sorted(p.name for p in (tmp_path / "other").iterdir()) == ["keep.txt"]
    (tmp_path / "empty").mkdir()
    a.save(tmp_path / "empty")  # an empty directory is fine
    a.save(tmp_path / "new" / "v1")  # parents are created
    assert _same(ItemKNN.load(tmp_path / "new" / "v1").recommend([1, 2], k=5), want)


def test_concurrent_saves_to_one_path_keep_one_model(tmp_path):
    """Two savers on one path aren't supported (all but one fail), but the one
    that wins keeps a complete model: a loser never removes it."""
    a = ItemKNN(_S(20, False, seed=1))

    def save():
        try:
            a.save(tmp_path / "m")
        except OSError:
            pass

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda _: save(), range(16)))
    assert _same(ItemKNN.load(tmp_path / "m").recommend([1, 2], k=5), a.recommend([1, 2], k=5))


def _corrupt(d, name, fn):
    np.save(d / f"{name}.npy", fn(np.load(d / f"{name}.npy")))


def _field(name, fn):
    """A `_corrupt` function changing one field of `pairs.npy`."""

    def f(a):
        a = a.copy()
        a[name] = fn(a[name])
        return a

    return f


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
        (lambda d: (d / "pairs.npy").unlink(), FileNotFoundError),
        (lambda d: (d / "meta.json").unlink(), FileNotFoundError),
        (lambda d: (d / "pairs.npy").write_bytes((d / "pairs.npy").read_bytes()[:-8]), "file size"),
        (lambda d: (d / "pairs.npy").write_bytes((d / "pairs.npy").read_bytes() + b"\0" * 8), "file size"),
        (lambda d: (d / "pairs.npy").write_bytes(b"not numpy at all"), "npy"),
        (lambda d: _corrupt(d, "pairs", lambda a: a.astype([("col", "<i8"), ("val", "<f4")])), "col"),
        (lambda d: _corrupt(d, "indptr", lambda a: a.astype(np.int32)), "<i8"),
        (lambda d: _corrupt(d, "pairs", lambda a: a.astype([("col", "<i4"), ("val", ">f4")])), "val"),
        (lambda d: _corrupt(d, "pairs", lambda a: a["val"]), "col"),  # a plain float array
        (lambda d: _corrupt(d, "pairs", lambda a: np.asfortranarray(np.stack([a, a], axis=1))), "col"),
        (lambda d: np.save(d / "pairs.npy", np.array([1.0] * 3, dtype=object), allow_pickle=True), "col"),
        (lambda d: _corrupt(d, "indptr", lambda a: np.r_[a[0], a[2], a[1], a[3:]]), "monotone"),
        (lambda d: _corrupt(d, "indptr", lambda a: a[:-1]), "indptr"),
        (lambda d: _corrupt(d, "pairs", _field("col", lambda c: np.r_[999, c[1:]])), "range"),
        (lambda d: _corrupt(d, "pairs", _swap_first_row), "sorted"),
        (lambda d: _corrupt(d, "pairs", _field("col", lambda c: np.r_[c[1], c[1:]])), "sorted"),
        (lambda d: _corrupt(d, "pairs", _field("val", lambda v: np.r_[np.float32(np.nan), v[1:]])), "finite"),
        (lambda d: _corrupt(d, "new_to_old", lambda a: np.r_[a[1], a[1:]].astype(np.int32)), "distinct"),
        (lambda d: _corrupt(d, "new_to_old", lambda a: np.r_[-1, a[1:]].astype(np.int32)), "distinct"),
        (lambda d: _corrupt(d, "item_ids", lambda a: np.r_[a[1], a[1:]]), "unique"),
        (lambda d: _corrupt(d, "item_ids", lambda a: a[:-1]), "one id per item"),
        (lambda d: (d / "meta.json").write_text("{not json"), "meta.json"),
        (lambda d: _meta(d, format=2), "newer"),
        (lambda d: _meta(d, format="1"), "integer"),
        (lambda d: _meta(d, kind="UserKNN"), "not a saved"),
        (lambda d: _meta(d, value_dtype="bfloat16"), "value dtype"),
        (lambda d: _meta(d, n_items=3), "match"),
        (lambda d: _meta(d, n_cols=10), "one id per item"),
        (lambda d: _meta(d, n_cols=10**12), "one id per item"),  # checked before allocating
        (lambda d: _meta(d, item_ids=None), "true or false"),
    ],
)
def test_load_rejects_corrupted_files(tmp_path, corrupt, error):
    S = _S(40, False, seed=13)
    rec = ItemKNN(S, item_ids=np.arange(40) * 3 + 1)
    d = tmp_path / "m"
    rec.save(d)
    corrupt(d)
    for mmap in (True, False):
        if isinstance(error, str):
            with pytest.raises(ValueError, match=error):
                ItemKNN.load(d, mmap=mmap)
        else:
            with pytest.raises(error):
                ItemKNN.load(d, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_loads_a_committed_format_1_model(mmap):
    # 1.x loads every older 1.x format: these files were written by format 1 and
    # stay as they are. Changing what the files mean bumps the format instead.
    model = ItemKNN.load(Path(__file__).parent / "data" / "itemknn-format1", mmap=mmap)
    assert np.array_equal(model.item_ids, [10, 20, 30, 40, 50, 60, 70])
    f32 = lambda x: np.array(x, dtype=np.float32)
    assert _same(model.recommend([20, 40], k=3), ([30, 10, 50], f32([1.75, 0.75, 0.5])))
    got = model.recommend([10, 50, 70], weights=[1.0, 0.5, 3.0], k=3)
    assert _same(got, ([60, 40, 20], f32([1.0, 0.5, 0.5])))
    assert model._model.parts()[-1] == "auto"  # its saved block_size (null) ignored: a runtime setting


def test_strided_and_float64_coo_input():
    """Non-contiguous arrays are fine; COO float64 duplicates are cast before
    they are summed, as for CSR input: same model either way."""
    rng = np.random.default_rng(3)
    rows, cols = rng.integers(0, 30, 400), rng.integers(0, 30, 400)
    vals = rng.random(400) * 1e3  # not exact in float32
    order = np.argsort(rows, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=30))]
    raw_csr = sp.csr_array((vals[order], cols[order], indptr), shape=(30, 30))  # duplicates kept
    want = ItemKNN(raw_csr)._model.parts()
    coo = sp.coo_array((vals, (rows, cols)), shape=(30, 30))
    d2, i2 = np.repeat(vals[order], 2), np.repeat(cols[order], 2)
    strided = sp.csr_array((d2[::2], i2[::2], indptr), shape=(30, 30))
    for m in (coo, strided):
        got = ItemKNN(m)._model.parts()
        for a, b in zip(got[:5], want[:5]):
            assert np.array_equal(a, b)
