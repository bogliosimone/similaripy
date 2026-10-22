"""
Serving engine: a prepared model built once from an item-item S, then top-K
recommendations per history (`recommend`) or per row of a user-item matrix
(`recommend_batch`), in Rust without the GIL.

Build: the canonical copy of S (`_canonical_csr`); items with neither a row
nor a column are dropped; the rest get internal ids in popularity order
(descending column count, as the kernel's `_popularity_layout`, ties to the
lower column), rows and columns alike, so a history item's internal id is both
its row and its column. The Rust model copies S once into that layout; the
canonical copy is freed after the build.
"""

import numpy as np
import scipy.sparse as sp

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import DEFAULT_BLOCK_SIZE, _canonical_csr

_INT32_MAX = np.iinfo(np.int32).max
_INT64_MAX = np.iinfo(np.int64).max


def _ids(name, ids) -> np.ndarray:
    """A private 1-d int64 copy; integer ids only (a float id is a bug, not
    an id), uint64 only within int64's range (no silent wrap)."""
    a = np.asarray(ids)
    if a.ndim > 1:
        raise ValueError(f"{name} must be 1-d, got shape {a.shape}")
    if a.size == 0:
        return np.empty(0, dtype=np.int64)
    if a.dtype.kind not in "iu":
        raise TypeError(f"{name} must be integer ids, got dtype {a.dtype}")
    if a.dtype.kind == "u" and a.max() > _INT64_MAX:
        raise ValueError(f"{name} must fit in int64, got {a.max()}")
    return np.array(a, dtype=np.int64)


def _weights(weights):
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim > 1:
        raise ValueError(f"weights must be 1-d, got shape {w.shape}")
    return np.array(w)


def _check_k(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}")
    return min(int(k), _INT64_MAX)  # clamped to the item count in Rust


def _check_mask(allowed):
    if allowed is not None and not isinstance(allowed, _core.ServeMask):
        raise TypeError(f"allowed must be a mask from Recommender.mask(ids), got {type(allowed).__name__}")
    return allowed


class Recommender:
    """
    Top-K items for a history from an item-item similarity matrix S.

    `score[j] = sum_i w_i * S[i, j]` over the history items `i`; the best `k`
    items with a positive score, best first (ties: the item with more entries
    in its column of S, then the lower column). The model is immutable and
    safe to share across threads; each call releases the GIL.

    Args:
        S: square sparse item-item matrix (e.g. `sim.cosine(URM.T, k=100)`).
        item_ids: one unique int64 id per item (column of S); None = the ids
            are S's column indices.
        allowed: only these item ids may be recommended (baked into the
            model, see `with_allowed`); None = all.
        num_threads: threads for the build (0 = all).
    """

    def __init__(self, S, item_ids=None, allowed=None, num_threads: int = 0, _block_size=DEFAULT_BLOCK_SIZE):
        S = _canonical_csr(S, np.float32, num_threads)
        n = S.shape[0]
        if S.shape[1] != n:
            raise ValueError(f"S must be square (item x item), got shape {S.shape}")
        if not np.isfinite(S.data).all():  # incl. float64 values past float32's range
            raise ValueError("S must be finite (in float32)")
        ext = None
        if item_ids is not None:
            ext = _ids("item_ids", item_ids)
            if ext.shape != (n,):
                raise ValueError(f"item_ids must have one id per item ({n}), got shape {np.shape(item_ids)}")
        col_counts = np.bincount(S.indices, minlength=n)
        keep = (np.diff(S.indptr) > 0) | (col_counts > 0)
        order = np.argsort(-col_counts, kind="stable")
        new_to_old = order[keep[order]].astype(np.int32)
        model = _core.ServeModel(
            indptr=S.indptr,
            indices=S.indices,
            data=S.data,
            n=n,
            new_to_old=new_to_old,
            ext=ext,
            block=_block_size,
            num_threads=num_threads,
        )
        del S
        if allowed is not None:
            model = model.with_allowed(model.mask(_ids("allowed", allowed), "skip"))
        self._model = model

    @classmethod
    def _wrap(cls, model):
        rec = cls.__new__(cls)
        rec._model = model
        return rec

    @property
    def n_items(self) -> int:
        """Items in the model (S's items minus those with no row and no column)."""
        return self._model.n_items

    def mask(self, ids, unknown: str = "skip"):
        """
        A reusable per-call availability mask: only `ids` may be recommended.
        Valid for this model and the models `with_allowed` derives from it.
        """
        return self._model.mask(_ids("ids", ids), unknown)

    def with_allowed(self, ids, unknown: str = "skip") -> "Recommender":
        """
        A new model that only recommends `ids` (their columns of S are kept,
        the others dropped: no per-call cost). Histories still use every
        item. Calling it on a model that already has `allowed` narrows it
        further; to widen again, call it on the unrestricted model.
        """
        return self._wrap(self._model.with_allowed(self.mask(ids, unknown)))

    def recommend(self, history, weights=None, k: int = 10, exclude_seen: bool = True, allowed=None, unknown="skip"):
        """
        The top `k` items for one history.

        Args:
            history: item ids. Order and duplicates don't matter (a duplicate
                adds its weight).
            weights: one finite weight per id (None = all 1.0).
            k: results wanted; fewer when fewer items score > 0.
            exclude_seen: never recommend a history item.
            allowed: a mask from `mask(ids)`.
            unknown: ids not in the model: 'skip' or 'raise' (ValueError).

        Returns:
            (ids int64, scores float32), best first.
        """
        k = _check_k(k)
        ids = _ids("history", history)
        weights = _weights(weights)
        allowed = _check_mask(allowed)
        return self._model.recommend(ids, weights, k=k, exclude_seen=exclude_seen, allowed=allowed, unknown=unknown)

    def recommend_batch(
        self, H, k: int = 10, exclude_seen: bool = True, allowed=None, num_threads: int = 0
    ) -> sp.csr_array:
        """
        The top `k` items for every row of H (users x items, columns = S's
        columns, values = weights), as `recommend` per row: row u equals
        `recommend(H[u].indices, H[u].data, ...)` bit for bit, ids given as
        S's column indices.

        Returns:
            csr_array (H rows x S columns), each row sorted best first. Past
            2^31 results the indptr is int64.
        """
        k = _check_k(k)
        allowed = _check_mask(allowed)
        n_cols = self._model.n_cols
        if H.ndim != 2 or H.shape[1] != n_cols:
            raise ValueError(f"H must have one column per item of S ({n_cols}), got shape {H.shape}")
        H = _canonical_csr(H, np.float32, num_threads)
        indptr, cols, scores = self._model.batch(
            H.indptr.astype(np.int64),
            H.indices,
            H.data,
            k=k,
            exclude_seen=exclude_seen,
            allowed=allowed,
            num_threads=num_threads,
        )
        if indptr[-1] <= _INT32_MAX:
            indptr = indptr.astype(np.int32)
        return sp.csr_array((scores, cols, indptr), shape=(H.shape[0], n_cols))

    def __reduce__(self):
        # Same-version pickles only (Q1): the arrays, rebuilt and re-checked on load.
        return (_from_parts, self._model.parts())


def _from_parts(indptr, indices, data, new_to_old, n_cols, ext):
    """A model from `ServeModel.parts()`: S back in its original ids, same layout."""
    rows = np.repeat(new_to_old, np.diff(indptr))
    S = _canonical_csr(sp.coo_array((data, (rows, new_to_old[indices])), shape=(n_cols, n_cols)), np.float32, 0)
    model = _core.ServeModel(
        indptr=S.indptr,
        indices=S.indices,
        data=S.data,
        n=n_cols,
        new_to_old=new_to_old,
        ext=ext,
        block=DEFAULT_BLOCK_SIZE,  # derived at load, never stored (Q1)
        num_threads=0,
    )
    return Recommender._wrap(model)
