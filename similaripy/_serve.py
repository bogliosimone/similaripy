"""
Serving engine, step 4.1 exploration (private until 4.2 names and documents
it): a prepared model built once from an item-item S, then top-K
recommendations for one history per call, in Rust without the GIL.

Build: the canonical copy of S (`_canonical_csr`), then the symmetric
popularity layout: items renumbered by descending column count (as the
kernel's `_popularity_layout`, but rows too, so internal row id = column id),
copied once into the Rust model. Columns without entries sort last, so the
accumulator stops at the last one with entries (`n_active`).
"""

import numpy as np

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import DEFAULT_BLOCK_SIZE, _canonical_csr


class Recommender:
    """
    Top-K items for a history from an item-item similarity matrix S.

    `score[j] = sum_i w_i * S[i, j]` over the history items `i`; the best `k`
    with a positive score, best first (ties: the item with more neighbours in
    S, then the lower column).

    Args:
        S: square sparse item-item matrix (e.g. `sim.cosine(URM.T, k=100)`).
        item_ids: external int64 id per column of S (unique); None = ids are
            S's column indices.
        num_threads: threads for the build (0 = all).
    """

    def __init__(self, S, item_ids=None, num_threads: int = 0):
        S = _canonical_csr(S, np.float32, num_threads)
        n = S.shape[0]
        if S.shape[1] != n:
            raise ValueError(f"S must be square (item x item), got shape {S.shape}")
        if item_ids is None:
            ext = np.arange(n, dtype=np.int64)
        else:
            ext = np.ascontiguousarray(item_ids, dtype=np.int64)
            if ext.shape != (n,):
                raise ValueError(f"item_ids must have one id per item ({n}), got shape {ext.shape}")
        new_to_old = np.argsort(-np.bincount(S.indices, minlength=n), kind="stable").astype(np.int32)
        old_to_new = np.empty(n, dtype=np.int32)
        old_to_new[new_to_old] = np.arange(n, dtype=np.int32)
        P = S[new_to_old]  # rows in the new order (a copy; S is ours)
        indptr = P.indptr.astype(np.int32)
        indices = P.indices.astype(np.int32)
        data = P.data.astype(np.float32)
        _core.remap_sort_rows(indptr=indptr, indices=indices, data=data, forward=old_to_new, num_threads=num_threads)
        self._model = _core.ServeModel(indptr=indptr, indices=indices, data=data, new_to_old=new_to_old, ext=ext)
        self._default = _core.ServeVariant(lookup="dense" if item_ids is None else "binary", block=DEFAULT_BLOCK_SIZE)

    @property
    def n_items(self) -> int:
        return self._model.n_items

    def mask(self, ids):
        """A reusable per-call availability mask: only `ids` may be recommended."""
        return self._model.mask(np.ascontiguousarray(ids, dtype=np.int64), self._default)

    def recommend(self, history, weights=None, k: int = 10, exclude_seen: bool = True, allowed=None, _variant=None):
        """
        The top `k` items for `history` (external ids; unknown ids skipped,
        order and duplicates don't matter: duplicates add their weights).

        Args:
            history: item ids.
            weights: one finite weight per id (None = all 1.0).
            k: results wanted; fewer when fewer items score > 0.
            exclude_seen: never recommend a history item.
            allowed: a mask from `self.mask(ids)`.

        Returns:
            (ids int64, scores float32), best first.
        """
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        ids = np.ascontiguousarray(history, dtype=np.int64).ravel()
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype=np.float32).ravel()
        return self._model.recommend(
            ids,
            weights,
            k=int(k),
            exclude_seen=exclude_seen,
            allowed=allowed,
            variant=self._default if _variant is None else _variant,
        )
