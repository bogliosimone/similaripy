"""
From interaction logs to the matrices the similarities and the serving engine
take: `from_interactions` (logs -> URM + id maps + timestamps) and
`to_histories` (URM rows -> `recommend_batch` input). One import path:
`from similaripy.preprocessing import from_interactions`.

Ids are integers everywhere, with the same rules as `similaripy.serving`
(strings: map them at your edge, e.g. `pd.factorize`). The caller's arrays
never change. The grouping (a counting sort by user, then each user's row in
parallel) and `rank_decay`'s per-row rank run in the Rust core.
"""

from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from .serving import _check_ids, _ids

__all__ = ["from_interactions", "to_histories"]

_INT32_MAX = np.iinfo(np.int32).max
# Most stored pairs a URM holds (int32 CSR, until int64 offsets); a constant so tests can lower it.
_MAX_PAIRS = _INT32_MAX


def _codes(name: str, ids: np.ndarray, vocab) -> tuple[np.ndarray, np.ndarray]:
    """(vocabulary, int32 code of each id, -1 = not in the given vocabulary).
    Dense-ish ids (range <= 2x the events): a lookup table, O(n); else a sort."""
    if vocab is not None:
        vocab = _ids(name, vocab)
        if vocab.size > _INT32_MAX:
            raise ValueError(f"at most {_INT32_MAX} {name} (int32 indices), got {vocab.size}")
        if np.unique(vocab).size != vocab.size:
            raise ValueError(f"{name} must be unique")
    if ids.size == 0:
        return (np.empty(0, dtype=np.int64) if vocab is None else vocab), np.empty(0, dtype=np.int32)
    lo, hi = int(ids.min()), int(ids.max())
    if hi - lo < 2 * ids.size + 1024:
        offset = ids - ids.dtype.type(lo)  # in the ids' dtype (int32 ids: 4 B/event)
        if vocab is None:
            present = np.zeros(hi - lo + 1, dtype=bool)
            present[offset] = True
            vocab = np.flatnonzero(present) + lo
            table = np.cumsum(present, dtype=np.int32) - 1
        else:
            table = np.full(hi - lo + 1, -1, dtype=np.int32)
            inside = (vocab >= lo) & (vocab <= hi)
            table[vocab[inside] - lo] = np.flatnonzero(inside)
        return vocab, table[offset]
    if vocab is None:
        vocab, codes = np.unique(ids, return_inverse=True)
        vocab = vocab.astype(np.int64)
        if vocab.size > _INT32_MAX:
            raise ValueError(f"at most {_INT32_MAX} {name} (int32 indices), got {vocab.size}")
        return vocab, codes.ravel().astype(np.int32)  # ravel: numpy 2.0.x returned the input's shape
    order = np.argsort(vocab)
    sorted_vocab = vocab[order]
    pos = np.minimum(np.searchsorted(sorted_vocab, ids), max(vocab.size - 1, 0))
    if vocab.size == 0:
        return vocab, np.full(ids.size, -1, dtype=np.int32)
    return vocab, np.where(sorted_vocab[pos] == ids, order[pos], -1).astype(np.int32)


def _timestamps(timestamps, n: int) -> np.ndarray:
    """float64, datetime64 of any unit as seconds."""
    t = np.asarray(timestamps)
    if t.dtype.kind == "M":
        t = (t - np.datetime64(0, "s")) / np.timedelta64(1, "s")  # epoch seconds; NaT -> NaN, caught below
    elif t.dtype.kind in "iuf":
        t = np.ascontiguousarray(t, dtype=np.float64)
    else:
        raise TypeError(
            f"timestamps must be numbers or datetime64, got dtype {t.dtype} "
            "(timezone-aware pandas datetimes: `.dt.tz_convert(None)` first)"
        )
    if t.shape != (n,):
        raise ValueError(f"timestamps must have one value per interaction, got shape {t.shape}")
    if not np.isfinite(t).all():
        raise ValueError("timestamps must be finite (no NaN / NaT)")
    return t


def from_interactions(
    users: ArrayLike,
    items: ArrayLike,
    values: ArrayLike | None = None,
    *,
    timestamps: ArrayLike | None = None,
    duplicates: Literal["sum", "first", "last"] = "sum",
    user_ids: ArrayLike | None = None,
    item_ids: ArrayLike | None = None,
) -> tuple[sp.csr_array, np.ndarray, np.ndarray, sp.csr_array | None]:
    """
    A user-item matrix from interaction logs, with the id maps.

    Args:
        users, items: one integer id per interaction (any ints: gaps and
            unaligned ids are fine).
        values: one finite value per interaction (None = 1.0 each), e.g. a
            rating or an event weight.
        timestamps: one per interaction (numbers, or datetime64 of any unit,
            stored as seconds), or None.
        duplicates: what a repeated (user, item) pair keeps: "sum" adds the
            values (counts), "first" / "last" keep the earliest / latest
            event's value, by timestamp (ties: input order), or by input
            order without timestamps.
        user_ids, item_ids: a fixed vocabulary (unique ints) instead of the
            ids seen, e.g. to give matrices built from different slices the
            same columns; interactions with other ids are dropped.

    Returns:
        (URM, user_ids, item_ids, T): `URM` a float32 `csr_array` (users x
        items), row `u` = `user_ids[u]`, column `j` = `item_ids[j]` (int64,
        sorted, or the vocabulary as given); pass `item_ids` to
        `ItemKNN(S, item_ids=item_ids)`. `T` a float64 `csr_array` with
        URM's exact pattern, the latest timestamp of each pair (whatever
        `duplicates`), for `time_decay` / `rank_decay`; None without
        timestamps. Pairs whose value is 0 in float32 are not stored (in
        either matrix).

    Memory: about 30 bytes per interaction above the inputs (int32 codes,
    then per-user buckets: item, value, timestamp), plus the output.
    """
    users, items = _check_ids("users", users), _check_ids("items", items)  # no int64 copy: ~8 B/event each
    if users.shape != items.shape:
        raise ValueError(f"users and items must have the same length, got {users.size} and {items.size}")
    n = users.size
    v = None
    if values is not None:
        v = np.ascontiguousarray(values, dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"values must have one value per interaction, got shape {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("values must be finite")
    t = None if timestamps is None else _timestamps(timestamps, n)
    if duplicates not in ("sum", "first", "last"):
        raise ValueError(f"duplicates must be 'sum', 'first' or 'last', got {duplicates!r}")

    user_ids, rows = _codes("user_ids", users, user_ids)
    item_ids, cols = _codes("item_ids", items, item_ids)
    if n and ((rows < 0) | (cols < 0)).all():
        raise ValueError("no interaction left: none of the ids are in user_ids / item_ids")
    indptr, indices, data, latest = _core.group_interactions(
        rows=rows,
        cols=cols,
        values=v,
        t=t,
        n_rows=user_ids.size,
        n_cols=item_ids.size,
        duplicates=duplicates,
    )
    if data.size > _MAX_PAIRS:
        raise ValueError(f"at most {_MAX_PAIRS} distinct (user, item) pairs (int32 indices), got {data.size}")
    shape = (user_ids.size, item_ids.size)
    URM = sp.csr_array((data, indices, indptr), shape=shape)
    T = None if latest is None else sp.csr_array((latest, indices.copy(), indptr.copy()), shape=shape)
    return URM, user_ids, item_ids, T


def to_histories(X, item_ids: ArrayLike | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Each row of X as a history for `ItemKNN.recommend_batch`.

    Args:
        X: user-item matrix (e.g. the URM, or a `rank_decay` of it to keep
            each user's latest items).
        item_ids: your id of each column (the `item_ids` of
            `from_interactions`); None = column indices (for `ItemKNN(S)`
            without `item_ids`).

    Returns:
        (histories, weights): two lists with one array per row of X (int64
        ids, float32 weights = X's values), row u = `user_ids[u]`; views into
        one copy of X's arrays. Use as
        `model.recommend_batch(histories, weights=weights)`, in chunks of
        rows for many users.
    """
    X = sp.csr_array(X, dtype=np.float32, copy=True)  # sum_duplicates must not touch the caller's
    X.sum_duplicates()
    if item_ids is None:
        ids = X.indices.astype(np.int64)
    else:
        item_ids = _ids("item_ids", item_ids)
        n_cols = X.shape[1]  # ty: ignore[index-out-of-bounds]  (scipy-stubs: csr_array(X) as 1-d)
        if item_ids.shape != (n_cols,):
            raise ValueError(f"item_ids must have one id per column ({n_cols}), got shape {item_ids.shape}")
        ids = item_ids[X.indices]
    cuts = X.indptr[1:-1]
    return np.split(ids, cuts), np.split(X.data, cuts)
