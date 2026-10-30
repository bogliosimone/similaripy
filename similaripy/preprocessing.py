"""
From interaction logs to the matrices the similarities and the serving engine
take: `from_interactions` (logs -> URM + id maps + timestamps) and
`to_histories` (URM rows -> `recommend_batch` input). One import path:
`from similaripy.preprocessing import from_interactions`.

Ids are integers everywhere, with the same rules as `similaripy.serving`
(strings: map them at your edge, e.g. `pd.factorize`). Cold code: one numpy
path, the caller's arrays never change.
"""

from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from .serving import _ids

__all__ = ["from_interactions", "to_histories"]

_INT32_MAX = np.iinfo(np.int32).max
# Most stored pairs a URM holds (int32 CSR, until int64 offsets); a constant so tests can lower it.
_MAX_PAIRS = _INT32_MAX


def _codes(name: str, ids: np.ndarray, vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(vocabulary, code of each id, mask of the known ids or None = all)."""
    if vocab is None:
        vocab, codes = np.unique(ids, return_inverse=True)
        return vocab, codes.ravel(), None  # ravel: numpy 2.0.x returned the input's shape
    vocab = _ids(name, vocab)
    order = np.argsort(vocab, kind="stable")
    sorted_vocab = vocab[order]
    if sorted_vocab.size and (sorted_vocab[1:] == sorted_vocab[:-1]).any():
        raise ValueError(f"{name} must be unique")
    pos = np.minimum(np.searchsorted(sorted_vocab, ids), max(vocab.size - 1, 0))
    known = sorted_vocab[pos] == ids if vocab.size else np.zeros(ids.size, dtype=bool)
    return vocab, order[pos] if vocab.size else pos, known


def _timestamps(timestamps, n: int) -> np.ndarray:
    """float64, datetime64 of any unit as seconds."""
    t = np.asarray(timestamps)
    if t.dtype.kind == "M":
        t = (t - np.datetime64(0, "s")) / np.timedelta64(1, "s")  # epoch seconds; NaT -> NaN, caught below
    elif t.dtype.kind in "iuf":
        t = t.astype(np.float64)
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

    Memory: a few int64/float64 arrays of one entry per interaction (one sort).
    """
    users, items = _ids("users", users), _ids("items", items)
    if users.shape != items.shape:
        raise ValueError(f"users and items must have the same length, got {users.size} and {items.size}")
    n = users.size
    if values is None:
        v = np.ones(n, dtype=np.float64)
    else:
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"values must have one value per interaction, got shape {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("values must be finite")
    t = None if timestamps is None else _timestamps(timestamps, n)
    if duplicates not in ("sum", "first", "last"):
        raise ValueError(f"duplicates must be 'sum', 'first' or 'last', got {duplicates!r}")

    user_ids, rows, known_users = _codes("user_ids", users, user_ids)
    item_ids, cols, known_items = _codes("item_ids", items, item_ids)
    for name, ids in (("user_ids", user_ids), ("item_ids", item_ids)):
        if ids.size > _INT32_MAX:
            raise ValueError(f"at most {_INT32_MAX} {name} (int32 indices), got {ids.size}")
    if known_users is not None or known_items is not None:
        keep = np.ones(n, dtype=bool)
        for known in (known_users, known_items):
            if known is not None:
                keep &= known
        if n and not keep.any():
            raise ValueError("no interaction left: none of the ids are in user_ids / item_ids")
        rows, cols, v = rows[keep], cols[keep], v[keep]
        t = None if t is None else t[keep]

    # One sort by (row, col). "sum" needs no order within a pair: the default
    # sort (deterministic, ~4x faster than stable on 20M int64 keys); first/last
    # sort stably by input order, or by time (ties: input order).
    # ponytail: lexsort ~1 s per M events; a Rust helper if large logs need it.
    key = rows.astype(np.int64) * item_ids.size + cols
    if duplicates == "sum":
        order = np.argsort(key)
    elif t is not None:
        order = np.lexsort((t, key))
    else:
        order = np.argsort(key, kind="stable")
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if key.size else np.empty(0, dtype=np.intp)
    v = v[order]
    if duplicates == "sum":
        data = np.add.reduceat(v, starts) if starts.size else v
    elif duplicates == "first":
        data = v[starts]
    else:
        data = v[np.r_[starts[1:], key.size] - 1] if starts.size else v
    data = data.astype(np.float32)
    stored = data != 0  # after the cast: what canonical CSR keeps
    if int(stored.sum()) > _MAX_PAIRS:
        raise ValueError(f"at most {_MAX_PAIRS} distinct (user, item) pairs (int32 indices), got {int(stored.sum())}")
    key = key[starts][stored]
    pair_rows = key // max(item_ids.size, 1)
    indices = (key - pair_rows * item_ids.size).astype(np.int32)
    indptr = np.zeros(user_ids.size + 1, dtype=np.int32)
    np.cumsum(np.bincount(pair_rows, minlength=user_ids.size), out=indptr[1:])
    shape = (user_ids.size, item_ids.size)
    URM = sp.csr_array((data[stored], indices, indptr), shape=shape)
    T = None
    if t is not None:
        latest = np.maximum.reduceat(t[order], starts) if starts.size else t
        T = sp.csr_array((latest[stored], indices.copy(), indptr.copy()), shape=shape)
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
