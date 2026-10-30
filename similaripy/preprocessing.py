"""
From interaction logs to the matrices the similarities and the serving engine
take: `from_interactions` (logs -> URM + id maps), `time_decay` /
`rank_decay` (one decay weight per event, offline and for a live history)
and `to_histories` (URM rows -> `recommend_batch` input). One import path:
`from similaripy.preprocessing import from_interactions`.

Ids are integers everywhere, with the same rules as `similaripy.serving`
(strings: map them at your edge, e.g. `pd.factorize`). The caller's arrays
never change. The grouping (a counting sort by user, then each user's row in
parallel) runs in the Rust core.
"""

import datetime
from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from .serving import _check_ids, _ids

__all__ = ["from_interactions", "rank_decay", "time_decay", "to_histories"]

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
        # in the ids' dtype when the span fits it (int32 ids: 4 B/event), else int64
        dtype = ids.dtype if hi - lo <= np.iinfo(ids.dtype).max else np.dtype(np.int64)
        offset = ids.astype(dtype, copy=False) - dtype.type(lo)
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
) -> tuple[sp.csr_array, np.ndarray, np.ndarray]:
    """
    A user-item matrix from interaction logs, with the id maps.

    Args:
        users, items: one integer id per interaction (any ints: gaps and
            unaligned ids are fine).
        values: one finite value per interaction (None = 1.0 each), e.g. a
            rating, an event weight, or a decay: `values *
            time_decay(ts, ...)` / `rank_decay(users, ...)` (each
            event weighted by its own age, then summed: as `recommend` sums
            a live history).
        timestamps: one per interaction (numbers or datetime64), only to
            order "first" / "last"; None = input order.
        duplicates: what a repeated (user, item) pair keeps: "sum" adds the
            values (counts), "first" / "last" keep the earliest / latest
            event's value, by timestamp (ties: input order), or by input
            order without timestamps.
        user_ids, item_ids: a fixed vocabulary (unique ints) instead of the
            ids seen, e.g. to give matrices built from different slices the
            same columns; interactions with other ids are dropped.

    Returns:
        (URM, user_ids, item_ids): `URM` a float32 canonical `csr_array`
        (users x items), row `u` = `user_ids[u]`, column `j` = `item_ids[j]`
        (int64, sorted, or the vocabulary as given); pass `item_ids` to
        `ItemKNN(S, item_ids=item_ids)`. Pairs whose value is 0 in float32
        are not stored.

    Memory: about 20 bytes per interaction above the inputs (int32 codes,
    then per-user buckets: item, value; + 8 with timestamps), plus the output.
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
    if duplicates == "sum":
        t = None  # sums need no order
    if duplicates not in ("sum", "first", "last"):
        raise ValueError(f"duplicates must be 'sum', 'first' or 'last', got {duplicates!r}")

    user_ids, rows = _codes("user_ids", users, user_ids)
    item_ids, cols = _codes("item_ids", items, item_ids)
    if n and ((rows < 0) | (cols < 0)).all():
        raise ValueError("no interaction left: none of the ids are in user_ids / item_ids")
    indptr, indices, data = _core.group_interactions(
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
    return sp.csr_array((data, indices, indptr), shape=(user_ids.size, item_ids.size)), user_ids, item_ids


def to_histories(X, item_ids: ArrayLike | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Each row of X as a history for `ItemKNN.recommend_batch`.

    Args:
        X: user-item matrix (e.g. the URM).
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


# ---- Time and rank decay: one weight per event ----

_SHAPES = ("exponential", "power", "linear", "step")


def _scale(scale) -> float:
    """A positive number (np.inf = no decay); durations as seconds."""
    if isinstance(scale, np.timedelta64):
        scale = scale / np.timedelta64(1, "s")
    elif hasattr(scale, "total_seconds"):  # datetime.timedelta, pd.Timedelta
        scale = scale.total_seconds()
    scale = float(scale)
    if not scale > 0:  # also NaN
        raise ValueError(f"scale must be > 0 (np.inf = no decay), got {scale}")
    return scale


def _check_shape(shape: str, exponent) -> None:
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {_SHAPES}, got {shape!r}")
    if exponent is not None and shape != "power":
        raise ValueError(f"exponent applies to shape='power' only, got shape={shape!r}")


def _user_codes(users, n: int) -> tuple[np.ndarray, int]:
    ids = _check_ids("users", users)
    if ids.shape != (n,):
        raise ValueError(f"users must have one id per event ({n}), got shape {ids.shape}")
    vocab, codes = _codes("users", ids, None)
    return codes, vocab.size


def _weights(age: np.ndarray, shape: str, scale: float, exponent) -> np.ndarray:
    return _core.decay_weights(age=age, shape=shape, scale=scale, exponent=exponent)


def time_decay(
    timestamps,
    *,
    reference,
    users=None,
    shape: str = "exponential",
    scale,
    exponent: float | None = None,
) -> np.ndarray:
    """
    The decay weight of each event from its age in time: age = reference -
    timestamp. Multiply the weights into `from_interactions`' values (each
    event weighted by its own age, repeats summed), or pass them to
    `recommend` for a live history: one formula offline and online.

    Shapes (weight 1 at age 0): "exponential" 0.5^(age / scale) (scale =
    half-life); "power" (1 + (2^(1/p) - 1) age / scale)^-p (also 0.5 at age
    = scale, a longer tail for a smaller p = `exponent`, default 1);
    "linear" max(0, 1 - age / scale) (0 from age = scale); "step" 1 if age
    < scale else 0 (a window).

    Example::

        w = time_decay(df.ts, reference=cutoff, scale=30 * 86400)
        URM, user_ids, item_ids = from_interactions(df.user, df.item, values * w)
        # online, the same formula
        ids, scores = model.recommend(history, weights=time_decay(history_ts, reference=now, scale=30 * 86400))

    Args:
        timestamps: one per event: numbers, or datetime64 (read as seconds).
        reference: where age 0 is: a time, the same for every event (e.g.
            the training cutoff; online: now): a number in the timestamps'
            units, or a datetime (datetime64, `datetime`, `pd.Timestamp`)
            for datetime timestamps; or
            "user": each user's latest event (needs `users`; online, the
            same is `reference=history_ts.max()`).
        users: one integer id per event, for `reference="user"` only.
        shape: "exponential", "power", "linear" or "step".
        scale: in the timestamps' units (seconds for datetimes), or a
            duration (`np.timedelta64`, `datetime.timedelta`,
            `pd.Timedelta`); required. `np.inf` = no decay (every weight 1).
        exponent: the power-law p (> 0), shape "power" only.

    Returns:
        float32 weights, one per event, in input order. A negative age (an
        event after the reference, e.g. clock skew online) counts as 0.

    Raises:
        ValueError: on NaN / inf timestamps, a bad shape, scale or exponent,
            `reference="user"` without `users` (or `users` with a fixed
            reference).
        TypeError: on a datetime reference with numeric timestamps, or the
            other way round.
    """
    _check_shape(shape, exponent)
    scale = _scale(scale)
    dates = np.asarray(timestamps).dtype.kind == "M"
    t = _timestamps(timestamps, np.size(timestamps))
    if isinstance(reference, str):
        if reference != "user":
            raise ValueError(f"reference must be a time or 'user', got {reference!r}")
        if users is None:
            raise ValueError("reference='user' needs users (one id per event)")
        codes, n_users = _user_codes(users, t.size)
        age = _core.user_ages(users=codes, n_users=n_users, t=t, kind="latest")
    else:
        if users is not None:
            raise ValueError("users applies to reference='user' only")
        if isinstance(reference, datetime.datetime):  # also pd.Timestamp
            if reference.tzinfo is not None:
                raise TypeError("reference must be timezone-naive (as the timestamps), e.g. `.tz_convert(None)`")
            reference = np.datetime64(reference)
        ref = np.asarray(reference)
        if (ref.dtype.kind == "M") != dates:  # e.g. ms timestamps vs a date: every age wrong, silently
            raise TypeError("reference and timestamps must both be datetimes or both be numbers (same units)")
        ref = float((ref - np.datetime64(0, "s")) / np.timedelta64(1, "s")) if dates else float(ref)
        age = ref - t
    return _weights(age, shape, scale, exponent)


def rank_decay(
    users=None,
    *,
    timestamps=None,
    shape: str = "exponential",
    scale: float,
    exponent: float | None = None,
) -> np.ndarray:
    """
    The decay weight of each event from its recency rank within its user's
    events: 0 = the latest, 1 = the one before, ... (events, not distinct
    items). Use as `time_decay`'s weights; `scale` is in events.

    Shapes (weight 1 at age 0): "exponential" 0.5^(age / scale) (scale =
    half-life); "power" (1 + (2^(1/p) - 1) age / scale)^-p (also 0.5 at age
    = scale, a longer tail for a smaller p = `exponent`, default 1);
    "linear" max(0, 1 - age / scale) (0 from age = scale); "step" 1 if age
    < scale else 0 (a window).

    Example::

        w = rank_decay(df.user, shape="step", scale=200)  # each user's 200 latest events
        URM, user_ids, item_ids = from_interactions(df.user, df.item, w)

    Online, a history in arrival order has the same weights on every
    request: compute them once and slice::

        table = rank_decay(np.zeros(1000, dtype=int), shape="step", scale=200)
        history = history[-1000:]  # at most the table's length
        ids, scores = model.recommend(history, weights=table[-len(history):])

    Args:
        users: one integer id per event (offline); None = all events are
            one history (online, with `timestamps`).
        timestamps: one per event, to order them (equal timestamps: the
            later row is the more recent); None = input order, oldest first.
        shape: "exponential", "power", "linear" or "step".
        scale: in events (> 0, `np.inf` = no decay); required.
        exponent: the power-law p (> 0), shape "power" only.

    Returns:
        float32 weights, one per event, in input order.

    Raises:
        ValueError: on neither `users` nor `timestamps`, lengths that don't
            match, NaN / inf timestamps, a bad shape, scale or exponent.
        TypeError: on a duration as `scale`.
    """
    _check_shape(shape, exponent)
    if isinstance(scale, np.timedelta64) or hasattr(scale, "total_seconds"):
        raise TypeError("rank_decay's scale is a number of events, not a duration (time_decay takes durations)")
    scale = _scale(scale)
    if users is None and timestamps is None:
        raise ValueError("rank_decay needs users and/or timestamps (a history in arrival order: see the table example)")
    t = None
    if timestamps is not None:
        t = _timestamps(timestamps, np.size(timestamps))
    if users is None:
        assert t is not None
        codes, n_users = np.zeros(t.size, dtype=np.int32), 1
    else:
        codes, n_users = _user_codes(users, np.asarray(users).size if t is None else t.size)
    age = _core.user_ages(users=codes, n_users=n_users, t=t, kind="rank")
    return _weights(age, shape, scale, exponent)
