"""
From interaction logs to the matrices the similarities and the serving engine
take: `interactions_to_urm` (log -> URM + id maps), `interactions_to_histories`
and `urm_to_histories` (log or URM rows -> `recommend_batch` input), and
`time_decay` / `rank_decay` (one decay weight per event, offline and for a
live history). One import path: `from similaripy.preprocessing import ...`.

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

__all__ = ["interactions_to_histories", "interactions_to_urm", "rank_decay", "time_decay", "urm_to_histories"]

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


def interactions_to_urm(
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
    The user-item matrix (URM) from one entry per event, with the id maps.

    Examples:
        ```python
        URM, user_ids, item_ids = interactions_to_urm(log.user, log.item)
        # with a time decay: each event weighted by its own age, then summed
        w = time_decay(log.ts, reference=cutoff, scale=30 * 86400)
        URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, w)
        ```

    Args:
        users: one integer user id per event; gaps and unaligned ids are fine.
        items: one integer item id per event.
        values: one finite value per event, e.g. a rating, an event weight
            or a decay weight; None = 1.0 each.
        timestamps: one per event (numbers or datetime64), only to order
            "first" / "last" (ties: input order); None = input order.
        duplicates: what a repeated (user, item) pair keeps: "sum" adds the
            values (counts), "first" / "last" keep the earliest / latest
            event's value.
        user_ids: a fixed user vocabulary (unique ints) instead of the ids
            seen; events with other ids are dropped.
        item_ids: a fixed item vocabulary, e.g. to give matrices built from
            different slices the same columns; events with other ids are
            dropped.

    Returns:
        URM: float32 CSR matrix (users x items), duplicates merged, pairs
            whose value is 0 in float32 not stored.
        user_ids: int64, `user_ids[u]` is your id of row `u` (sorted, or
            the vocabulary as given).
        item_ids: int64, `item_ids[j]` is your id of column `j`; pass it to
            `ItemKNN(S, item_ids=item_ids)`.

    Raises:
        TypeError: on non-integer ids or timestamps of another type.
        ValueError: on lengths that don't match, non-finite values or
            timestamps, a non-unique vocabulary, no event left with a
            vocabulary, or more than 2^31 - 1 distinct pairs.

    Note:
        Memory: about 20 bytes per event above the inputs (+ 8 with
        timestamps), plus the output; built in the Rust core.
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


def interactions_to_histories(
    users: ArrayLike,
    items: ArrayLike,
    weights: ArrayLike | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Each user's events as a history for `ItemKNN.recommend_batch`, straight
    from the log, e.g. a holdout set: no matrix and no id mapping (ids the
    model doesn't have are ignored by `recommend`).

    Repeated items stay (`recommend` sums their weights: the same scores as
    through a URM); events with weight 0 are dropped, so a `"step"` decay
    also caps the length.

    Examples:
        ```python
        user_ids, histories, weights = interactions_to_histories(test.user, test.item)
        ids, scores = model.recommend_batch(histories, weights=weights, k=20)  # element u: user_ids[u]
        ```

    Args:
        users: one integer user id per event.
        items: one integer item id per event.
        weights: one finite weight per event, e.g. a `time_decay` /
            `rank_decay`; None = 1.0 each.

    Returns:
        user_ids: int64, sorted: `user_ids[u]` is the user of element `u`.
        histories: one int64 array of item ids per user, in input order.
        weights: one float32 array of weights per user.

    Raises:
        TypeError: on non-integer ids.
        ValueError: on lengths that don't match or non-finite weights.
    """
    users, items = _check_ids("users", users), _ids("items", items)
    if users.shape != items.shape:
        raise ValueError(f"users and items must have the same length, got {users.size} and {items.size}")
    if weights is None:
        w = np.ones(users.size, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != users.shape:
            raise ValueError(f"weights must have one value per event, got shape {w.shape}")
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite")
        keep = w != 0
        if not keep.all():
            users, items, w = users[keep], items[keep], w[keep]
    user_ids, codes = _codes("users", users, None)
    if user_ids.size == 0:
        return user_ids, [], []
    order = np.argsort(codes, kind="stable")  # by user, input order within
    cuts = np.cumsum(np.bincount(codes, minlength=user_ids.size))[:-1]
    return user_ids, np.split(items[order], cuts), np.split(w[order], cuts)


def urm_to_histories(URM, item_ids: ArrayLike | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Each row of a URM as a history for `ItemKNN.recommend_batch`, e.g. every
    training user, or a benchmark dataset given as a matrix.

    Examples:
        ```python
        histories, weights = urm_to_histories(URM, item_ids)
        ids, scores = model.recommend_batch(histories, weights=weights, k=20)  # element u: row u
        ```

    Args:
        URM: user-item sparse matrix; its values become the weights.
        item_ids: your id of each column (the `item_ids` of
            `interactions_to_urm`); None = column indices (for `ItemKNN(S)`
            without `item_ids`).

    Returns:
        histories: one int64 array of item ids per row.
        weights: one float32 array of the row's values per row.

    Raises:
        TypeError: on non-integer `item_ids`.
        ValueError: on `item_ids` without one id per column.
    """
    URM = sp.csr_array(URM, dtype=np.float32, copy=True)  # sum_duplicates must not touch the caller's
    URM.sum_duplicates()
    if item_ids is None:
        ids = URM.indices.astype(np.int64)
    else:
        item_ids = _ids("item_ids", item_ids)
        n_cols = URM.shape[1]  # ty: ignore[index-out-of-bounds]  (scipy-stubs: csr_array(URM) as 1-d)
        if item_ids.shape != (n_cols,):
            raise ValueError(f"item_ids must have one id per column ({n_cols}), got shape {item_ids.shape}")
        ids = item_ids[URM.indices]
    if URM.shape[0] == 0:
        return [], []
    cuts = URM.indptr[1:-1]
    return np.split(ids, cuts), np.split(URM.data, cuts)


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
    timestamps: ArrayLike,
    *,
    reference: float | str | np.datetime64 | datetime.datetime,
    scale: float | np.timedelta64 | datetime.timedelta,
    shape: Literal["exponential", "power", "linear", "step"] = "exponential",
    exponent: float | None = None,
    users: ArrayLike | None = None,
) -> np.ndarray:
    """
    The decay weight of each event from its age in time: age = reference -
    timestamp. Multiply the weights into the values of `interactions_to_urm`
    (offline), or pass them as `weights` to `recommend` (online): one
    formula for both.

    Examples:
        ```python
        # offline: the same reference for every user
        w = time_decay(log.ts, reference=cutoff, scale=30 * 86400)  # half-life 30 days
        URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, w)

        # online: one live history, from the user's latest event
        w = time_decay(history_ts, reference=history_ts.max(), scale=30 * 86400)
        ids, scores = model.recommend(history, weights=w)
        ```

    Args:
        timestamps: one per event: numbers, or datetime64 (read as seconds).
        reference: where age 0 is. A time, the same for every event (a
            number in the timestamps' units, or a datetime for datetime
            timestamps), or "user": each user's latest event (needs
            `users`).
        scale: in the timestamps' units, or a duration (`np.timedelta64`,
            `datetime.timedelta`, `pd.Timedelta`); `np.inf` = no decay.
        shape: "exponential" (scale = half-life), "power" (half-life, longer
            tail), "linear" (0 from age = scale) or "step" (a window); see
            the Preprocessing guide for the formulas.
        exponent: the power-law p (> 0), for shape "power" only; None = 1.
        users: one integer id per event, for `reference="user"` only.

    Returns:
        weights: float32, one per event, in input order; a negative age (an
            event after the reference, e.g. clock skew online) counts as 0.

    Raises:
        TypeError: on a datetime reference with numeric timestamps, or the
            other way round.
        ValueError: on non-finite timestamps, a bad shape, scale or
            exponent, or `users` without `reference="user"` (or the other
            way round).
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
    users: ArrayLike | None = None,
    *,
    scale: float,
    timestamps: ArrayLike | None = None,
    shape: Literal["exponential", "power", "linear", "step"] = "exponential",
    exponent: float | None = None,
) -> np.ndarray:
    """
    The decay weight of each event from its recency rank in its user's
    history: 0 = the latest, 1 = the one before, ... (events, not distinct
    items). Multiply the weights into the values of `interactions_to_urm`
    (offline), or pass them as `weights` to `recommend` (online).

    Examples:
        ```python
        # offline: each user's 200 latest events
        w = rank_decay(log.user, scale=200, shape="step")
        URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, w)

        # online: one user's live history, so no users; the timestamps order it
        w = rank_decay(timestamps=history_ts, scale=200, shape="step")
        ids, scores = model.recommend(history, weights=w)
        ```

    Args:
        users: one integer user id per event; None = all events are one
            history (then `timestamps` is needed).
        scale: in events; `np.inf` = no decay.
        timestamps: one per event, to order each user's events (ties: the
            later row is the more recent); None = input order, oldest first.
        shape: as in `time_decay`, with ages in events.
        exponent: the power-law p (> 0), for shape "power" only; None = 1.

    Returns:
        weights: float32, one per event, in input order.

    Raises:
        TypeError: on non-integer ids or a duration as `scale`.
        ValueError: on neither `users` nor `timestamps`, lengths that don't
            match, non-finite timestamps, a bad shape, scale or exponent.
    """
    _check_shape(shape, exponent)
    if isinstance(scale, np.timedelta64) or hasattr(scale, "total_seconds"):
        raise TypeError("rank_decay's scale is a number of events, not a duration (time_decay takes durations)")
    scale = _scale(scale)
    if users is None and timestamps is None:
        raise ValueError("rank_decay needs users and/or timestamps")
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
