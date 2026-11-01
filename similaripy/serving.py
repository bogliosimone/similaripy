"""
Serving engine, the online half: `ItemKNN`, a prepared item-item model built
once from S (computed offline by the similarity functions, e.g. `sim.cosine`),
then the top-k items of one history (`recommend`) or of many
(`recommend_batch`), in Rust without the GIL. One import path:
`from similaripy.serving import ItemKNN`.

Everything is in the caller's item ids (`item_ids`, or S's column indices
without them); ids the model doesn't have are ignored. The internal layout
(items in popularity order, items with no row and no column dropped) never
shows.

Build: the Rust core reads S's arrays in place (cast here only when their
dtypes aren't int32/float32) and copies S once into the model, made canonical
on the way (rows sorted, duplicates summed, zeros dropped). Peak memory: S
plus the model.

Availability, three levels: `candidate_items` at build trims S (permanent
until rebuilt); `view(include=, exclude=)` gives a model sharing S with a
default mask (cheap to change: make a new view, swap the reference);
`include=` / `exclude=` per call. They combine with AND.
"""

from typing import Literal

import numpy as np
from scipy.sparse import sparray, spmatrix

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import SparseInput, _canonical_csr, _indptr_dtype

__all__ = ["ItemKNN"]

_INT64_MAX = np.iinfo(np.int64).max


def _check_ids(name, ids) -> np.ndarray:
    """The ids as a 1-d array, as given (no copy when already one); integer
    ids only (a float id is a bug, not an id), uint64 only within int64's
    range (no silent wrap)."""
    a = np.asarray(ids)
    if a.ndim > 1:
        raise ValueError(f"{name} must be 1-d, got shape {a.shape}")
    if a.size == 0:
        return np.empty(0, dtype=np.int64)
    if a.dtype.kind not in "iu":
        raise TypeError(
            f"{name} must be integer ids, got dtype {a.dtype} (map other ids to integers first, e.g. pd.factorize)"
        )
    # int(): exact; numpy 1.x compares uint64 with a Python int as float64 (2**63 == 2**63 - 1)
    if a.dtype.kind == "u" and int(a.max()) > _INT64_MAX:
        raise ValueError(f"{name} must fit in int64, got {a.max()}")
    return a


def _ids(name, ids) -> np.ndarray:
    """A private 1-d int64 copy of `_check_ids`."""
    return np.array(_check_ids(name, ids), dtype=np.int64)


def _opt_ids(name, ids):
    return None if ids is None else _ids(name, ids)


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


def _per_user(name, lists, n=None, convert=None):
    """One entry per user, flattened: (offsets int64, values, flags of the
    None entries or None). `n`: the expected user count."""
    if isinstance(lists, np.ndarray) or not isinstance(lists, (list, tuple)):
        shared = " (a rule for every user goes in view())" if name in ("include", "exclude") else ""
        raise TypeError(f"{name} must be a list with one entry per user{shared}, got {type(lists).__name__}")
    if n is not None and len(lists) != n:
        raise ValueError(f"{name} must have one entry per user ({n}), got {len(lists)}")
    convert = convert or (lambda x: _ids(name, x))
    arrays = [None if x is None else convert(x) for x in lists]
    none = np.fromiter((a is None for a in arrays), bool, len(arrays))
    lens = np.fromiter((0 if a is None else a.size for a in arrays), np.int64, len(arrays))
    parts = [a for a in arrays if a is not None and a.size]
    flat = np.concatenate(parts) if parts else convert([])
    return np.r_[0, np.cumsum(lens)].astype(np.int64), flat, (none if none.any() else None)


def _csr_arrays(S: SparseInput, num_threads):
    """S's size and CSR arrays as contiguous indptr (`_indptr_dtype`) / int32 /
    float32, copied only when the
    dtype or layout differs (the build reads them in place and never changes
    them). Other formats take the similarity functions' route
    (`_canonical_csr`: one copy, values cast before duplicates are summed), so
    the model doesn't depend on the input format."""
    n = S.shape[0]
    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError(f"S must be square (item x item), got shape {S.shape}")
    if S.format != "csr":
        S = _canonical_csr(S, np.float32, num_threads)
    nnz = int(S.indptr[-1])
    with np.errstate(over="ignore"):  # inf: rejected by the build's finite check
        data = np.ascontiguousarray(S.data[:nnz], dtype=np.float32)
    return (
        n,
        np.ascontiguousarray(S.indptr, dtype=_indptr_dtype(nnz)),
        np.ascontiguousarray(S.indices[:nnz], dtype=np.int32),
        data,
    )


class ItemKNN:
    """
    Item-based KNN recommendations from an item-item similarity matrix S.

    `score[j] = sum_i w_i * S[i, j]` over the history items `i`; the best `k`
    items with a positive score, best first (ties: the item with more entries
    in its column of S, then the lower column). Ids are yours (`item_ids`)
    everywhere; ids the model doesn't have are ignored. The model is
    immutable and safe to share across threads; each call releases the GIL.

    Args:
        S: square sparse item-item matrix, e.g. `sim.cosine(URM.T, k=100).T`
            (see "Orientation of S" in the Serving guide).
        item_ids: one unique int64 id per item: `item_ids[j]` is your id for
            column j of S (e.g. the `item_ids` of `interactions_to_urm`); None =
            the ids are S's column indices.
        candidate_items: only these items can ever be recommended: the other
            columns are dropped from the model (less memory and work; changing
            it means a rebuild). Histories still use every item. For items
            that are never available, `target_cols` when computing S is
            better: it keeps k available neighbours per item. None = all.
        block_size: column block size of every request, as the similarity
            functions': 'auto' (the library's choice), None (no blocking) or
            an int > 0. Results are the same; blocking is faster under
            concurrent requests, no blocking for a lone request on an idle
            machine.
        num_threads: threads for the build (0 = all).

    Raises:
        TypeError: on non-integer `item_ids` or `candidate_items`.
        ValueError: on a non-square S, or `item_ids` without one unique id
            per item.

    Note:
        S is read in place during the build (not copied first): don't
        change it from another thread meanwhile.
    """

    def __init__(
        self,
        S: sparray | spmatrix,
        *,
        item_ids: list[int] | np.ndarray | None = None,
        candidate_items: list[int] | np.ndarray | None = None,
        block_size: int | Literal["auto"] | None = "auto",
        num_threads: int = 0,
    ):
        n, indptr, indices, data = _csr_arrays(S, num_threads)
        ext = None
        if item_ids is not None:
            ext = _ids("item_ids", item_ids)
            if ext.shape != (n,):
                raise ValueError(f"item_ids must have one id per item ({n}), got shape {np.shape(item_ids)}")
        model = _core.ServeModel(
            indptr=indptr,
            indices=indices,
            data=data,
            n=n,
            ext=ext,
            block_size=block_size,
            num_threads=num_threads,
        )
        if candidate_items is not None:
            model = model.trimmed(_ids("candidate_items", candidate_items))
        self._model = model

    @classmethod
    def _wrap(cls, model):
        rec = cls.__new__(cls)
        rec._model = model
        return rec

    @property
    def item_ids(self) -> np.ndarray:
        """The id of each item (column of S), read-only: the `item_ids` given,
        or `arange(n)` without them."""
        ids = getattr(self, "_item_ids", None)
        if ids is None:
            ids = self._model.item_ids()
            if ids is None:
                ids = np.arange(self._model.n_cols, dtype=np.int64)
            ids.flags.writeable = False
            self._item_ids = ids
        return ids

    def view(
        self,
        *,
        include: list[int] | np.ndarray | None = None,
        exclude: list[int] | np.ndarray | None = None,
    ) -> "ItemKNN":
        """
        The same model with a default availability: only the `include` items
        (None = all) minus the `exclude` items. S is shared (no copy), only a
        bitset of one bit per item is new, so it is cheap to change often.
        Combined with the per-call `include` / `exclude` by AND. A view of a
        view narrows: the parent's rules AND the new ones (e.g. a country
        view, then a package view of it); to widen, start from the base model.

        Examples:
            To update availability in a running service, build the new view
            and swap the reference; requests in flight keep the model they
            started with:

            ```python
            state["model"] = base.view(include=available_ids)
            ```

        Args:
            include: only these item ids can be recommended (None = all).
            exclude: never recommend these item ids.

        Returns:
            model: a new `ItemKNN` sharing S, with these rules for every
                request.

        Raises:
            TypeError: on non-integer ids.
        """
        return self._wrap(self._model.view(_opt_ids("include", include), _opt_ids("exclude", exclude)))

    def recommend(
        self,
        history: list[int] | np.ndarray,
        *,
        weights: list[float] | np.ndarray | None = None,
        k: int = 10,
        exclude_seen: bool = True,
        include: list[int] | np.ndarray | None = None,
        exclude: list[int] | np.ndarray | None = None,
        sorted: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The top `k` items for one history.

        Args:
            history: item ids. Order doesn't matter; a duplicate adds its
                weight.
            weights: one finite weight per id (None = all 1.0).
            k: results wanted; fewer when fewer items score > 0.
            exclude_seen: never recommend a history item.
            include: only these items can be recommended (None = all).
            exclude: never recommend these items.
            sorted: best first; False returns the same items in an
                unspecified order (no final sort: a few % faster at k in the
                thousands, nothing at k ~ 100).

        Returns:
            ids: int64, your item ids, best first (unless `sorted=False`).
            scores: float32, one per id.

        Raises:
            TypeError: on non-integer ids.
            ValueError: on a negative `k` or weights without one per id.
        """
        k = _check_k(k)
        return self._model.recommend(
            _ids("history", history),
            _weights(weights),
            k=k,
            exclude_seen=exclude_seen,
            include=_opt_ids("include", include),
            exclude=_opt_ids("exclude", exclude),
            sorted=sorted,
        )

    def recommend_batch(
        self,
        histories: list[list[int] | np.ndarray],
        *,
        weights: list[list[float] | np.ndarray | None] | None = None,
        k: int = 10,
        exclude_seen: bool = True,
        include: list[list[int] | np.ndarray | None] | None = None,
        exclude: list[list[int] | np.ndarray | None] | None = None,
        sorted: bool = True,
        num_threads: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        `recommend` for many histories at once, in parallel: element u of the
        result equals `recommend(histories[u], weights=weights[u], ...)` bit
        for bit.

        Args:
            histories: a list with one array of item ids per user.
            weights: None, or a list with one array of weights per user (an
                entry may be None: all 1.0).
            include: None, or a list with one entry per user (an entry may be
                None). A rule for every user goes in `view()`.
            exclude: as `include`.
            k: as `recommend`, for every user.
            exclude_seen: as `recommend`, for every user.
            sorted: as `recommend`, for every user.
            num_threads: 0 = all cores (one shared pool), n = n threads.

        Returns:
            ids: one int64 array of item ids per user.
            scores: one float32 array of scores per user.

        Raises:
            TypeError: on non-integer ids, or rules that aren't a list with
                one entry per user.
            ValueError: on a negative `k`, lists without one entry per
                user, or weights without one per history id.
        """
        k = _check_k(k)
        offsets, ids, _ = _per_user("histories", histories)
        n = len(offsets) - 1
        w = None
        if weights is not None:
            w_offsets, w, none = _per_user("weights", weights, n, convert=_weights)
            sizes, given = np.diff(offsets), ~none if none is not None else np.ones(n, bool)
            if not np.array_equal(np.diff(w_offsets)[given], sizes[given]):
                raise ValueError("weights: one weight per history id expected")
            if not given.all():  # None entries: weight 1.0
                full = np.ones(ids.size, dtype=np.float32)
                full[np.repeat(given, sizes)] = w
                w = full
        inc = (None, None, None) if include is None else _per_user("include", include, n)
        exc = (None, None, None) if exclude is None else _per_user("exclude", exclude, n)
        out_offsets, out_ids, out_scores = self._model.batch(
            offsets,
            ids,
            w,
            inc[0],
            inc[1],
            inc[2],
            exc[0],
            exc[1],
            k=k,
            exclude_seen=exclude_seen,
            sorted=sorted,
            num_threads=num_threads,
        )
        if n == 0:  # np.split would give one empty piece
            return [], []
        cuts = out_offsets[1:-1]
        return np.split(out_ids, cuts), np.split(out_scores, cuts)

    def save(self, path, *, overwrite: bool = False) -> None:
        """
        Saves the model to a directory, for `ItemKNN.load` in this or any
        later 1.x version. A view can't be saved: save the base model and
        build views after `load`. Details: the Serving guide.

        Args:
            path: the directory: missing or empty.
            overwrite: replace a model saved there before (never other
                files).

        Raises:
            FileExistsError: on a non-empty directory without
                `overwrite=True`, or one that holds more than a saved model.
            ValueError: on a view.
        """
        from . import __version__

        self._model.save(path, f"similaripy {__version__}", overwrite)

    @classmethod
    def load(cls, path, *, mmap: bool = True) -> "ItemKNN":
        """
        A model saved by `save`, checked before use.

        Args:
            path: the directory written by `save`.
            mmap: map the files read-only: one copy in the OS page cache,
                shared by every process that loads them, and fast to load.
                False reads them into memory (e.g. on a network file system).

        Returns:
            model: the `ItemKNN`.

        Raises:
            FileNotFoundError: on a missing directory or file.
            ValueError: on a corrupted or unsupported file.

        Note:
            Load files you trust: the check makes a corrupted file an error,
            not a crash, but doesn't guard against resource exhaustion.
        """
        return cls._wrap(_core.ServeModel.load(path, mmap))

    def __reduce__(self):
        # Same-version pickles only (Q1): the model's arrays, re-checked on
        # load (as a saved model); block size as given, default mask kept.
        return (_from_parts, (type(self), *self._model.parts()))


def _from_parts(cls, *parts):
    return cls._wrap(_core.ServeModel.from_parts(*parts))
