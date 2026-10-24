"""
Serving engine, the online half: `ItemKNN`, a prepared item-item model built
once from S (computed offline by the similarity functions, e.g. `sim.cosine`),
then top-K recommendations per history (`recommend`) or per row of a user-item
matrix (`recommend_batch`), in Rust without the GIL. One import path:
`from similaripy.serving import ItemKNN`.

Build: the Rust core reads S's arrays in place (cast here only when their
dtypes aren't int32/float32) and copies S once into the model, made canonical
on the way (rows sorted, duplicates summed, zeros dropped). Items with neither
a row nor a column are dropped; the rest get internal ids in popularity order
(descending column count, as the kernel's `_popularity_layout`, ties to the
lower column), rows and columns alike, so a history item's internal id is both
its row and its column. Peak memory: S plus the model.

Availability, three levels: `candidate_items` at build trims S (permanent
until rebuilt); `view(allowed_items=...)` gives a model sharing S with a
default mask (cheap to change: make a new view, swap the reference); a mask
per call (`allowed=`). The two masks combine with AND.
"""

import numpy as np
import scipy.sparse as sp

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import _canonical_csr

__all__ = ["ItemKNN"]

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
    # int(): exact; numpy 1.x compares uint64 with a Python int as float64 (2**63 == 2**63 - 1)
    if a.dtype.kind == "u" and int(a.max()) > _INT64_MAX:
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
        raise TypeError(f"allowed must be a mask from ItemKNN.mask(ids), got {type(allowed).__name__}")
    return allowed


def _csr_arrays(S, num_threads):
    """S's CSR arrays as contiguous int32 / float32, copied only when the
    dtype or layout differs (the build reads them in place and never changes
    them). Other formats take the similarity functions' route
    (`_canonical_csr`: one copy, values cast before duplicates are summed), so
    the model doesn't depend on the input format."""
    if S.nnz > _INT32_MAX:
        raise ValueError(f"S must have at most {_INT32_MAX} non-zeros")
    if S.format != "csr":
        S = _canonical_csr(S, np.float32, num_threads)
    nnz = int(S.indptr[-1])
    with np.errstate(over="ignore"):  # inf: rejected by the build's finite check
        data = np.ascontiguousarray(S.data[:nnz], dtype=np.float32)
    return (
        np.ascontiguousarray(S.indptr, dtype=np.int32),
        np.ascontiguousarray(S.indices[:nnz], dtype=np.int32),
        data,
    )


class ItemKNN:
    """
    Item-based KNN recommendations from an item-item similarity matrix S.

    `score[j] = sum_i w_i * S[i, j]` over the history items `i`; the best `k`
    items with a positive score, best first (ties: the item with more entries
    in its column of S, then the lower column). The model is immutable and
    safe to share across threads; each call releases the GIL.

    Args:
        S: square sparse item-item matrix (e.g. `sim.cosine(URM.T, k=100)`).
        item_ids: one unique int64 id per item (column of S); None = the ids
            are S's column indices.
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

    S is read in place during the build (not copied first): don't change it
    from another thread meanwhile.
    """

    def __init__(self, S, item_ids=None, candidate_items=None, block_size="auto", num_threads: int = 0):
        n = S.shape[0]
        if S.ndim != 2 or S.shape[1] != n:
            raise ValueError(f"S must be square (item x item), got shape {S.shape}")
        indptr, indices, data = _csr_arrays(S, num_threads)
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
            model = model.trimmed(model.mask(_ids("candidate_items", candidate_items), "skip"))
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
        A reusable availability mask allowing only `ids`, for `recommend(...,
        allowed=mask)` or `recommend_batch`. Valid for every model of this
        build (its views included).
        """
        return self._model.mask(_ids("ids", ids), unknown)

    def view(self, allowed_items=None, unknown: str = "skip") -> "ItemKNN":
        """
        The same model with `allowed_items` as its default availability: S is
        shared (no copy), only a bitset of one bit per item is new, so it is
        cheap to change often. Combined with a per-call `allowed` mask by AND.
        A view of a view replaces the default (None = everything allowed).

        To update availability in a running service, build the new view and
        swap the reference; requests in flight keep the model they started
        with::

            state["model"] = state["model"].view(allowed_items=available_ids)
        """
        mask = None if allowed_items is None else self.mask(allowed_items, unknown)
        return self._wrap(self._model.view(mask))

    def recommend(self, history, weights=None, k: int = 10, exclude_seen: bool = True, allowed=None, unknown="skip"):
        """
        The top `k` items for one history.

        Args:
            history: item ids. Order and duplicates don't matter (a duplicate
                adds its weight).
            weights: one finite weight per id (None = all 1.0).
            k: results wanted; fewer when fewer items score > 0.
            exclude_seen: never recommend a history item.
            allowed: a mask from `mask(ids)`, ANDed with the view's default.
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
        columns, values = weights), as `recommend` per row. H is made
        canonical first (duplicates summed, zeros dropped): an explicit zero
        or entries that cancel to zero are not part of the history, so
        `exclude_seen` doesn't exclude them. On canonical H, row u equals
        `recommend(item_ids[H[u].indices], H[u].data, ...)` bit for bit
        (`H[u].indices` without `item_ids`).

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

    def save(self, path, overwrite: bool = False) -> None:
        """
        Saves the model to the directory `path`: one `.npy` file per array
        (`numpy.load` opens them) and `meta.json`, for `ItemKNN.load` in this
        or any later 1.x version. Saved as is: the view's availability mask,
        `block_size` as given.

        `path` must be missing or empty. `overwrite=True` replaces a saved
        model there (a directory holding only a saved model's files, not a
        symlink) with a new directory renamed into place, so processes that
        mapped the old files keep them (Linux, macOS; on Windows the replace
        fails with PermissionError while any process has the model loaded with
        `mmap=True`, and the old model stays). Between the two renames the
        directory doesn't exist: a `load` at that moment fails, two savers
        must not overwrite the same path at once, and nothing else may write
        into the directory meanwhile (its files are replaced). For a live service, save each
        version to a new directory and switch what the workers load (a symlink
        or config you flip); never change a saved model's files in place, a
        process mapping them would crash.

        The files: `indptr`, `indices`, `data` (S in internal ids, rows sorted),
        `new_to_old` (internal id -> S's column), optional `item_ids` (id per
        column) and `allowed` (uint64 bitset), each a 1-d little-endian `.npy`
        with numpy's header layout (as `np.save` writes it); `meta.json` holds
        the format version, `value_dtype`, sizes and `block_size`.
        """
        from . import __version__

        self._model.save(path, f"similaripy {__version__}", overwrite)

    @classmethod
    def load(cls, path, mmap: bool = True) -> "ItemKNN":
        """
        A model saved by `save`, checked before use.

        `mmap=True` maps the arrays read-only: they stay in the OS page cache,
        shared by every process that loads the same files (one copy of S for
        all workers of a server), and loading is fast. `mmap=False` reads them
        into memory (e.g. on a network file system). Masks of the saved model
        don't apply to the loaded one; pickling a mapped model copies its
        arrays.

        The check makes a corrupted file an error, never an out-of-bounds read,
        but it doesn't guard against resource exhaustion (e.g. a `meta.json`
        asking for billions of items): load files you trust.
        """
        return cls._wrap(_core.ServeModel.load(path, mmap))

    def __reduce__(self):
        # Same-version pickles only (Q1): the model's arrays, re-checked on
        # load (as a saved model); block size as given, default mask kept.
        return (_from_parts, (type(self), *self._model.parts()))


def _from_parts(cls, *parts):
    return cls._wrap(_core.ServeModel.from_parts(*parts))
