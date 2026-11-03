"""
Other libraries' exact sparse top-k item-item similarity, for the `crosslib` workload (D78).

One `Library` per library: the similarities it maps to, `prepare` (the call,
built once) and `to_csr(output, n_items)` (its output as an items x items CSR, for the check).
Every library starts from the same users x items CSR (float32, int32
indices); its own prep (transpose, normalize, dtype casts, LensKit's Dataset)
is part of the call, as a user would write it. Imports are lazy: a missing
library errors only when asked for.

`prepare(URM, similarity, k, threads, entry)` returns (kwargs, call): kwargs
= what the library's entry point gets (recorded as the case params), entry =
that function or class, resolved by name before the run (the tests spy on
it), call() -> (output, extras).

Mapping (verified vs a dense float64 reference, notes cross-lib/mapcheck.py):
  - sparse_dot_topn / sp_matmul_rs: plain top-n matmul; cosine = rows
    L2-normalized by the caller. Both reject scipy's csr_array (TypeError):
    wrapped in csr_matrix. n_threads 0 means 1, so threads is always explicit.
  - implicit: dot_product needs float64 input (float32 raises); output float64.
    The macOS wheel has no OpenMP (single-threaded whatever num_threads).
  - scikit-learn: cosine distance, brute force; rows always hold k entries
    (zero-similarity fillers dropped here); float32 1 - d loses precision on
    small cosines (atol 1e-6 in the check). n_jobs 0 raises.
  - LensKit: implicit feedback = binary cosine, self never in its neighbours
    (save_nbrs = k - 1, the check drops the reference's diagonal), min_sim
    1e-6 drops ~0 scores. Threads: once per process. No CSR constructor: the
    Dataset is built from the COO arrays, timed (also reported as dataset_s).
  - scipy: the "no library" baseline (RecPack's ItemKNN): full SpGEMM, then
    top-k per row. Single-threaded: threads must be 1.
"""

import importlib
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import scipy.sparse as sp

# Check tolerance vs similaripy: two float32 libraries summing ~1e5-term rows differ by up to ~6e-4
# (ML-32M: similaripy's cosine diagonal is off by up to 5e-4 vs float64, D78 step 1). A mapping mistake
# (k, normalization, self) shows at 1e-2 and more. The parity scripts keep 1e-4.
RTOL = 1e-3

# Shared dependencies recorded in every report (None when not installed)
DEPENDENCIES = ("numpy", "scipy", "scikit-learn", "torch", "pyarrow")


@dataclass(frozen=True)
class Library:
    module: str  # import name
    package: str  # distribution name (its version goes in the report)
    entries: dict  # similarity -> "module:attr" of the call that gets the kwargs
    prepare: object
    to_csr: object
    self_in_topk: bool = True
    atol: float | None = None  # check tolerance when not the parity default
    threshold: float = 0.0  # the library's own cut-off (check near-ties)

    @property
    def similarities(self):
        return tuple(self.entries)

    def entry(self, similarity):
        module, attr = self.entries[similarity].split(":")
        return getattr(importlib.import_module(module), attr)


def installed_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _csr(out, n):
    return sp.csr_matrix(out)


def _topn_prepare(URM, similarity, k, threads, entry):
    """sparse_dot_topn and sp_matmul_rs: one top-n matmul of the items x users rows with their transpose."""
    from sklearn.preprocessing import normalize

    kwargs = {"top_n": k, "n_threads": threads}
    if threads > 1 and entry.__module__.startswith("sp_matmul_rs"):
        import sp_matmul_rs

        if not sp_matmul_rs._rust_parallel_enabled:  # it would only warn and run on 1 thread
            raise RuntimeError("sp_matmul_rs was built without rayon: n_threads would be ignored")

    def call():
        X = sp.csr_matrix(URM.T)  # the transpose copy; csr_array raises
        if similarity != "dot_product":
            X = normalize(X)
        return entry(X, X.T, **kwargs), {}

    return kwargs, call


def _implicit_prepare(URM, similarity, k, threads, entry):
    kwargs = {"K": k, "num_threads": threads}
    dtype = np.float64 if similarity == "dot_product" else URM.dtype

    def call():
        model = entry(**kwargs)
        model.fit(sp.csr_matrix(URM, dtype=dtype), show_progress=False)  # csr_array: a warning per call
        return model.similarity, {}

    return kwargs, call


def _sklearn_prepare(URM, similarity, k, threads, entry):
    kwargs = {"n_neighbors": k, "algorithm": "brute", "metric": "cosine", "n_jobs": threads}

    def call():
        X = URM.T
        return entry(**kwargs).fit(X).kneighbors(X), {}

    return kwargs, call


def _sklearn_to_csr(out, n):
    dist, ids = out
    k = ids.shape[1]
    S = sp.csr_matrix((1 - dist.ravel(), ids.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    S.eliminate_zeros()  # the zero-similarity fillers of rows with < k neighbours (distance exactly 1)
    return S


def _lenskit_prepare(URM, similarity, k, threads, entry):
    from lenskit._accel import thread_count
    from lenskit.config import ParallelSettings
    from lenskit.data import DatasetBuilder
    from lenskit.parallel import get_parallel_config, init_threading

    init_threading(ParallelSettings(num_threads=threads))  # a no-op (warning) once set
    if get_parallel_config().num_threads != threads:
        raise ValueError(f"LensKit threads are set once per process: already {get_parallel_config().num_threads}")
    if thread_count() != threads:  # a failed pool init only logs a warning
        raise RuntimeError(f"LensKit's Rayon pool has {thread_count()} threads, not {threads}")
    kwargs = {"max_nbrs": k, "save_nbrs": k - 1, "feedback": "implicit"}

    def call():
        start = time.perf_counter()
        coo = URM.tocoo()
        builder = DatasetBuilder()
        builder.add_interactions(
            "interaction", {"user_id": coo.row, "item_id": coo.col}, entities=["user", "item"], missing="insert"
        )
        data = builder.build()
        dataset_s = time.perf_counter() - start
        scorer = entry(**kwargs)
        scorer.train(data)
        return scorer, {"dataset_s": dataset_s}

    return kwargs, call


def _lenskit_to_csr(scorer, n):
    """The neighbour matrix with LensKit's item numbering mapped back to the URM's n columns."""
    S = scorer.sim_matrix.to_scipy().tocoo()
    ids = np.asarray(scorer.items.ids())
    return sp.csr_matrix((S.data, (ids[S.row], ids[S.col])), shape=(n, n))


def _scipy_prepare(URM, similarity, k, threads, entry):
    if threads != 1:
        raise ValueError("scipy is single-threaded: run it with --threads 1")
    kwargs = {} if similarity == "dot_product" else {"dense_output": False}

    def call():
        X = sp.csr_matrix(URM.T)
        full = entry(X, X.T, **kwargs) if similarity == "dot_product" else entry(X, **kwargs)
        return _top_k_rows(sp.csr_matrix(full), k), {}

    return kwargs, call


def _top_k_rows(S, k):
    """The k largest entries of each CSR row (np.argpartition per longer row), as RecPack's get_top_K_values."""
    lengths = np.diff(S.indptr)
    keep = np.ones(S.nnz, dtype=bool)
    for r in np.flatnonzero(lengths > k):
        lo, hi = S.indptr[r], S.indptr[r + 1]
        drop = np.argpartition(S.data[lo:hi], hi - lo - k)[: hi - lo - k]
        keep[lo + drop] = False
    indptr = np.concatenate(([0], np.cumsum(np.minimum(lengths, k))))
    return sp.csr_matrix((S.data[keep], S.indices[keep], indptr), shape=S.shape)


COSINES = ("cosine", "cosine_binary")
ALL = ("cosine", "dot_product", "cosine_binary")

LIBRARIES = {
    "sparse_dot_topn": Library(
        "sparse_dot_topn",
        "sparse-dot-topn",
        {s: "sparse_dot_topn:sp_matmul_topn" for s in ALL},
        _topn_prepare,
        _csr,
    ),
    "sp_matmul_rs": Library(
        "sp_matmul_rs",
        "sp-matmul-rs",
        {s: "sp_matmul_rs:sp_matmul_topn" for s in ALL},
        _topn_prepare,
        _csr,
    ),
    "implicit": Library(
        "implicit",
        "implicit",
        {
            "cosine": "implicit.nearest_neighbours:CosineRecommender",
            "dot_product": "implicit.nearest_neighbours:ItemItemRecommender",
            "cosine_binary": "implicit.nearest_neighbours:CosineRecommender",
        },
        _implicit_prepare,
        _csr,
    ),
    "sklearn": Library(
        "sklearn",
        "scikit-learn",
        {s: "sklearn.neighbors:NearestNeighbors" for s in COSINES},
        _sklearn_prepare,
        _sklearn_to_csr,
        atol=1e-6,
    ),
    "lenskit": Library(
        "lenskit",
        "lenskit",
        {"cosine_binary": "lenskit.knn:ItemKNNScorer"},
        _lenskit_prepare,
        _lenskit_to_csr,
        self_in_topk=False,
        threshold=1e-6,  # ItemKNNScorer's default min_sim
    ),
    "scipy": Library(
        "scipy",
        "scipy",
        {
            "cosine": "sklearn.metrics.pairwise:cosine_similarity",
            "dot_product": "operator:matmul",
            "cosine_binary": "sklearn.metrics.pairwise:cosine_similarity",
        },
        _scipy_prepare,
        _csr,
    ),
}
