# 📊 Benchmarks

How SimilariPy compares with other Python libraries that compute exact sparse
top-k item-item similarity. Timings and peak memory on the same datasets come
with the published benchmark runs; this page starts with what each library
does.

## Features

Each column is a library at the version shown, as installed from PyPI. ✓ marks
a cell checked by running the library; every other cell comes from its source
code or documentation (see [Sources](#sources)).

| | SimilariPy | sparse_dot_topn | sp_matmul_rs | implicit | scikit-learn | LensKit |
|---|---|---|---|---|---|---|
| Version | 1.0 | 1.2.0 | 0.2.1 | 0.7.3 | 1.9.1 | 2026.4.0 |
| Item-item call | `cosine(URM.T, k=k)` | `sp_matmul_topn(X, X.T, top_n=k)` | `sp_matmul_topn(X, X.T, top_n=k)` | `CosineRecommender(K=k).fit(URM)` | `NearestNeighbors(metric="cosine", algorithm="brute")` | `ItemKNNScorer(max_nbrs=k)` |
| Similarities | 9: dot product, cosine, asymmetric cosine, Tversky, Jaccard, Dice, P3α, RP3β, S-Plus | sparse product only (cosine: L2-normalize the rows first) | sparse product only (cosine: L2-normalize the rows first) | dot product, cosine, TF-IDF, BM25 | cosine (as a distance) | cosine (explicit: mean-centred; implicit: binary) |
| Shrink | stabilized, bayesian, additive | no | no | no | no | no |
| Weighting | BM25, BM25+, TF-IDF, l1 / l2 / max, binarize, time and rank decay | no | no | BM25, TF-IDF, l2 | separate (`normalize`, `TfidfTransformer`) | built in: mean-centring, l2 |
| Row / column selectors | `target_rows`, `target_cols`, `filter_cols` | no | no | at recommend time | query rows only | at scoring time |
| Negative values | yes ([rules](signed-input.md)) | dropped when `top_n` is shorter than the row ✓ | kept ✓ | kept ✓ | kept ✓ | implicit: binarized, signs lost ✓; explicit: negative similarities dropped |
| Threshold | `threshold` | `threshold` | `threshold` | no | `radius_neighbors` only | `min_sim` (default 1e-6) |
| Output | `csr_array` or `coo_array`, float32 ✓ | `csr_matrix`, input dtype ✓ | `csr_matrix`, input dtype ✓ | `csr_matrix`, float64 ✓ | dense arrays of distances and indices, always k per row ✓ | `SparseRowArray` (pyarrow), float32 ✓ |
| Item in its own top-k | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes with `kneighbors(X)` ✓ | no ✓ |
| Threads, macOS arm64 wheel | yes (default: all cores) | yes, `n_threads` (default 1) ✓ | no Python 3.14 wheel; built from source: yes, `n_threads` (default 1) ✓ | no (wheel built without OpenMP) ✓ | `n_jobs` over row chunks (sparse cosine skips the multi-threaded kernel) ✓ | yes |
| Threads, Linux x86_64 wheel | yes (default: all cores) | yes, `n_threads` (default 1) | yes, `n_threads` (default 1) | yes, needs the system's OpenMP runtime | as macOS | yes |
| Free-threaded wheels | yes | no | no | no | yes | yes |
| Recommend API | `serving.ItemKNN` | no | no | `recommend`, `similar_items` | no | scoring pipelines |
| Saved model | `save` / `load` (memory-mapped), pickle | no | no | `save` / `load`, pickle | pickle | pickle |
| License | MIT | Apache-2.0 | Apache-2.0 | MIT | BSD-3-Clause | MIT |

Notes:

- **SimilariPy 0.7.0** (C++ and OpenMP) has the same similarities, shrinks,
  weightings, selectors and threshold; its output defaults to `coo_array`. It
  has no serving or preprocessing module, and it does not support negative
  values ([Migration from 0.7.0](migration.md)).
- **scikit-learn's `NearestNeighbors`** returns distances (`1 − cosine`); rows
  with fewer than k neighbours are filled with zero-similarity entries.
- **LensKit** builds its own `Dataset` from the interactions; it has no
  constructor from a CSR matrix.
- **Full product baseline:** computing the whole item x item product with
  scipy and keeping the top k per row (as RecPack's ItemKNN does) appears in the
  timings as a baseline, not as a library.

## Sources

Read at the versions above (installed packages, Python 3.14), plus the release
files on PyPI for wheels and licenses.

- **sparse_dot_topn 1.2.0** ([PyPI](https://pypi.org/project/sparse-dot-topn/1.2.0/), [source](https://github.com/ing-bank/sparse_dot_topn/tree/v1.2.0)): `api.py` (`sp_matmul_topn`: input types, `threshold`, `n_threads`, output); the macOS wheel bundles `libomp`.
- **sp_matmul_rs 0.2.1** ([PyPI](https://pypi.org/project/sp-matmul-rs/0.2.1/), [source](https://github.com/bergvca/sp_matmul_rs)): `api.py` (`sp_matmul_topn`); wheels up to Python 3.13, rayon enabled by default (README).
- **implicit 0.7.3** ([PyPI](https://pypi.org/project/implicit/0.7.3/), [source](https://github.com/benfred/implicit/tree/v0.7.3)): `implicit/nearest_neighbours.py` (recommenders, weightings, `recommend`, `save` / `load`); `pyproject.toml` (wheel builds: no free-threaded builds, Linux wheels not repaired, so OpenMP comes from the system); the macOS extension links no OpenMP runtime.
- **scikit-learn 1.9.1** ([PyPI](https://pypi.org/project/scikit-learn/1.9.1/), [source](https://github.com/scikit-learn/scikit-learn/tree/1.9.1)): `sklearn/neighbors/_base.py` (`kneighbors`: brute force, the multi-threaded kernel's sparse limits, self handling), `sklearn/metrics/pairwise.py` (`cosine_distances`).
- **LensKit 2026.4.0** ([PyPI](https://pypi.org/project/lenskit/2026.4.0/), [source](https://github.com/lenskit/lkpy)): `lenskit/knn/item.py` (`ItemKNNScorer`: centring, normalization, `min_sim`, `save_nbrs`, scoring), `lenskit/parallel/config.py` (rayon pool).
- **SimilariPy**: this documentation ([Similarities](similarity.md), [Normalizations](normalization.md), [Serving](serving.md), [Signed input](signed-input.md)); 0.7.0 from its `v0.7.0` tag.
