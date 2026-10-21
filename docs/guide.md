# 📌 Getting Started

SimilariPy provides **high-performance KNN similarity functions** in Python, optimized for sparse matrices.

Primarily designed for Recommender Systems and Information Retrieval (IR) tasks, but can be applied to other domains as well.

The package also includes a set of normalization functions useful for pre-processing data before the similarity computation.

## Similarity Functions

SimilariPy provides a range of high-performance similarity functions for sparse matrices.  
All functions run in a multi-threaded Rust core, GIL-free, on sparse CSR matrices.

### Core 

- **Dot Product** – Simple raw inner product between vectors.
- **Cosine** – Normalized dot product based on L2 norm.
- **Asymmetric Cosine** – Skewed cosine similarity using an `alpha` parameter.
- **Jaccard**, **Dice**, **Tversky** – Set-based generalized similarities.

### Graph-Based

- **P3α** – Graph-based similarity computed through random walk propagation with exponentiation.
- **RP3β** – Similar to P3α but includes popularity penalization using a `beta` parameter.

### Advanced 

- **S-Plus** – A hybrid model combining Tversky and Cosine components, with RP3Beta-style depopularization and full control over weights and smoothing. 

## Normalization Functions

SimilariPy provides a suite of normalization functions for sparse matrix pre-processing.  
All functions return a new canonical CSR matrix (multi-threaded Rust core); the input never changes.

- **L1, L2** – Applies row- or column-wise normalization.
- **TF-IDF** – Computes TF-IDF weighting with customizable term-frequency and IDF modes.
- **BM25** – Applies classic BM25 weighting used in information retrieval.
- **BM25+** – Variant of BM25 with additive smoothing for low-frequency terms.

## Example

Here’s a minimal example to get you up and running with SimilariPy:


```python
import similaripy as sim
import scipy.sparse as sps

# Create a random User-Rating Matrix (URM)
urm = sps.random_array((1000, 2000), density=0.025)

# Normalize the URM using BM25
urm = sim.normalization.bm25(urm)

# Train an item-item cosine similarity model
similarity_matrix = sim.cosine(urm.T, k=50, verbose=True)  # verbose: progress bar

# Compute recommendations for user 1, 14, 8
# filtering out already-seen items
recommendations = sim.dot_product(
    urm,
    similarity_matrix.T,
    k=100,
    target_rows=[1, 14, 8],
    filter_cols=urm,
)
```

Tips:

- `urm.T` is used to switch from user-item to item-user when training the similarity model.
- You can use `bm25plus`, `tfidf`, or `normalize` for different pre-processing strategies.
- All operations are multithreaded and scale with available CPU cores.

## Signed Input

Input values can be negative, e.g. mean-centered ratings (adjusted cosine) or
explicit dislikes. A score is evidence of similarity: only **positive** scores
(`>= threshold`) are results; negative scores and scores that cancel to exactly
`0` are dropped, so a row with only negative scores comes back empty.

```python
import numpy as np

# Adjusted cosine: center each user's ratings, then item-item cosine
urm = sps.csr_array(urm, dtype=np.float32)  # a copy; integer ratings can't be centered in place
counts = np.diff(urm.indptr)
sums = np.ravel(urm.sum(axis=1))
means = np.divide(sums, counts, out=np.zeros(urm.shape[0]), where=counts > 0)
urm.data -= np.repeat(means, counts)
similarity_matrix = sim.cosine(urm.T, k=50, threshold=1e-6)
```

- **Dissimilar items** are not returned. To rank them, negate `matrix2`:
  `sim.cosine(m, -m.T)` keeps the most negative scores, as positives.
- **`threshold > 0`** is a good idea: sums that should cancel can leave a tiny
  float32 residue (e.g. `1e-8`) instead of an exact `0`. On float data in any
  similarity; for `cosine`, `asymmetric_cosine` and `rp3beta` also on integer
  data (they run as a dot product of pre-scaled inputs, whose rounding doesn't
  cancel exactly).
- **Selectors:** `filter_cols` / `target_cols` matrices select the non-zero
  entries. Pass the raw (or binarized) interaction matrix, not a mean-centered
  one: a rating equal to the user's mean centers to `0` and would not be
  filtered.
- **Range:** values must be finite in float32 (`|x| < 3.4e38`). Sums that
  overflow float32 give `inf` (kept) or NaN (`inf - inf`, dropped) scores
  (`ValueError` where noted in the [migration notes](migration.md#api)): rescale very
  large values first.
- What each function accepts on signed input is listed in
  [Similarities](similarity.md#signed-input) and
  [Normalizations](normalization.md#signed-input).

Coming from 0.7.0? See [Migration from 0.7.0](migration.md) for every change.

## In-depth Introduction

Looking for a full end-to-end example?

Check the MovieLens notebook:
- [MovieLens 32M Item-Item Recommender (GitHub)](https://github.com/bogliosimone/similaripy/blob/master/notebooks/movielens32m_item_item_recommender.ipynb)
