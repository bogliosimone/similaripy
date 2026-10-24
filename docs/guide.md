# 📌 Quickstart

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

To serve the model online (one history per call, GIL-free, saved and
memory-mapped by every worker), see [Serving](serving.md):

```python
from similaripy.serving import ItemKNN

model = ItemKNN(similarity_matrix.T)   # the same scores as dot_product above
ids, scores = model.recommend([3, 17, 42], k=10)
```

Tips:

- `urm.T` is used to switch from user-item to item-user to train an item-item similarity model.
- You can use `bm25plus`, `tfidf`, or `normalize` for different pre-processing strategies.
- All operations are multithreaded and scale with available CPU cores.

## In-depth Introduction

Looking for a full end-to-end example?

Check the MovieLens notebook:
- [MovieLens 32M Item-Item Recommender (GitHub)](https://github.com/bogliosimone/similaripy/blob/master/notebooks/movielens32m_item_item_recommender.ipynb)
