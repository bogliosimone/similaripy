# 📌 Quickstart

SimilariPy is a high-performance Python package, built in Rust, to **compute KNN similarity matrices** from sparse data and **serve them with low latency**.

It also includes normalization functions to pre-process the data.

Primarily designed for Recommender Systems and Information Retrieval (IR) tasks, but can be applied to other domains as well.

## Similarity Functions

SimilariPy provides a range of high-performance similarity functions for sparse matrices.  
All functions run in a multi-threaded Rust core, GIL-free, on sparse CSR matrices.

### Core 

- **Dot Product**: Simple raw inner product between vectors.
- **Cosine**, **Asymmetric Cosine**: Normalized dot product, symmetric or skewed.
- **Jaccard**, **Dice**, **Tversky**: Set-based generalized similarities.

### Graph-Based

- **P3α**: Graph-based similarity computed through random walk propagation with exponentiation.
- **RP3β**: Similar to P3α but includes a popularity penalization term.

### Advanced

- **S-Plus**: A hybrid model combining Tversky and Cosine components, with RP3Beta-style depopularization and full control over weights and smoothing.

## Serving

The serving engine turns a similarity matrix into recommendations, in your own item ids.  
All requests run in the Rust core, GIL-free: one user per call, or a batch of users on multiple threads.

- **Offline**: recommendations for a batch of users, e.g. to evaluate a model or precompute results.
- **Online**: one user per call, in tens to hundreds of microseconds.
- **Availability rules**: include or exclude items for all requests, per group, or per request.

## Normalization and Weighting

Re-weight your data before the similarity computation: normalization functions on sparse matrices (multi-threaded Rust core, GIL-free, a new CSR matrix), and time decay of interactions.

- **L1, L2**: Applies row- or column-wise normalization.
- **TF-IDF**: Computes TF-IDF weighting with customizable term-frequency and IDF modes.
- **BM25**: Applies classic BM25 weighting used in information retrieval.
- **BM25+**: Variant of BM25 with additive smoothing for low-frequency terms.
- **Time and rank decay**: weight each interaction by its age or recency, the same formula offline and for a live history ([Preprocessing](preprocessing.md)).

## Example

Here’s a minimal example to get you up and running with SimilariPy:


```python
import similaripy as sim
import scipy.sparse as sps
from similaripy.serving import ItemKNN

# Create a random User-Rating Matrix (URM)
urm = sps.random_array((1000, 2000), density=0.025, format="csr")

# Normalize the URM using BM25
urm = sim.normalization.bm25(urm)

# Train an item-item cosine similarity model
similarity_matrix = sim.cosine(urm.T, k=50, verbose=True)  # verbose: progress bar

# Build the serving model
model = ItemKNN(similarity_matrix.T)

# Offline: the top 10 items to evaluate a few test users, already-seen items excluded
# (exclude_seen=False to keep them, e.g. music or grocery)
test_users = [1, 14, 8]
histories = [urm[[u]].indices for u in test_users]
weights = [urm[[u]].data for u in test_users]
ids, scores = model.recommend_batch(histories, weights=weights, k=10)

# Online: a new user, from the items they interacted with (no retraining)
ids, scores = model.recommend([3, 17, 256], k=10)
```

From an interaction log (one entry per event, your integer ids), with a time decay:

```python
import numpy as np
from similaripy.preprocessing import from_interactions, time_decay, to_histories

rng = np.random.default_rng(0)
users = rng.integers(0, 500, 20_000)                  # your user id per event
items = rng.integers(0, 800, 20_000) * 10 + 7         # your item ids (gaps are fine)
ts = rng.integers(0, 90 * 86400, 20_000)              # seconds

w = time_decay(ts, reference=90 * 86400, scale=30 * 86400)   # half-life 30 days
URM, user_ids, item_ids = from_interactions(users, items, w)

model = ItemKNN(sim.cosine(sim.bm25(URM).T, k=50).T, item_ids=item_ids)
histories, weights = to_histories(URM, item_ids)
ids, scores = model.recommend_batch(histories, weights=weights, k=10)   # ids are your item ids
```

Tips:

- `urm.T` is used to switch from user-item to item-user to train an item-item similarity model.
- You can use `bm25plus`, `tfidf`, or `normalize` for different pre-processing strategies.
- `ItemKNN` is the serving engine: save it once, load it in every worker.
- All operations are multithreaded and scale with available CPU cores.
- See [Preprocessing](preprocessing.md) for repeated events, rank decay, string ids and tuning.

## In-depth Introduction

Looking for a full end-to-end example?

Check the MovieLens notebook:
- [MovieLens 32M Item-Item Recommender (GitHub)](https://github.com/bogliosimone/similaripy/blob/master/notebooks/movielens32m_item_item_recommender.ipynb)
