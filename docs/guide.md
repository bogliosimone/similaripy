# 📌 Quickstart

SimilariPy is a high-performance Python package, built in Rust, to **compute KNN similarity matrices** from sparse data and **serve them with low latency**.

It also includes normalization and weighting functions to pre-process the data.

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

More details in [Similarities](similarity.md).

## Serving

The serving engine turns a similarity matrix into recommendations, in your own item ids.  
All requests run in the Rust core, GIL-free: one user per call, or a batch of users on multiple threads.

- **Offline**: recommendations for a batch of users, e.g. to evaluate a model or precompute results.
- **Online**: one user per call, in tens to hundreds of microseconds.
- **Availability rules**: include or exclude items for all requests, per group, or per request.

More details in [Serving](serving.md).

## Normalization and Weighting

Re-weight your data before the similarity computation: normalization functions on sparse matrices, and time decay of interactions.

- **L1, L2**: Applies row- or column-wise normalization.
- **TF-IDF**: Computes TF-IDF weighting with customizable term-frequency and IDF modes.
- **BM25**: Applies classic BM25 weighting used in information retrieval.
- **BM25+**: Variant of BM25 with additive smoothing for low-frequency terms.
- **Time and rank decay**: weight each interaction by its age or recency.

More details in [Normalizations](normalization.md) and [Preprocessing](preprocessing.md).

## Example

From an interaction log to recommendations:


```python
import similaripy as sim
from similaripy.preprocessing import interactions_to_histories, interactions_to_urm
from similaripy.serving import ItemKNN

# An interaction log: one (user, item) pair per event, in your own ids
users = [10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 50, 50]
items = [101, 102, 103, 101, 103, 104, 102, 103, 105, 101, 104, 102, 105]

# The user-item matrix (URM), with the id maps: row u is user_ids[u], column j is item_ids[j]
# (older events can count less: see time_decay in Preprocessing)
urm, user_ids, item_ids = interactions_to_urm(users, items)

# Train an item-item cosine similarity model
similarity_matrix = sim.cosine(urm.T, k=50)

# Build the serving model, in your item ids
model = ItemKNN(similarity_matrix.T, item_ids=item_ids)

# Online: a user from the items they interacted with (no retraining)
# exclude_seen=True drops the items in the history (the default); False keeps them, e.g. music or grocery
ids, scores = model.recommend([101, 104], k=3, exclude_seen=True)
print(ids)  # [103 102]: item ids from the log

# Offline: every user of the log at once, e.g. to evaluate a model
user_ids, histories, _ = interactions_to_histories(users, items)  # _: weights, all 1.0 here
ids, scores = model.recommend_batch(histories, k=3, exclude_seen=True)  # ids[u]: user user_ids[u]
```

Tips:

- `urm.T` is used to switch from user-item to item-user to train an item-item similarity model.
- On real data, re-weight the URM before the similarity, e.g. `urm = sim.bm25(urm)`; `bm25plus`, `tfidf` and `normalize` are the alternatives ([Normalizations](normalization.md)).
- `ItemKNN` is the serving engine: save it once, load it in every worker.
- All operations are multithreaded and scale with available CPU cores.

## In-depth Introduction

Looking for a full end-to-end example?

Check the MovieLens notebook:
- [MovieLens 32M Item-Item Recommender (GitHub)](https://github.com/bogliosimone/similaripy/blob/master/notebooks/movielens32m_item_item_recommender.ipynb)
