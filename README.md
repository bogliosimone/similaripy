<p align="center">
  <img src="https://raw.githubusercontent.com/bogliosimone/similaripy/master/docs/logo.png" alt="similaripy" width="350"/>
</p>

# SimilariPy

[![PyPI version](https://img.shields.io/pypi/v/similaripy.svg?logo=pypi&logoColor=white)](https://pypi.org/project/similaripy/)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bogliosimone.github.io/similaripy/)
[![Build and Test](https://github.com/bogliosimone/similaripy/actions/workflows/python-package.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/python-package.yml)
[![Publish to PyPI](https://github.com/bogliosimone/similaripy/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/pypi-publish.yml)
[![Docs Status](https://github.com/bogliosimone/similaripy/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/deploy-docs.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/similaripy.svg?logo=python&logoColor=white)](https://pypi.org/project/similaripy/)
[![License](https://img.shields.io/github/license/bogliosimone/similaripy.svg)](https://github.com/bogliosimone/similaripy/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2583851.svg)](https://doi.org/10.5281/zenodo.2583851)


SimilariPy is a high-performance Python package, built in Rust, to compute KNN similarity matrices from sparse data and serve them with low latency.

It also includes normalization functions to pre-process the data.

SimilariPy is primarily designed for Recommender Systems and Information Retrieval (IR) tasks, but can be applied to other domains as well.

The official documentations is available at **[📘 SimilariPy Guide](https://bogliosimone.github.io/similaripy/)**

## 🔍 Similarity Functions

SimilariPy provides a range of high-performance similarity functions for sparse matrices.  
All functions run in a multi-threaded Rust core, GIL-free, on sparse CSR matrices.

### Core 

- **Dot Product**: Simple raw inner product between vectors.
- **Cosine**, **Asymmetric Cosine**: Normalized dot product, symmetric or skewed by `alpha`.
- **Jaccard**, **Dice**, **Tversky**: Set-based generalized similarities.

### Graph-Based

- **P3α**: Graph-based similarity computed through random walk propagation with exponentiation.
- **RP3β**: Similar to P3α but includes popularity penalization using a `beta` parameter.

### Advanced 

- **S-Plus**: A hybrid model combining Tversky and Cosine components, with full control over weights and smoothing.

For mathematical definitions and parameter details, see the **[📘 SimilariPy Guide](https://bogliosimone.github.io/similaripy/)**.

## 🛎️ Serving

The serving engine runs item-based KNN models online, in your own item ids:

- **Fast**: recommendations in tens to hundreds of microseconds, GIL-free; many users in parallel.
- **Multi-process**: save once, load memory-mapped in every worker: one copy of the model for all of them.
- **Availability rules**: include or exclude items for all requests, per group (e.g. a country, then a package) or per request.

See **[Serving](https://bogliosimone.github.io/similaripy/serving/)**.

## 🧮 Normalization Functions

SimilariPy provides a suite of normalization functions for sparse matrix pre-processing.  
All functions run in a multi-threaded Rust core, GIL-free, and return a new CSR matrix: the input never changes.

- **L1, L2**: Applies row- or column-wise normalization.
- **TF-IDF**: Computes TF-IDF weighting with customizable term-frequency and IDF modes.
- **BM25**: Applies classic BM25 weighting used in information retrieval.
- **BM25+**: Variant of BM25 with additive smoothing for low-frequency terms.

For more details, check the **[📘 SimilariPy Guide](https://bogliosimone.github.io/similaripy/)**.

## ⚡ Performance

SimilariPy is built for speed on large sparse matrices:

- **Rust + rayon**: multi-threaded, GIL-free computation compiled to native code
- **Gustavson's sparse product + cache blocking**: row-wise products with a dense accumulator, tiled by columns (most popular first) to stay in the CPU cache
- **Memory-efficient**: float32 precision, pre-allocated buffers, top-K heap filtering


## 🚀 Getting Started

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

# Offline: the top 10 items for a few test users (e.g. to evaluate them), already-seen items excluded
test_users = [1, 14, 8]
histories = [urm[[u]].indices for u in test_users]
weights = [urm[[u]].data for u in test_users]
ids, scores = model.recommend_batch(histories, weights=weights, k=10)

# Online: a new user, from the items they interacted with (no retraining)
ids, scores = model.recommend([3, 17, 256], k=10)
```

For a more in-depth, end-to-end example with SimilariPy, see the notebook:
**[MovieLens 32M Item-Item Recommender](https://github.com/bogliosimone/similaripy/blob/master/notebooks/movielens32m_item_item_recommender.ipynb)**.

## 📦 Installation

SimilariPy can be installed from PyPI with:

```cmd
pip install similaripy
```

Pre-built wheels are available for Python 3.11 to 3.15 on:

- Linux x86_64 and aarch64 (glibc 2.28+, or musl 1.2+ e.g. Alpine)
- macOS 12+ (Apple Silicon) and 10.15+ (Intel)
- Windows x64

No compiler is needed on these platforms. Elsewhere, pip builds the package from source, which needs a Rust toolchain: see **[Build from source](https://bogliosimone.github.io/similaripy/installation/#build-from-source)**.

Free-threaded Python (3.14t, 3.15t) is supported: the GIL stays disabled. When calling SimilariPy from several Python threads at once, set `num_threads` on each call to avoid oversubscribing the CPU cores.

## 📦 Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| python                          |   >= 3.11      |
| numpy                           |   >= 1.23.5    |
| scipy                           |   >= 1.15.0    |

## 📜 History

This library originated during the **[Spotify Recsys Challenge 2018](https://research.atspotify.com/publications/recsys-challenge-2018-automatic-music-playlist-continuation/)**.

Our team, The Creamy Fireflies, faced major challenges computing large similarity models on a dataset with over 66 million interactions. Standard Python/Numpy solutions were too slow as a whole day was required to compute one single model.

To overcome this, I developed high-performance versions of the core similarity functions in Cython and OpenMP. Encouraged by my teammates, I open-sourced this work to help others solve similar challenges. For 1.0 the core was rewritten in Rust.

Thanks to my Creamy Fireflies friends for the support! 🙏 

## 📄 License

This project is released under the MIT License.

## 🔖 Citation

If you use SimilariPy in your research, please cite:

```text
@misc{boglio_simone_similaripy,
  author       = {Boglio Simone},
  title        = {bogliosimone/similaripy},
  doi          = {10.5281/zenodo.2583851},
  url          = {https://doi.org/10.5281/zenodo.2583851}
}
```
