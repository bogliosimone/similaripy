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


High-performance KNN similarity functions in Python, optimized for sparse matrices.

SimilariPy is primarily designed for Recommender Systems and Information Retrieval (IR) tasks, but can be applied to other domains as well.

The package also includes a set of normalization functions useful for pre-processing data before the similarity computation.

The official documentations is available at **[📘 SimilariPy Guide](https://bogliosimone.github.io/similaripy/)**

## 🔍 Similarity Functions

SimilariPy provides a range of high-performance similarity functions for sparse matrices.  
All functions are multi-threaded and implemented in Cython + OpenMP for fast parallel computation on CSR matrixes.

### Core 

- **Dot Product** – Simple raw inner product between vectors.
- **Cosine** – Normalized dot product based on L2 norm.
- **Asymmetric Cosine** – Skewed cosine similarity using an `alpha` parameter.
- **Jaccard**, **Dice**, **Tversky** – Set-based generalized similarities.

### Graph-Based

- **P3α** – Graph-based similarity computed through random walk propagation with exponentiation.
- **RP3β** – Similar to P3α but includes popularity penalization using a `beta` parameter.

### Advanced 

- **S-Plus** – A hybrid model combining Tversky and Cosine components, with full control over weights and smoothing.

For mathematical definitions and parameter details, see the **[📘 SimilariPy Guide](https://bogliosimone.github.io/similaripy/)**.

## 🧮 Normalization Functions

SimilariPy provides a suite of normalization functions for sparse matrix pre-processing.  
All functions return a new canonical CSR matrix (multi-threaded Rust core); the input never changes.

- **L1, L2** – Applies row- or column-wise normalization.
- **TF-IDF** – Computes TF-IDF weighting with customizable term-frequency and IDF modes.
- **BM25** – Applies classic BM25 weighting used in information retrieval.
- **BM25+** – Variant of BM25 with additive smoothing for low-frequency terms.

For more details, check the **[📘 SimilariPy Guide](https://bogliosimone.github.io/similaripy/)**.

## ⚡ Performance

SimilariPy is built for speed on large sparse matrices:

- **Cython + C++ + OpenMP** — multi-threaded, GIL-free computation compiled to native code
- **Cache-optimized** — column-blocked accumulation with popularity reordering minimizes L2 cache misses
- **Memory-efficient** — float32 precision, pre-allocated buffers, top-K heap filtering


## 🚀 Getting Started

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

For a more in-depth, end-to-end example with SimilariPy, see the notebook:
**[MovieLens 32M Item-Item Recommender](notebooks/movielens32m_item_item_recommender.ipynb)**.

## 📦 Installation

SimilariPy can be installed from PyPI with:

```cmd
pip install similaripy
```

Pre-built wheels are available for Python 3.11–3.15 on:

- Linux x86_64 and aarch64 (glibc 2.28+, or musl 1.2+ e.g. Alpine)
- macOS 15+ (Apple Silicon and Intel)
- Windows x64

No compiler is needed on these platforms. Elsewhere, pip builds the package from source (see below).

Free-threaded Python (3.14t, 3.15t) is supported: the GIL stays disabled. When calling SimilariPy from several Python threads at once, set `num_threads` on each call to avoid oversubscribing the CPU cores.

### 🔧 Build from source

Building from source is needed only when no wheel matches your platform, or to compile with CPU-specific optimizations. It requires a GCC-compatible compiler with OpenMP.

#### Ubuntu / Debian

Install the official dev-tools:

```bash
sudo apt update && sudo apt install build-essential
```

#### MacOS (Intel & Apple Silicon)

Install GCC with homebrew:

```bash
brew install gcc
```

#### Windows

Install the official **[Visual C++ Build Tools](https://visualstudio.microsoft.com/en/visual-cpp-build-tools/)**.

#### Optional: native CPU optimizations

Wheels target a generic CPU baseline. To build for your exact CPU (use `-mcpu=native` on ARM):

```bash
pip install similaripy --no-binary similaripy -C cmake.define.CMAKE_CXX_FLAGS="-march=native"
```

The speedup depends on the CPU: on Apple Silicon we measured none, so benchmark it on your workload.

#### Optional Optimization: Intel MKL for Intel CPUs

For Intel CPUs, using SciPy/Numpy with MKL (Math Kernel Library) is highly recommended for best performance.
The easiest way to achieve this is to install them via Anaconda.

## 📦 Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| python                          |   >= 3.11      |
| numpy                           |   >= 1.23.5    |
| scipy                           |   >= 1.15.0    |

## 📜 History

This library originated during the **[Spotify Recsys Challenge 2018](https://research.atspotify.com/publications/recsys-challenge-2018-automatic-music-playlist-continuation/)**.

Our team, The Creamy Fireflies, faced major challenges computing large similarity models on a dataset with over 66 million interactions. Standard Python/Numpy solutions were too slow as a whole day was required to compute one single model.

To overcome this, I developed high-performance versions of the core similarity functions in Cython and OpenMP. Encouraged by my teammates, I open-sourced this work to help others solve similar challenges.

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
