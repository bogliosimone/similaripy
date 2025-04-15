<p align="center">
  <img src="https://raw.githubusercontent.com/bogliosimone/similaripy/master/logo.png" alt="similaripy" width="350"/>
</p>

# SimilariPy

[![PyPI version](https://img.shields.io/pypi/v/similaripy.svg?logo=pypi&logoColor=white)](https://pypi.org/project/similaripy/)
[![Build and Test](https://github.com/bogliosimone/similaripy/actions/workflows/python-package.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/python-package.yml)
[![Publish to PyPI](https://github.com/bogliosimone/similaripy/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/pypi-publish.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/similaripy.svg?logo=python&logoColor=white)](https://pypi.org/project/similaripy/)
[![License](https://img.shields.io/github/license/bogliosimone/similaripy.svg)](https://github.com/bogliosimone/similaripy/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2583851.svg)](https://doi.org/10.5281/zenodo.2583851)


Similaripy provides high-performance KNN (K-Nearest Neighbors) similarity functions in Python, optimized for sparse matrices. 

It's primarily designed for Recommender Systems and IR tasks, but can be applied to other domains as well.

The package also includes a set of normalization functions useful for pre-processing data before the similarity computation.

## Similarities

**Base similarity models:**

- Dot Product
- Cosine and Asymmetric Cosine
- Tversky, Jaccard, and Dice

**Graph-based similarity models:**

- P3&alpha; and RP3&beta;

**Advanced similarity model:**

- S-Plus

For additional details about parameters and mathematical formulas, check the [📘 Similarity Guide](docs/guide.md).

## Normalizations

The package includes normalization methods such as:

- **L1**, **L2**, **max**, **tf-idf**, **bm25**, **bm25+**

For *tf-idf*, *bm25*, and *bm25+*, you can chose how the *log base*, the *term frequency* (TF) and the *inverse document frequency* (IDF) are computed.

All functions are compiled and optimized to operate in-place on CSR matrices for memory efficiency.

---

## 🚀 Usage Example

```python
import similaripy as sim
import scipy.sparse as sps

# Create a random user-rating matrix (URM)
urm = sps.random(1000, 2000, density=0.025)

# Normalize matrix with BM25
urm = sim.normalization.bm25(urm)

# Train the model with 50 nearest neighbors per item 
model = sim.cosine(urm.T, k=50)

# Recommend 100 items to users 1, 14, and 8, filtering already seen items
user_recommendations = sim.dot_product(urm, model.T, k=100, target_rows=[1, 14, 8], filter_cols=urm)
```

## 📦 Installation

SimilariPy can be installed from PyPI with:

```cmd
pip install similaripy
```

### Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| numpy                           |   >= 1.21      |
| scipy                           |   >= 1.10.1    |
| tqdm                            |   >= 4.65.2    |

### 🔧 GCC Compiler - Required

All similarities are multi-threaded using Cython and OpenMP for fast parallel computation across CPU cores.

To install the package and compile the Cython code, a GCC-compatible compiler with OpenMP is required.

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

⚠️ On Windows, set *format_output='coo'* (default) in all similarity functions, as *'csr'* is currently not supported.


#### 🧠 Optional Optimization: Intel MKL for Intel CPUs

For Intel CPUs, using SciPy/Numpy with MKL (Math Kernel Library) is highly recommended for best performance.
The easiest way to achieve this is to install them via Anaconda.

## History

This library originated during the **[Spotify Recsys Challenge 2018](https://research.atspotify.com/publications/recsys-challenge-2018-automatic-music-playlist-continuation/)**.

Our team, The Creamy Fireflies, faced major challenges computing large similarity models on a dataset with over 66 million interactions. Standard Python/Numpy solutions were too slow as a whole day was required to compute one single model.

To overcome this, I developed high-performance versions of the core similarity functions in Cython and OpenMP. Encouraged by my teammates, I open-sourced this work to help others solve similar challenges.

🙏 Thanks to my Creamy Fireflies friends for their support!

## License

This project is released under the MIT License.

## Citation

If you use SimilariPy in your research, please cite:

```text
@misc{boglio_simone_similaripy,
  author       = {Boglio Simone},
  title        = {bogliosimone/similaripy},
  doi          = {10.5281/zenodo.2583851},
  url          = {https://doi.org/10.5281/zenodo.2583851}
}
```
