<p align="center"><img src="https://raw.githubusercontent.com/bogliosimone/similaripy/master/logo.png" alt="similaripy" width="350"/></p>

# SimilariPy

![PY_PIC]
[![PYPI_PIC]][PYPI_LINK]
[![BUILD_STATUS_PIC]][BUILD_STATUS_LINK]
[![LICENSE_PIC]][LICENSE_LINK]
[![DOI_PIC]][DOI_LINK]

This project provides fast Python implementation of several KNN (K-Nearest Neighbors) similarity algorithms using sparse matrices, useful in Collaborative Filtering Recommender Systems and others.

The package also include some normalization functions that could be useful in the pre-processing phase before the similarity computation.

## Similarities

Base similarity models:

* Dot Product
* Cosine and Asymmetric Cosine
* Tversky, Jaccard and Dice

 Graph-based similarity models:

* P3&alpha; and RP3&beta;

 Advanced similarity model:

* S-Plus

For additional information see the [similarities documentation](https://github.com/bogliosimone/similaripy/blob/master/guide/temp_guide.md).

All the models have multi-threaded routines, using Cython and OpenMP to fit the models in parallel among all available CPU cores.

## Normalizations

The package contains normalization functions like: l1, l2, max, tf-idf, bm25, bm25+.

For tf-idf, bm25, bm25+ you could chose the log-base and how the term-frequency (TF) and the inverse document frequency (IDF) are computed.

All the functions are compiled at low-level and could operate in-place on csr-matrixes, if you need to save memory.

## Usage

```python
import similaripy as sim
import scipy.sparse as sps

# create a random user-rating matrix (URM)
urm = sps.random(1000, 2000, density=0.025)

# normalize matrix with bm25
urm = sim.normalization.bm25(urm)

# train the model with 50 knn per item 
model = sim.cosine(urm.T, k=50)

# recommend 100 items to users 1, 14 and 8 filtering the items already seen by each users
user_recommendations = sim.dot_product(urm, model.T, k=100, target_rows=[1,14,8], filter_cols=urm)

```

## Installation

Similaripy can be installed from PyPI with:

```cmd
pip install similaripy
```

### Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| numpy                           |   >= 1.14      |
| scipy                           |   >= 1.0.0     |
| tqdm                            |   >= 4.19.6    |
| cython                          |   >= 0.28.1    |

### GCC compiler

In order to compile the Cython code it is required a GCC compiler with OpenMP.

To install it you can follow the next steps according to your OS.

#### Ubuntu / Debian

It can be installed with the dev-tools:

```cmd
sudo apt update && sudo apt install build-essential
```

#### MacOS

It can be installed with homebrew:

```cmd
brew install gcc
```

NOTE: the package works on both Intel and Apple Silicon chips.

#### Windows

It can be installed with the Visual C++ Build Tools.

They can be downloaded directly from Microsoft: **https://visualstudio.microsoft.com/en/visual-cpp-build-tools/**.

NOTE: on Windows there are issues with the flag *format_output='csr'*, just leave it set to the default value *'coo'*.

#### Optimal Configuration for Intel CPUs

I recommend configuring SciPy/Numpy to use Intel's MKL matrix libraries.
The easiest way of doing this is by installing the Anaconda Python distribution.

## History

The idea of build this library comes from the **[RecSys Challenge 2018](https://recsys-challenge.spotify.com)** organized by Spotify.

My team, the Creamy Fireflies, had problems in computing huge similarity models in a reasonable time (66 million interactions in the user-rating matrix), and the use of Python and Numpy was not suitable as a full day was required to compute one single model.

As a member of the team, I have spent a lot of time developing these high-performance similarities in Cython to overcome this issue. At the end of the competition, pushed by my teammates, I decided to release my job to help people who will one day face the same problem.

Thanks to my Creamy Fireflies friends for supporting me.

## License

Released under the MIT License

Citation information: [![DOI_PIC]][DOI_LINK]

```text
@misc{boglio_simone_similaripy,
  author       = {Boglio Simone},
  title        = {bogliosimone/similaripy},
  doi          = {10.5281/zenodo.2583851},
  url          = {https://doi.org/10.5281/zenodo.2583851}
}
```

[DOI_PIC]: https://zenodo.org/badge/DOI/10.5281/zenodo.2583851.svg
[DOI_LINK]: https://doi.org/10.5281/zenodo.2583851
[LICENSE_PIC]: https://img.shields.io/github/license/bogliosimone/similaripy.svg
[LICENSE_LINK]: https://github.com/bogliosimone/similaripy/blob/master/LICENSE
[PYPI_PIC]: https://img.shields.io/pypi/v/similaripy.svg
[PYPI_LINK]: https://pypi.org/project/similaripy/
[PY_PIC]: https://img.shields.io/pypi/pyversions/similaripy.svg
[BUILD_STATUS_PIC]: https://github.com/bogliosimone/similaripy/workflows/Python%20package/badge.svg?branch=master
[BUILD_STATUS_LINK]: https://github.com/bogliosimone/similaripy/actions?query=workflow%3A%22Python+package%22
