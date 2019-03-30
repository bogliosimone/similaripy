SimilariPy
==========

![PY_PIC]
[![PYPI_PIC]][PYPI_LINK]
[![LICENSE_PIC]][LICENSE_LINK]
[![DOI_PIC]][DOI_LINK]


This project provides fast Python implementation of several KNN (K-Nearest Neighbors) similarity algorithms using sparse matrices, useful in Collaborative Filtering Recommender Systems and others.

The package also include some normalization functions that could be useful in the pre-processing phase before the similarity computation.

#### Similarities

Base similarity models:
 * Dot Product
 * Cosine
 * Asymmetric Cosine
 * Jaccard
 * Dice
 * Tversky

 Graph-based similarity models:
 * P3&alpha;
 * RP3&beta;

 Advanced similarity model:
 * S-Plus

[Similarities Documentation](https://github.com/bogliosimone/similaripy/blob/master/guide/temp_guide.md)

All models have multi-threaded routines, using Cython and OpenMP to fit the models in parallel among all available CPU cores.

#### Normalizations

The package contains normalization functions like: l1, l2, max, tf-idf, bm25, bm25+.

All the functions are compiled at low-level and could operate in-place, on csr-matrixes, if you need to save memory.

For tf-idf, bm25, bm25+ you could chose the log-base and how the term-frequency (TF) and the inverse document frequency (IDF) are computed.

#### Installation and usage

To install:

```cmd
pip install similaripy
```

Basic usage:

```python
import similaripy as sim
import scipy.sparse as sps

# create a random user-rating matrix (URM)
urm = sps.random(1000, 2000, density=0.025)

# normalize matrix with bm25
urm = sim.normalization.bm25(urm)

# train the model with 50 knn per item 
model = sim.cosine(urm.T, k=50)

# recommend 100 items to users 1, 14 and 8
user_recommendations = sim.dot_product(urm, model.T, target_rows=[1,14,8], k=100)

```

#### Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|   
| numpy                           |   >= 1.14      |   
| scipy                           |   >= 1.0.0     |
| tqdm                            |   >= 4.19.6    |
| cython                          |   >= 0.28.1    |


NOTE: In order to compile the Cython code it is required a GCC compiler with OpenMP 
(on OSX it can be installed with homebrew: ```brew install gcc```).

This library has been tested with Python 3.6 on Ubuntu, OSX and Windows.

(Note: on Windows there are problem with flag *format_output='csr'*, just let it equals to the default value *'coo'*)

#### Optimal Configuration

I recommend configuring SciPy/Numpy to use Intel's MKL matrix libraries.
The easiest way of doing this is by installing the Anaconda Python distribution.

#### Future work

I plan to release in the next future some utilities:
- Utilities for sparse matrices
- New similarity functions ( good ideas are welcome :)  )

#### History
The idea of build this library comes from the **[RecSys Challenge 2018](https://recsys-challenge.spotify.com)** organized by Spotify. 

My team, the Creamy Fireflies, had problem in compute very huge similarity models in a reasonable time (66 million of interactions in the user-rating matrix) and using python and numpy were not suitable since a full day was required to compute one single model.

As a member of the the team I spent a lot of hours to develop these high-performance similarities in Cython to overcome the problem. At the end of the competition, pushed by my team friends, I decide to release my work to help people that one day will encounter our same problem.

Thanks to my Creamy Fireflies friends for support me.

#### License
Released under the MIT License

Citation information: [![DOI_PIC]][DOI_LINK]

```
@misc{boglio_simone_2019_2616944,
  author       = {Boglio Simone},
  title        = {bogliosimone/similaripy: v0.0.13 stable release},
  month        = mar,
  year         = 2019,
  doi          = {10.5281/zenodo.2616944},
  url          = {https://doi.org/10.5281/zenodo.2616944}
}
```

[DOI_PIC]: https://zenodo.org/badge/DOI/10.5281/zenodo.2616944.svg
[DOI_LINK]: https://doi.org/10.5281/zenodo.2616944
[LICENSE_PIC]: https://img.shields.io/github/license/bogliosimone/similaripy.svg
[LICENSE_LINK]: https://github.com/bogliosimone/similaripy/blob/master/LICENSE
[PYPI_PIC]: https://img.shields.io/pypi/v/similaripy.svg?color=blue
[PYPI_LINK]: https://pypi.org/project/similaripy/
[PY_PIC]: https://img.shields.io/pypi/pyversions/similaripy.svg
