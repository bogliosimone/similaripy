SimilariPy
==========

Fast Python KNN-Similarities for Collaborative Filtering models in Recommender System and others.

This project provides fast Python implementations of several different popular KNN (topK-Nearest Neighbors) similarities for Recommender System models.

Base similarites:
 * Dot Product
 * Cosine
 * Asymmetric Cosine
 * Jaccard
 * Dice
 * Tversky

 Graph-based similarities:
 * P3Alpha
 * RP3Beta

 Advanced similarity:
 * S-plus

[ Complete documentation coming soon... ] [ TODO ]

All models have multi-threaded routines, using Cython and OpenMP to fit the models in parallel among all available CPU cores.

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

# train the model with 50 knn per item 
model = sim.cosine(urm.T, k=50)

# recommend items for users 1, 14 and 8
user_recommendations = dot_product(urm, model, target_rows=[1,14,8], k=100)

```

For more information see the [documentation](http://similaripy.readthedocs.io/). [ TODO ]


#### Requirements

This library requires:
- SciPy >> 0.16
- Numpy >>
- Cython.

In order to compile the Cython code it is required a GCC compiler with OpenMP 
(on OSX it can be installed with homebrew: ```brew install gcc```).

This library has been tested with Python 3.6 on Ubuntu, OSX and Windows.

#### Optimal Configuration

I recommend configuring SciPy/Numpy to use Intel's MKL matrix libraries.
The easiest way of doing this is by installing the Anaconda Python distribution.

Released under the MIT License

