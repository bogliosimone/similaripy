SimilariPy
==========

Fast Python KNN-Similarity algorithms for Collaborative Filtering models in Recommender System and others.

This project provides fast Python implementations of several different popular KNN (K-Nearest Neighbors) similarity algorithms for Recommender System models using sparse matrices.

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

[ Complete documentation coming soon... ]

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

# recommend 100 items to users 1, 14 and 8
user_recommendations = sim.dot_product(urm, model.T, target_rows=[1,14,8], k=100)

```

For more information see the [documentation](). [ TODO ]


#### Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|   
| numpy                           |   >= 1.14      |   
| scipy                           |   >= 1.0.0     |
| tqdm                            |   >= 4.19.6    |
| scikit-learn                    |   >= 0.19.1    |
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
- Pre-processing / post-processing functions (TF-IDF, BM25 and more)
- New similarity functions ( good ideas are welcome :)  )

#### History
The idea of build this library comes from the **[RecSys Challenge 2018](https://recsys-challenge.spotify.com)** organized by Spotify. 

My team, the Creamy Fireflies, had problem in compute very huge similarity models in a reasonable time (66 million of interactions in the user-rating matrix) and using python and numpy were not suitable since a full day was required to compute one single model.

As a member of the the team I spent a lot of hours to develop these high-performance similarities in Cython to overcome the problem. At the end of the competition, pushed by my team friends, I decide to release my work to help people that one day will encounter our same problem.

Thanks to my Creamy Fireflies friends for support me.

#### License
Released under the MIT License

