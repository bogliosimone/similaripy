SimilariPy
--------

To use simply do::

    >>> import similaripy as sim
    >>> import scipy.sparse as sps
    >>> m = sps.random(100, 100, density=0.2)
    >>> s = sim.cosine_similarity(m, k=10)

Requirements:
- Cython
- GCC