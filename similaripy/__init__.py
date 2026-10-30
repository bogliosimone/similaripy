from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("similaripy")
except PackageNotFoundError:
    __version__ = "unknown"

# Functions on matrices (0.x API) at the top level; `similaripy.serving`
# (the online engine) and `similaripy.preprocessing` have one path.
from . import normalization, preprocessing, serving, similarity
from .normalization import binarize, bm25, bm25plus, normalize, rank_decay, tfidf, time_decay
from .similarity import (
    asymmetric_cosine,
    cosine,
    dice,
    dot_product,
    jaccard,
    p3alpha,
    rp3beta,
    s_plus,
    tversky,
)

__all__ = [
    "__version__",
    "asymmetric_cosine",
    "binarize",
    "bm25",
    "bm25plus",
    "cosine",
    "dice",
    "dot_product",
    "jaccard",
    "normalization",
    "normalize",
    "p3alpha",
    "preprocessing",
    "rank_decay",
    "rp3beta",
    "s_plus",
    "serving",
    "similarity",
    "tfidf",
    "time_decay",
    "tversky",
]
