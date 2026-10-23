from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("similaripy")
except PackageNotFoundError:
    __version__ = "unknown"

from ._serve import ItemKNN
from .normalization import binarize, bm25, bm25plus, normalize, tfidf
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
    "ItemKNN",
    "__version__",
    "asymmetric_cosine",
    "binarize",
    "bm25",
    "bm25plus",
    "cosine",
    "dice",
    "dot_product",
    "jaccard",
    "normalize",
    "p3alpha",
    "rp3beta",
    "s_plus",
    "tfidf",
    "tversky",
]
