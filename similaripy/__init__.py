from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("similaripy")
except PackageNotFoundError:
    __version__ = "unknown"

from .normalization import normalize, binarize, bm25, bm25plus, tfidf
from .similarity import (
    dot_product,
    cosine,
    asymmetric_cosine,
    jaccard,
    dice,
    tversky,
    p3alpha,
    rp3beta,
    s_plus,
)

__all__ = [
    "__version__",
    "normalize",
    "binarize",
    "bm25",
    "bm25plus",
    "tfidf",
    "dot_product",
    "cosine",
    "asymmetric_cosine",
    "jaccard",
    "dice",
    "tversky",
    "p3alpha",
    "rp3beta",
    "s_plus",
]
