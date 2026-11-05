from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("similaripy")
except PackageNotFoundError:
    __version__ = "unknown"

# The similarities (0.x API) at the top level; every other module has one path:
# `similaripy.normalization`, `similaripy.preprocessing`, `similaripy.serving`.
from . import normalization, preprocessing, serving, similarity
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
    "cosine",
    "dice",
    "dot_product",
    "jaccard",
    "normalization",
    "p3alpha",
    "preprocessing",
    "rp3beta",
    "s_plus",
    "serving",
    "similarity",
    "tversky",
]


if not TYPE_CHECKING:  # type checkers would resolve every name through it

    def __getattr__(name: str):
        # The normalizations left the top level in 1.0: say where they are.
        if name in normalization.__all__:
            raise AttributeError(f"similaripy.{name} is not at the top level: use similaripy.normalization.{name}")
        raise AttributeError(f"module 'similaripy' has no attribute {name!r}")
