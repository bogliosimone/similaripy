"""Private kernel switch during the Rust port (v1 Step 2a), removed with Cython."""

import os


def resolve(backend):
    """`backend` if given, else the SIMILARIPY_BACKEND env var, else 'cython'."""
    backend = backend or os.environ.get('SIMILARIPY_BACKEND', 'cython')
    if backend not in ('cython', 'rust'):
        raise ValueError(f"unknown backend: {backend!r}")
    return backend
