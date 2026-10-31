"""
Everything that differs between similaripy builds, in one place.

Checked by feature, never by version string: v1 builds report
`__version__` 0.7.0 until the 1.0 release. What 0.7.0 (the last Cython
release, PyPI wheel) lacks: `similaripy._s_plus` (phase timer, column order
switch), `similaripy.serving`, `similaripy.preprocessing`. Its similarity
signatures match v1 by name; it defaults to verbose=True and
format_output='coo' (the harness always passes both) and spells the library's
block size 0 (v1: 'auto'), so 'auto' is left out of the call.
"""

import importlib.util
import subprocess
from pathlib import Path

import numpy as np

import similaripy as sim

try:
    from similaripy import _s_plus
except ImportError:  # 0.7.0
    _s_plus = None

HAS_PHASES = hasattr(_s_plus, "_last_timings")
HAS_SERVING = importlib.util.find_spec("similaripy.serving") is not None
DEFAULT_BLOCK_SIZE = getattr(_s_plus, "DEFAULT_BLOCK_SIZE", None)


def phase_times():
    """{kernel, output} seconds of the last similarity call, or None (no phase timer in this build)."""
    return dict(_s_plus._last_timings) if HAS_PHASES else None


def set_column_order(order):
    """Set the private kernel column order ('popularity' / 'original'); None leaves the library default."""
    if order is None:
        return
    if not hasattr(_s_plus, "_COLUMN_ORDER"):
        raise ValueError("--column-order: this build has no column order switch")
    _s_plus._COLUMN_ORDER = order


def block_size_kwargs(block_size):
    """'auto' = the library's default, left out of the call; None (no blocking) or an int passed as is."""
    return {} if block_size == "auto" else {"block_size": block_size}


def build_info():
    """Where the timed similaripy comes from: version, path, editable source tree or not, and its git state."""
    package = Path(sim.__file__).resolve().parent
    root = package.parent
    editable = (root / "Cargo.toml").exists()  # a source checkout, not site-packages
    return {
        "similaripy_version": sim.__version__,
        "path": str(package),
        "editable": editable,
        "git": git_describe(root) if editable else None,
        "has_phase_timer": HAS_PHASES,
    }


def git_describe(path):
    """`git describe --always --dirty` of the tree at path, or None."""
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


# Serving internals the serve workload times (private API: a core change edits only these)


def serving_n_active(rec):
    """Items with similarities in an ItemKNN model."""
    return rec._model.n_active


def serving_batch_inputs(URM):
    """URM's rows as the core batch call takes them: (int64 offsets, int64 item ids, float32 weights),
    canonicalized as the kernel's prep does (ids = column indices: models built without item_ids)."""
    H = _s_plus._canonical_csr(URM, np.float32, 0)
    return H.indptr.astype(np.int64), H.indices.astype(np.int64), H.data


def serving_batch(rec, inputs, k, num_threads):
    """The core batch call of `recommend_batch` (no per-user list handling): (offsets, ids, scores)."""
    offsets, ids, weights = inputs
    return rec._model.batch(
        offsets,
        ids,
        weights,
        None,
        None,
        None,
        None,
        None,
        k=k,
        exclude_seen=True,
        sorted=True,
        num_threads=num_threads,
    )
