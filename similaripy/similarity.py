"""Similarity functions: top-K item-item (or user-user) similarity matrices
from sparse inputs. Also at the top level (`sim.cosine`, as in 0.x)."""

from typing import Literal

import numpy as np
from scipy.sparse import coo_array, csr_array, sparray, spmatrix

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import _check_scalar, _col_sums
from ._s_plus import s_plus as _s_plus

__all__ = [
    "asymmetric_cosine",
    "cosine",
    "dice",
    "dot_product",
    "jaccard",
    "p3alpha",
    "rp3beta",
    "s_plus",
    "tversky",
]


def dot_product(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k dot products between the rows of matrix1 and the columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        k: Number of top-k items per row.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "dot_product", cosine=False, denominator=False
    )
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def cosine(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k cosine similarities between the rows of matrix1 and the columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "cosine", cosine=True, denominator=True
    )
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        l2=1,
        c1=0.5,
        c2=0.5,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def asymmetric_cosine(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    alpha: float = 0.5,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k asymmetric cosine similarities between the rows of matrix1 and the columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        alpha: Controls asymmetry in cosine weighting, in [0, 1]:
            `alpha=1` weighs only matrix1; `alpha=0.5` is symmetric.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "asymmetric_cosine", cosine=True, denominator=True
    )
    if not 0 <= alpha <= 1:  # c2 = 1 - alpha must stay >= 0; also rejects NaN
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        l2=1,
        c1=alpha,
        c2=1 - alpha,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def tversky(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k Tversky similarities between the rows of matrix1 and the columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        alpha: Tversky weight for elements unique to matrix1 (>= 0).
        beta: Tversky weight for elements unique to matrix2 (>= 0). `alpha + beta = 1` is
            always safe; on weighted or signed input other values may raise ValueError
            (the pair rule in the Similarities guide).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "tversky", cosine=False, denominator=True
    )
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        l1=1,
        t1=alpha,
        t2=beta,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def jaccard(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k Jaccard similarities (intersection over union) between the rows of matrix1 and the
    columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "jaccard", cosine=False, denominator=True
    )
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        l1=1,
        t1=1,
        t2=1,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def dice(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k Dice similarities between the rows of matrix1 and the columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "dice", cosine=False, denominator=True
    )
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        l1=1,
        t1=0.5,
        t2=0.5,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def p3alpha(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    alpha: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k P3alpha similarities: a normalized 3-step random walk from the rows of matrix1 to the
    columns of matrix2.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input,
            an extension: rows are l1-normalized by the sum of |x|, so likes and dislikes
            share the walk's budget, and the power keeps the sign, sign(p) |p|^alpha; only
            positive scores are kept; shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        alpha: Exponent for transition probabilities to control popularity effect (> 0).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    _check_scalar("alpha", alpha, strict=True)  # 0 makes every probability 1, < 0 inverts them
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "p3alpha", cosine=False, denominator=False
    )
    return _s_plus(
        matrix1=matrix1,
        matrix2=matrix2,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
        prepare=_random_walk(alpha, binary, popularity=False),
    )


def rp3beta(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k RP3beta similarities: P3alpha with a popularity penalization.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input,
            an extension as in `p3alpha`; popularity is the column sum of |x|).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        alpha: Exponent for transition probabilities (> 0).
        beta: Exponent to penalize popularity based on matrix2's column sums of |x| (>= 0).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    _check_scalar("alpha", alpha, strict=True)  # 0 makes every probability 1, < 0 inverts them
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "rp3beta", cosine=False, denominator=True
    )
    return _s_plus(
        matrix1=matrix1,
        matrix2=matrix2,
        p2=beta,
        l3=1,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
        prepare=_random_walk(alpha, binary, popularity=True),
    )


def s_plus(
    matrix1: sparray | spmatrix,
    matrix2: sparray | spmatrix | None = None,
    *,
    l1: float = 0.5,
    l2: float = 0.5,
    l3: float = 0.0,
    t1: float = 1.0,
    t2: float = 1.0,
    c1: float = 0.5,
    c2: float = 0.5,
    pop1: Literal["none", "sum"] | np.ndarray = "none",
    pop2: Literal["none", "sum"] | np.ndarray = "none",
    alpha: float = 1.0,
    beta1: float = 0.0,
    beta2: float = 0.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal["stabilized", "bayesian", "additive"] = "stabilized",
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: list[int] | np.ndarray | None = None,
    target_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    filter_cols: list[int] | np.ndarray | sparray | spmatrix | None = None,
    verbose: bool = False,
    format_output: Literal["csr", "coo"] = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
) -> csr_array | coo_array:
    """
    Top-k S-Plus similarities: Tversky and cosine normalizations with an RP3beta-style popularity
    penalization, each with its own weight.

    Args:
        matrix1: (n_rows, d) sparse matrix, e.g. `URM.T` for item-item: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: (d, n_cols) sparse matrix; None = matrix1.T.
        l1: Tversky normalization strength.
        l2: Cosine normalization strength.
        l3: Popularity penalization strength.
        t1: Tversky alpha for matrix1.
        t2: Tversky beta for matrix2. With `l1 != 0` the Tversky term must stay > 0:
            `t1 + t2 = 1` is always safe; other values may raise ValueError (the
            Tversky pair rule in the Similarities guide).
        c1: Cosine exponent coefficient for matrix1.
        c2: Cosine exponent coefficient for matrix2.
        pop1: Popularity weights for matrix1. 'none', 'sum' (sum of |x|), or custom
            array (finite, >= 0; > 0 on rows/columns with entries).
        pop2: Popularity weights for matrix2. 'none', 'sum' (sum of |x|), or custom
            array (finite, >= 0; > 0 on rows/columns with entries).
        alpha: Power applied to the raw similarity value before normalizations (also without
            them); must be > 0, and 1 on signed input (a power of a negative score is NaN or
            positive).
        beta1: Popularity penalization coefficient for matrix1 items (>= 0).
        beta2: Popularity penalization coefficient for matrix2 items (>= 0).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Binarize both inputs before the computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix;
            None = all columns, an empty one raises ValueError.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation: 'auto' (the library's choice,
            currently 262144 float32 entries, ~1 MB; may adapt in future versions),
            None = disabled, int > 0 = explicit size in float32 entries.

    Returns:
        Top-k similarities, shape (n_rows, n_cols), in `format_output` (CSR or COO).

    Raises:
        TypeError: on a non-sparse input, or `target_rows` / `target_cols` /
            `filter_cols` of a wrong type.
        ValueError: on inputs of mismatched shapes, non-finite values, more
            than 2^31 - 1 non-zeros, an invalid parameter value, an empty
            `target_cols`, or a `shrink_type` that does nothing for this
            similarity.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, "s_plus", cosine=l2 != 0, denominator=l1 != 0 or l2 != 0 or l3 != 0
    )
    return _s_plus(
        matrix1,
        matrix2=matrix2,
        l1=l1,
        l2=l2,
        l3=l3,
        t1=t1,
        t2=t2,
        c1=c1,
        c2=c2,
        a1=alpha,
        weight_depop_matrix1=pop1,
        weight_depop_matrix2=pop2,
        p1=beta1,
        p2=beta2,
        k=k,
        stabilized_shrink=stabilized_shrink,
        bayesian_shrink=bayesian_shrink,
        additive_shrink=additive_shrink,
        threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads,
        block_size=block_size,
    )


def _random_walk(alpha: float, binary: bool, popularity: bool):
    """
    `prepare` for p3alpha / rp3beta, on the call's canonical copies (cleaned
    once, no extra copy): binarize if asked, rp3beta's popularity (column sums
    of |x| of m2, the item degree when binary), then the transition
    probabilities (rows l1-normalized) raised to alpha, as in the papers.
    """

    def prepare(m1, m2):
        if binary:  # before the normalization, which the kernel's binary mode would erase
            m1.data[:] = 1
            m2.data[:] = 1
        pop = _col_sums(np.abs(m2.data), m2.indices, m2.shape[1]) if popularity else None
        for m in (m1, m2):
            _core.normalize(data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape, norm="l1")
            m.data = _signed_power(m.data, alpha)
        return pop

    return prepare


def _signed_power(x: np.ndarray, alpha: float) -> np.ndarray:
    """sign(x) |x|^alpha: x^alpha bit for bit on x > 0; on signed input a negative
    transition weight keeps its sign (np.power would give NaN for a fractional alpha).
    alpha == 1 returns x itself (the same bits, -0.0 included), saving the copies."""
    if alpha == 1:
        return x
    return np.copysign(np.abs(x) ** alpha, x)


def _shrink_values(
    shrink: float, shrink_type: str, name: str, cosine: bool, denominator: bool
) -> tuple[float, float, float]:
    """
    (stabilized, bayesian, additive) shrink for a similarity. Rejects a shrink
    that would do nothing: additive is added to the cosine norms only, and
    stabilized without a normalization term divides every score by the same
    constant (ranking unchanged).
    """
    if shrink_type not in ("stabilized", "bayesian", "additive"):
        raise ValueError("shrink_type must be one of 'stabilized', 'bayesian', or 'additive'")
    _check_scalar("shrink", shrink)
    if shrink != 0 and shrink_type == "additive" and not cosine:
        raise ValueError(f"{name}: additive shrink only applies to a cosine term; use 'stabilized' or 'bayesian'")
    if shrink != 0 and shrink_type == "stabilized" and not denominator:
        raise ValueError(
            f"{name}: stabilized shrink without a normalization only rescales the scores (same ranking); use 'bayesian'"
        )
    stabilized, bayesian, additive = (
        shrink if shrink_type == t else 0.0 for t in ("stabilized", "bayesian", "additive")
    )
    return stabilized, bayesian, additive
