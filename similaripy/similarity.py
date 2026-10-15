from ._s_plus import s_plus as _s_plus
from .normalization import binarize as _binarize, normalize as _normalize
import numpy as np

from typing import Optional, Union, Literal
from scipy.sparse import sparray



def dot_product(
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute dot product similarity between rows of matrix1 and columns of matrix2.

    Args:
        matrix1: Input sparse matrix (e.g., user-item or item-user): values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before similarity computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 means all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of shape (n_rows, n_cols) in the specified format,
        containing the top-k dot product similarities.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'dot_product', cosine=False, denominator=False)
    return _s_plus(
        matrix1, matrix2=matrix2,
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
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute cosine similarity between sparse vectors.

    Args:
        matrix1: Input sparse matrix: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k cosine similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'cosine', cosine=True, denominator=True)
    return _s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=0.5, c2=0.5,
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
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    alpha: float = 0.5,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute asymmetric cosine similarity.

    Args:
        matrix1: Input sparse matrix (e.g., user-item or item-user): values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Controls asymmetry in cosine weighting, in [0, 1].
               `alpha=1` weighs only matrix1; `alpha=0.5` is symmetric.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of shape (n_rows, n_cols) containing the top-k
        asymmetric cosine similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'asymmetric_cosine', cosine=True, denominator=True)
    if not 0 <= alpha <= 1:  # c2 = 1 - alpha must stay >= 0; also rejects NaN
        raise ValueError(f'alpha must be in [0, 1], got {alpha}')
    return _s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=alpha, c2=1-alpha,
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
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute Tversky similarity between sparse vectors.

    Args:
        matrix1: Input sparse matrix: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Tversky weight for elements unique to matrix1 (>= 0).
        beta: Tversky weight for elements unique to matrix2 (>= 0). Unless binary, alpha + beta = 1
            or 4 alpha beta > (1 - alpha - beta)^2 is needed when alpha + beta > 1, and also when
            alpha + beta < 1 on signed input (else ValueError: the denominator can be <= 0).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k Tversky similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'tversky', cosine=False, denominator=True)
    return _s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=alpha, t2=beta,
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
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute Jaccard similarity (intersection over union).

    Args:
        matrix1: Input sparse matrix: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k Jaccard similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'jaccard', cosine=False, denominator=True)
    return _s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=1, t2=1,
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
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute Dice similarity (harmonic mean of overlap and size).

    Args:
        matrix1: Input sparse matrix: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k Dice similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'dice', cosine=False, denominator=True)
    return _s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=0.5, t2=0.5,
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
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    alpha: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute P3alpha similarity using a normalized 3-step random walk.

    Args:
        matrix1: Input sparse matrix: values finite, negative ones allowed (signed input,
            an extension: rows are l1-normalized by the sum of |x|, so likes and dislikes
            share the walk's budget, and the power keeps the sign, sign(p) |p|^alpha; only
            positive scores are kept; shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Exponent for transition probabilities to control popularity effect (> 0).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k P3alpha similarities in the specified format.
    """
    if not (np.isfinite(alpha) and alpha > 0):  # 0 makes every probability 1, < 0 inverts them
        raise ValueError(f'alpha must be finite and > 0, got {alpha}')
    if matrix2 is None:
        matrix2 = matrix1.T
    if binary:  # before the normalization, which the kernel's binary mode would erase
        matrix1, matrix2 = _binarize(matrix1), _binarize(matrix2)
    matrix1 = _normalize(matrix1, norm='l1', axis=1, inplace=False)
    matrix1.data = _signed_power(matrix1.data, alpha)
    matrix2 = _normalize(matrix2, norm='l1', axis=1, inplace=False)
    matrix2.data = _signed_power(matrix2.data, alpha)
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'p3alpha', cosine=False, denominator=False)
    return _s_plus(
        matrix1=matrix1, matrix2=matrix2,
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
    )


def rp3beta(
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Compute RP3beta similarity: P3alpha with popularity penalization.

    Args:
        matrix1: Input sparse matrix: values finite, negative ones allowed (signed input,
            an extension as in `p3alpha`; popularity is the column sum of |x|).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Exponent for transition probabilities (> 0).
        beta: Exponent to penalize popularity based on matrix2's column sums of |x| (>= 0).
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value (>= 0) that prevents instability when normalizations are small.
            A shrink_type that would do nothing for this similarity raises ValueError.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain (>= 0).
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k RP3beta similarities in the specified format.
    """
    if not (np.isfinite(alpha) and alpha > 0):  # 0 makes every probability 1, < 0 inverts them
        raise ValueError(f'alpha must be finite and > 0, got {alpha}')
    if matrix2 is None:
        matrix2 = matrix1.T
    if binary:  # before the normalization, which the kernel's binary mode would erase
        matrix1, matrix2 = _binarize(matrix1), _binarize(matrix2)
    pop_m2 = _abs_col_sums(matrix2)  # item degree when binary
    matrix1 = _normalize(matrix1, norm='l1', axis=1, inplace=False)
    matrix1.data = _signed_power(matrix1.data, alpha)
    matrix2 = _normalize(matrix2, norm='l1', axis=1, inplace=False)
    matrix2.data = _signed_power(matrix2.data, alpha)
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 'rp3beta', cosine=False, denominator=True)
    return _s_plus(
        matrix1=matrix1, matrix2=matrix2,
        weight_depop_matrix2=pop_m2,
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
    )


def s_plus(
    matrix1: sparray,
    matrix2: Optional[sparray] = None,
    l1: float = 0.5,
    l2: float = 0.5,
    l3: float = 0.0,
    t1: float = 1.0,
    t2: float = 1.0,
    c1: float = 0.5,
    c2: float = 0.5,
    pop1: Optional[Union[Literal['none','sum'], np.ndarray]]= 'none',
    pop2: Optional[Union[Literal['none','sum'], np.ndarray]]= 'none',
    alpha: float = 1.0,
    beta1: float = 0.0,
    beta2: float = 0.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, sparray]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> sparray:
    """
    Combines Tversky and Cosine normalizations with RP3Beta-style depopularization, fully controlled by tunable weights

    Args:
        matrix1: Input sparse matrix: values finite,
            negative ones allowed (signed input: only positive scores are kept;
            shrink_type='bayesian' raises).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        l1: Tversky normalization strength.
        l2: Cosine normalization strength.
        l3: Popularity penalization strength.
        t1: Tversky alpha for matrix1.
        t2: Tversky beta for matrix2. With l1 != 0 and not binary, t1 + t2 = 1 or
            4 t1 t2 > (1 - t1 - t2)^2 is needed when t1 + t2 > 1, and also when t1 + t2 < 1 on
            signed input (else ValueError: the denominator can be <= 0).
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
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute (duplicates ignored). If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' (default) or 'coo'.
        num_threads: Number of threads to use (0 = all available cores).
        block_size: Block size for column-blocked accumulation (0=auto, None=disabled).

    Returns:
        A sparse matrix of top-k s_plus similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = _shrink_values(
        shrink, shrink_type, 's_plus', cosine=l2 != 0, denominator=l1 != 0 or l2 != 0 or l3 != 0)
    return _s_plus(
        matrix1, matrix2=matrix2,
        l1=l1, l2=l2, l3=l3,
        t1=t1, t2=t2,
        c1=c1, c2=c2,
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


def _signed_power(x: np.ndarray, alpha: float) -> np.ndarray:
    """sign(x) |x|^alpha: x^alpha bit for bit on x > 0; on signed input (D45) a negative
    transition weight keeps its sign (np.power would give NaN for a fractional alpha)."""
    return np.copysign(np.abs(x) ** alpha, x)


def _abs_col_sums(m: sparray) -> np.ndarray:
    """
    Column sums of |x| over the summed entries, as in the canonical matrix the
    kernel sees: duplicates (3, -1) count 2, not 4. scipy's abs() sums duplicates
    first, in place, hence the copy. m keeps its format: converting to CSR first
    would reorder the float additions and change the sums on non-negative input.
    """
    return np.asarray(abs(m.copy()).sum(axis=0)).ravel()


def _shrink_values(shrink: float, shrink_type: str, name: str, cosine: bool,
                   denominator: bool) -> tuple[float, float, float]:
    """
    (stabilized, bayesian, additive) shrink for a similarity. Rejects a shrink
    that would do nothing: additive is added to the cosine norms only, and
    stabilized without a normalization term divides every score by the same
    constant (ranking unchanged).
    """
    if shrink_type not in ('stabilized', 'bayesian', 'additive'):
        raise ValueError("shrink_type must be one of 'stabilized', 'bayesian', or 'additive'")
    if not (np.isfinite(shrink) and shrink >= 0):
        raise ValueError(f'shrink must be finite and >= 0, got {shrink}')
    if shrink != 0 and shrink_type == 'additive' and not cosine:
        raise ValueError(f"{name}: additive shrink only applies to a cosine term; use 'stabilized' or 'bayesian'")
    if shrink != 0 and shrink_type == 'stabilized' and not denominator:
        raise ValueError(f"{name}: stabilized shrink without a normalization only rescales the scores "
                         "(same ranking); use 'bayesian'")
    return tuple(shrink if shrink_type == t else 0.0 for t in ('stabilized', 'bayesian', 'additive'))
