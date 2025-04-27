from .cython_code import s_plus as _sim
from .normalization import normalize as _normalize
import numpy as np

from typing import Optional, Union, Literal
from scipy.sparse import spmatrix


def dot_product(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute dot product similarity between rows of matrix1 and columns of matrix2.

    Args:
        matrix1: Input sparse matrix (e.g., user-item or item-user).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before similarity computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format: 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 means all available cores).

    Returns:
        A sparse matrix of shape (n_rows, n_cols) in the specified format,
        containing the top-k dot product similarities.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def cosine(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute cosine similarity between sparse vectors.

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k cosine similarities in the specified format.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=0.5, c2=0.5,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def asymmetric_cosine(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 0.5,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute asymmetric cosine similarity.

    Args:
        matrix1: Input sparse matrix (e.g., user-item or item-user).
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Controls asymmetry in cosine weighting.
               `alpha=1` weighs only matrix1; `alpha=0.5` is symmetric.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of shape (n_rows, n_cols) containing the top-k
        asymmetric cosine similarities in the specified format.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l2=1,
        c1=alpha, c2=1-alpha,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def tversky(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute Tversky similarity between sparse vectors.

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Tversky weight for elements unique to matrix1.
        beta: Tversky weight for elements unique to matrix2.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k Tversky similarities in the specified format.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=alpha, t2=beta,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def jaccard(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute Jaccard similarity (intersection over union).

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k Jaccard similarities in the specified format.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=1, t2=1,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def dice(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute Dice similarity (harmonic mean of overlap and size).

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k Dice similarities in the specified format.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=1,
        t1=0.5, t2=0.5,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def p3alpha(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute P3alpha similarity using a normalized 3-step random walk.

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Exponent for transition probabilities to control popularity effect.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k P3alpha similarities in the specified format.
    """
    if matrix2 is None:
        matrix2 = matrix1.T
    matrix1 = _normalize(matrix1, norm='l1', axis=1, inplace=False)
    matrix1.data = np.power(matrix1.data, alpha)
    matrix2 = _normalize(matrix2, norm='l1', axis=1, inplace=False)
    matrix2.data = np.power(matrix2.data, alpha)
    return _sim.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def rp3beta(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute RP3beta similarity: P3alpha with popularity penalization.

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        alpha: Exponent for transition probabilities.
        beta: Exponent to penalize popularity based on column sums.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k RP3beta similarities in the specified format.
    """
    if matrix2 is None:
        matrix2 = matrix1.T
    pop_m2 = matrix2.sum(axis=0).A1
    matrix1 = _normalize(matrix1, norm='l1', axis=1, inplace=False)
    matrix1.data = np.power(matrix1.data, alpha)
    matrix2 = _normalize(matrix2, norm='l1', axis=1, inplace=False)
    matrix2.data = np.power(matrix2.data, alpha)
    return _sim.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        weight_depop_matrix2=pop_m2,
        p2=beta,
        l3=1,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )


def s_plus(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    l: float = 0.5,
    t1: float = 1.0,
    t2: float = 1.0,
    c: float = 0.5,
    k: int = 100,
    shrink: float = 0.0,
    threshold: float = 0.0,
    binary: bool = False,
    target_rows: Optional[Union[list[int], np.ndarray]] = None,
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Compute hybrid S Plus similarity with weighted Tversky and Cosine components.

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        l: Mixing parameter between Tversky (l1) and Cosine (l2).
        t1: Tversky alpha for matrix1.
        t2: Tversky beta for matrix2.
        c: Cosine exponent coefficient.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value applied to similarity scores.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k similarities based on combined Tversky and Cosine scoring.
    """
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=l, l2=1-l,
        t1=t1, t2=t2,
        c1=c, c2=1-c,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_rows=target_rows,
        target_cols=target_cols,
        filter_cols=filter_cols,
        verbose=verbose,
        format_output=format_output,
        num_threads=num_threads
    )
