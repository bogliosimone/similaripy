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
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
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
        num_threads=num_threads
    )


def cosine(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
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
        num_threads=num_threads
    )


def asymmetric_cosine(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 0.5,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
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
        num_threads=num_threads
    )


def tversky(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
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
        num_threads=num_threads
    )


def jaccard(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
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
        num_threads=num_threads
    )


def dice(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
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
        num_threads=num_threads
    )


def p3alpha(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
        matrix1=matrix1, matrix2=matrix2,
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
        num_threads=num_threads
    )


def rp3beta(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 100,
    shrink: float = 0.0,
    shrink_type: Literal['stabilized', 'bayesian', 'additive'] = 'stabilized',
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
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
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
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
        matrix1=matrix1, matrix2=matrix2,
        weight_depop_matrix2=pop_m2,
        p2=beta,
        l3=1,
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
        num_threads=num_threads
    )


def s_plus(
    matrix1: spmatrix,
    matrix2: Optional[spmatrix] = None,
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
    target_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    filter_cols: Optional[Union[list[int], np.ndarray, spmatrix]] = None,
    verbose: bool = True,
    format_output: Literal['csr', 'coo'] = 'coo',
    num_threads: int = 0
) -> spmatrix:
    """
    Combines Tversky and Cosine normalizations with RP3Beta-style depopularization, fully controlled by tunable weights

    Args:
        matrix1: Input sparse matrix.
        matrix2: Optional second matrix. If None, uses matrix1.T.
        l1: Tversky normalization strength.
        l2: Cosine normalization strength.
        l3: Popularity penalization strength.
        t1: Tversky alpha for matrix1.
        t2: Tversky beta for matrix2.
        c1: Cosine exponent coefficient for matrix1.
        c2: Cosine exponent coefficient for matrix2.
        pop1: Popularity weights for matrix1. 'none', 'sum', or custom array.
        pop2: Popularity weights for matrix2. 'none', 'sum', or custom array.
        alpha: Coefficient applied on the raw similarity value before normalizations.
        beta1: Popularity penalization coefficient for matrix1 items.
        beta2: Popularity penalization coefficient for matrix2 items.
        k: Number of top-k items per row to keep.
        shrink: Shrinkage value that prevents instability when normalizations are small.
        shrink_type: Type of shrinkage: 'stabilized', 'bayesian', 'additive'.
        threshold: Minimum similarity value to retain.
        binary: Whether to binarize the input matrix before computation.
        target_rows: List or array of row indices to compute. If None, computes all.
        target_cols: Columns to include before top-k. Can be a list or sparse mask matrix.
        filter_cols: Columns to exclude before top-k. Can be a list or sparse mask matrix.
        verbose: Whether to show a progress bar.
        format_output: Output format, either 'csr' or 'coo'. Use 'coo' on Windows.
        num_threads: Number of threads to use (0 = all available cores).

    Returns:
        A sparse matrix of top-k s_plus similarities in the specified format.
    """
    stabilized_shrink, bayesian_shrink, additive_shrink = __get_shrink_values__(shrink, shrink_type)
    return _sim.s_plus(
        matrix1, matrix2=matrix2,
        l1=l1, l2=l2,l3=l3,
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
        num_threads=num_threads
    )


def __get_shrink_values__(shrink: float, shrink_type: str):
    stabilized_shrink = 0.0
    bayesian_shrink = 0.0
    additive_shrink = 0.0
    if shrink_type == 'stabilized':
        stabilized_shrink = shrink
    elif shrink_type == 'bayesian':
        bayesian_shrink = shrink
    elif shrink_type == 'additive':
        additive_shrink = shrink
    else:
        raise ValueError("shrink_type must be one of 'stabilized', 'bayesian', or 'additive'")
    return stabilized_shrink, bayesian_shrink, additive_shrink
