# 🔍 Similarity Functions

SimilariPy provides a suite of similarity functions for sparse matrixes, all implemented in Cython and parallelized with OpenMP. These models compute item-to-item or user-to-user similarity based on vector math or graph-based transformations.

## Similarities

| Function           | Description |
|--------------------|-------------|
| `dot_product()`    | Simple raw inner product between vectors. |
| `cosine()`         | Cosine similarity with optional shrinkage. |
| `asymmetric_cosine(alpha=0.5)` | Asymmetric variant of cosine similarity, where `alpha` controls the weighting between vectors. |
| `jaccard()`        | Set-based similarity defined as the intersection over union. |
| `dice()`           | Harmonic mean of two vectors' lengths. |
| `tversky(alpha=1.0, beta=1.0)` | Tversky similarity, a generalization of Jaccard and Dice. |
| `p3alpha(alpha=1.0)` | Graph-based similarity computed as normalized matrix multiplication with `alpha` exponentiation. |
| `rp3beta(alpha=1.0, beta=1.0)` | P3alpha variant that penalizes popular items with a `beta` exponent. |
| `s_plus(l=0.5, t1=1.0, t2=1.0, c=0.5)` | Hybrid model combining Tversky and Cosine with tunable weights. |

## Common Parameters

All similarity functions in Similaripy share the following parameters:

| Parameter        | Description |
|------------------|-------------|
| `m1`             | Input sparse matrix for which to calculate the similarity. |
| `m2`             | Optional transpose matrix. If `None`, uses `m1.T`. *(default: `None`)* |
| `k`              | Number of top-k items per row. *(default: `100`)* |
| `h`              | Shrinkage coefficient applied during normalization. |
| `threshold`      | Minimum similarity value to retain. Values below are set to zero. *(default: `0`)* |
| `binary`         | If `True`, binarizes the input matrix. *(default: `False`)* |
| `target_rows`    | List or array of row indices to compute. If `None`, computes for all rows. *(default: `None`)* |
| `target_cols`    | Subset of columns to consider **before** applying top-k. Can be an array (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `filter_cols`    | Subset of columns to filter **before** applying top-k. Can be an array (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `verbose`        | If `True`, shows a progress bar. *(default: `True`)* |
| `format_output`  | Output format: `'coo'` or `'csr'`. *(default: `'coo'`)*<br/>*Note: `'csr'` not currently supported on Windows.* |
| `num_threads`    | Number of threads to use. `0` means use all available cores. *(default: `0`)* |

## Notes

- All similarity functions are implemented in **Cython + OpenMP** for high-performance computation on CSR matrixes.
- Computations are fully **multi-threaded** and scale with CPU cores.
- Supports **CSR** and **COO** sparse matrix formats as output.
- ⚠️ **Windows**: use `format_output='coo'` (CSR output is not supported on Windows due to a platform data type mismatch).

## Math Equations

### Dot Product
$s_{xy} = x \cdot y$

### Cosine
$s_{xy} = \frac{x \cdot y}{\|x\| \cdot \|y\| + h}$

### Asymmetric Cosine
$s_{xy} = \frac{x \cdot y}{\left(\sum x_i^2\right)^\alpha \left(\sum y_i^2\right)^{1 - \alpha} + h}$

- **`α`**: Asymmetry coefficient ∈ [0, 1]

### Jaccard
$s_{xy} = \frac{x \cdot y}{|x| + |y| - x \cdot y + h}$

### Dice

$s_{xy} = \frac{x \cdot y}{\frac{1}{2}|x| + \frac{1}{2}|y| - x \cdot y + h}$

### Tversky
$s_{xy} = \frac{x \cdot y}{\alpha(|x| - x \cdot y) + \beta(|y| - x \cdot y) + x \cdot y + h}$

- **`α`**, **`β`**: Tversky coefficients ∈ [0, 1]

### P3α
$s_{xy} =  x^\alpha \cdot  y^\alpha$

- **`α`**: P3α coefficient ∈ [0, 1]
- Normalizion row-wise (L1) is applied before exponentiation

### RP3β
$s_{xy} = \frac{x^\alpha \cdot y^\alpha}{{pop}(y)^\beta}$

- **`α`**: P3α coefficient ∈ [0, 1]  
- **`β`**: Popularity penalization coefficient ∈ [0, 1]
- **`pop(j)`** Number of interactions for item j
- Normalizion row-wise (L1) is applied before exponentiation
- Penalization is applied before the top k selection

### S-Plus

$s_{xy} = \frac{x \cdot y}{l \left(t_1(|x| - x \cdot y) + t_2(|y| - x \cdot y) + x \cdot y\right) + (1 - l)\left(\sum x_i^2\right)^c \left(\sum y_i^2\right)^{1 - c} + h}$

- **`l`**: Balance between Tversky and Cosine parts ∈ [0, 1]  
- **`t1`**, **`t2`**: Tversky coefficients ∈ [0, 1]  
- **`c`**: Cosine weighting exponent ∈ [0, 1]
