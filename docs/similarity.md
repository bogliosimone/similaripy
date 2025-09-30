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
| `s_plus(l1=0.5, l2=0.5, l3=0.0, t1=1.0, t2=1.0, c1=0.5, c2=0.5, alpha=1, pop1='none', pop2='none', beta1=0.0, beta2=0.0)` | Hybrid similarity model that combines Tversky and Cosine normalizations with RP3Beta-style depopularization, controlled by tunable weights. The `pop1` and `pop2` parameters define item popularity weights and may be provided as custom arrays of arbitrary values, or initialized with built-in options: `'sum'`: use the sum of interactions per item; `'none'`: disable popularity weighting  *(default: `'none'`)* |

## Common Parameters

All similarity functions in Similaripy share the following parameters:

| Parameter        | Description |
|------------------|-------------|
| `m1`             | Input sparse matrix for which to calculate the similarity. |
| `m2`             | Optional transpose matrix. If `None`, uses `m1.T`. *(default: `None`)* |
| `k`              | Number of top-k items per row. *(default: `100`)* |
| `h`              | Shrinkage coefficient applied during normalization. |
| `threshold`      | Minimum similarity value to retain. Values below are set to zero. *(default: `0`)* |
| `shrink_type`    | Shrinkage type: `stabilized`, `bayesian`, or `additive`.  *(default: `stabilized`)* |
| `binary`         | If `True`, binarizes the input matrix. *(default: `False`)* |
| `target_rows`    | List or array of row indices to compute. If `None`, computes for all rows. *(default: `None`)* |
| `target_cols`    | Subset of columns to consider **before** applying top-k. Can be an array (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `filter_cols`    | Subset of columns to filter **before** applying top-k. Can be an array (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `verbose`        | If `True`, shows a progress bar. *(default: `True`)* |
| `format_output`  | Output format: `'coo'` or `'csr'`. *(default: `'coo'`)*<br/>*Note: `'csr'` not currently supported on Windows.* |
| `num_threads`    | Number of threads to use. `0` means use all available cores. *(default: `0`)* |

## Shrinkage Types

The shrinkage equations are displayed with the cosine normalization for simplicity, however, they are available in all the similarities.

### Stabilized Shrinkage
$s_{xy} = \frac{x \cdot y}{\sqrt{\sum_i x_i^2} \cdot \sqrt{\sum_i y_i^2} + h}$

- Prevents instability when norms are small.
- **`h`** acts as the shrinkage strength.
- `shrink_type = 'stabilized'`

### Bayesian Shrinkage
$s_{xy} = \frac{x \cdot y}{\sqrt{\sum_i x_i^2} \cdot \sqrt{\sum_i y_i^2}} \cdot \frac{x \cdot y}{x \cdot y + h}$

- Penalizes similarities with items with low overlap support.  
- **`h`** acts as the shrinkage strength.
- `shrink_type = 'bayesian'`

### Additive Shrinkage
$s_{xy} = \frac{x \cdot y}{\sqrt{\sum_i (x_i^2 + h)} \cdot \sqrt{\sum_i (y_i^2 + h)}}$

- Penalizes similarities with items with low support.
- Adds shrinkage directly into the cosine denominator norms.
- **`h`** acts as the shrinkage strength.
- `shrink_type = 'additive'`

## Notes

- All similarity functions are implemented in **Cython + OpenMP** for high-performance computation on CSR matrixes.
- Computations are fully **multi-threaded** and scale with CPU cores.
- Supports **CSR** and **COO** sparse matrix formats as output.
- ⚠️ **Windows**: use `format_output='coo'` (CSR output is not supported on Windows due to a platform data type mismatch).

## Math Equations

### Dot Product
$s_{xy} = x \cdot y$

### Cosine
$s_{xy} = \frac{x \cdot y}{\|x\| \cdot \|y\|}$

### Asymmetric Cosine
$s_{xy} = \frac{x \cdot y}{\left(\sum x_i^2\right)^\alpha \left(\sum y_i^2\right)^{1 - \alpha}}$

- **`α`**: Asymmetry coefficient ∈ [0, 1]

### Jaccard
$s_{xy} = \frac{x \cdot y}{|x| + |y| - x \cdot y}$

### Dice

$s_{xy} = \frac{x \cdot y}{\frac{1}{2}|x| + \frac{1}{2}|y| - x \cdot y}$

### Tversky
$s_{xy} = \frac{x \cdot y}{\alpha(|x| - x \cdot y) + \beta(|y| - x \cdot y) + x \cdot y}$

- **`α`**, **`β`**: Tversky coefficients ∈ [0, 1]

### P3α
$s_{xy} =  x^\alpha \cdot  y^\alpha$

- **`α`**: P3α coefficient ∈ [0, 1]
- Normalizion row-wise (L1) is applied before exponentiation

### RP3β
$s_{xy} = \frac{x^\alpha \cdot y^\alpha}{{pop}(y)^\beta}$

- **`α`**: P3α coefficient ∈ [0, 1]  
- **`β`**: Popularity penalization coefficient ∈ [0, 1]
- **`pop(y)`** Number of interactions for item y
- Normalizion row-wise (L1) is applied before exponentiation
- Penalization is applied before the top k selection

### S-Plus

$s_{xy} = \frac{(x \cdot y)^\alpha}{l_1 \left(t_1(|x| - x \cdot y) + t_2(|y| - x \cdot y) + x \cdot y\right) + l_2\left(\sum x_i^2\right)^{c_1} \cdot \left(\sum y_i^2\right)^{c_2} + l_3(pop_1(x)^{\beta_1} \cdot pop_2(y)^{\beta_2})}$

- **`l1`**, **`l2`**: Tversky,Cosine normalization strength ∈ [0, 1]
- **`l3`**: Popularity penalization strength ∈ [0, 1]  
- **`t1`**, **`t2`**: Tversky coefficients ∈ [0, 1]  
- **`c1`**, **`c2`**: Cosine weighting exponent ∈ [0, 1]
- **`α`**: Coefficient for the raw interaction overlap (different from P3α)
- **`β1`**, **`β2`**: Popularity penalization coefficient for the item x/y ∈ [0, 1]
- **`pop(x)`**, **`pop(y)`**: Popularity value for the item x/y
