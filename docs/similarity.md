# 🔍 Similarity Functions

SimilariPy provides a suite of similarity functions for sparse matrices, all running in a multi-threaded Rust core. These models compute item-to-item or user-to-user similarity based on vector math or graph-based transformations.

## Similarities

| Function           | Description |
|--------------------|-------------|
| `dot_product()`    | Simple raw inner product between vectors. |
| `cosine()`         | Cosine similarity with optional shrinkage. |
| `asymmetric_cosine(alpha=0.5)` | Asymmetric variant of cosine similarity, where `alpha` controls the weighting between vectors. |
| `jaccard()`        | Set-based similarity defined as the intersection over union. |
| `dice()`           | Overlap over the mean of the two vectors' lengths (Tversky with `alpha = beta = 0.5`). |
| `tversky(alpha=1.0, beta=1.0)` | Tversky similarity, a generalization of Jaccard and Dice. |
| `p3alpha(alpha=1.0)` | Graph-based similarity computed as normalized matrix multiplication with `alpha` exponentiation. |
| `rp3beta(alpha=1.0, beta=1.0)` | P3alpha variant that penalizes popular items with a `beta` exponent. |
| `s_plus(l1=0.5, l2=0.5, l3=0.0, t1=1.0, t2=1.0, c1=0.5, c2=0.5, alpha=1, pop1='none', pop2='none', beta1=0.0, beta2=0.0)` | Hybrid similarity model that combines Tversky and Cosine normalizations with RP3Beta-style depopularization, controlled by tunable weights. The `pop1` and `pop2` parameters define item popularity weights and may be provided as custom arrays of arbitrary values, or initialized with built-in options: `'sum'`: use the sum of `|x|` per item (the interaction count on binary input); `'none'`: disable popularity weighting  *(default: `'none'`)* |

## Common Parameters

All similarity functions in Similaripy share the following parameters (everything after `matrix2` is keyword-only):

| Parameter        | Description |
|------------------|-------------|
| `matrix1`        | Input sparse matrix for which to calculate the similarity. |
| `matrix2`        | Optional second matrix. If `None`, uses `matrix1.T`. *(default: `None`)* |
| `k`              | Number of top-k items per row. *(default: `100`)* |
| `shrink`         | Shrinkage strength `h` (`>= 0`), see [Shrinkage Types](#shrinkage-types). *(default: `0`)* |
| `threshold`      | Minimum similarity value to retain: scores `>= threshold` are kept, lower ones are dropped; a score of exactly `0` is always dropped. *(default: `0`)* |
| `shrink_type`    | Shrinkage type: `stabilized`, `bayesian`, or `additive`.  *(default: `stabilized`)* |
| `binary`         | If `True`, binarizes the input matrix. *(default: `False`)* |
| `target_rows`    | List or array of row indices to compute. If `None`, computes for all rows. *(default: `None`)* |
| `target_cols`    | Subset of columns to consider **before** applying top-k. Can be an array (applied to all rows) or a sparse matrix (row-specific). `None` = all columns; an empty selector raises `ValueError`. *(default: `None`)* |
| `filter_cols`    | Subset of columns to filter **before** applying top-k. Can be an array (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `verbose`        | If `True`, shows a progress bar. *(default: `False`)* |
| `format_output`  | Output format: `'csr'` or `'coo'`. *(default: `'csr'`)* |
| `num_threads`    | Number of threads to use. `0` = all cores, on one shared thread pool (concurrent calls queue on it; `RAYON_NUM_THREADS` caps it). *(default: `0`)* |
| `block_size`     | Block size for column-blocked accumulation. `'auto'` (recommended: the library's choice, currently 262144 float32 entries, ~1 MB; may adapt in future versions), `None` = disabled, or an int > 0. Results are the same at any size. *(default: `'auto'`)* |

## Shrinkage Types

The shrinkage equations are displayed with the cosine normalization for simplicity; `h` is the `shrink` parameter. Not every type applies to every similarity: a shrink that would do nothing raises a `ValueError`.

| Shrink type | Applies to |
|-------------|------------|
| `stabilized` | Every similarity with a denominator: all but `dot_product`, `p3alpha` and `s_plus` with `l1 = l2 = l3 = 0` (there it would divide every score by the same constant). |
| `bayesian` | Every similarity (non-negative input only: see [Signed Input](#signed-input)). |
| `additive` | The cosine norms: `cosine`, `asymmetric_cosine`, `s_plus` with `l2 != 0`. |

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
$s_{xy} = \frac{x \cdot y}{\sqrt{\sum_i x_i^2 + h} \cdot \sqrt{\sum_i y_i^2 + h}}$

- Penalizes similarities with items with low support.
- Adds shrinkage directly into the cosine denominator norms.
- **`h`** acts as the shrinkage strength.
- `shrink_type = 'additive'`

## Signed Input

Negative values are accepted (see [Signed Input](guide.md#signed-input)); only
positive scores are results. Per function:

| Function | On signed input |
|----------|-----------------|
| `dot_product`, `cosine`, `asymmetric_cosine` | Accepted. `shrink_type='bayesian'` raises: `xy / (xy + h)` would flip the sign of negative scores. |
| `jaccard`, `dice`, `tversky` | Accepted; the norms are sums of `x²`. `tversky` needs `alpha + beta = 1` or `4·alpha·beta > (1 - alpha - beta)²` (the denominator then stays `> 0` on every pair); on weighted input this binds when `alpha + beta > 1`, on signed input also when `alpha + beta < 1` (binary and 0/1 data are exempt; in `s_plus` only the tversky term is checked). Bayesian shrink raises. |
| `p3alpha`, `rp3beta` | Accepted, as an extension (a signed walk): rows are l1-normalized by the sum of `|x|`, so likes and dislikes share the walk's budget, and the power keeps the sign, `sign(p)·|p|^alpha`. RP3β's popularity is the column sum of `|x|`. Bayesian shrink raises, so `p3alpha` takes no shrink on signed input. |
| `s_plus` | Accepted with `alpha = 1` (another power of a negative score is NaN or positive) and no bayesian shrink; the `t1`/`t2` rule above applies when `l1 != 0`; `pop='sum'` is the sum of `|x|`. |

With `binary=True` the input is never signed (every non-zero is `1`), so none
of these restrictions apply.

## Notes

- All similarity functions run in a **Rust core** on CSR matrices.
- Computations are fully **multi-threaded** and scale with CPU cores.
- Supports **CSR** and **COO** sparse matrix formats as output.
- **Accuracy:** values and sums are float32. On the 500 longest MovieLens 32M rows (14k–103k entries, item-item cosine), 99.996% of the top-100 slots equal an exact float64 computation, and scores are within about `1e-4` relative.

## Math Equations

### Dot Product
$s_{xy} = x \cdot y$

### Cosine
$s_{xy} = \frac{x \cdot y}{\|x\| \cdot \|y\|}$

### Asymmetric Cosine
$s_{xy} = \frac{x \cdot y}{\left(\sum x_i^2\right)^\alpha \left(\sum y_i^2\right)^{1 - \alpha}}$

- **`α`**: Asymmetry coefficient ∈ [0, 1]

### Jaccard

In Jaccard, Dice, Tversky and S-Plus, $|x| = \sum_i x_i^2$ (the number of non-zeros on binary data).

$s_{xy} = \frac{x \cdot y}{|x| + |y| - x \cdot y}$

### Dice

$s_{xy} = \frac{x \cdot y}{\frac{1}{2}|x| + \frac{1}{2}|y|}$

### Tversky
$s_{xy} = \frac{x \cdot y}{\alpha(|x| - x \cdot y) + \beta(|y| - x \cdot y) + x \cdot y}$

- **`α`**, **`β`**: Tversky coefficients `>= 0` (on weighted input not every pair is allowed: see [Signed Input](#signed-input))

### P3α
$s_{xy} =  x^\alpha \cdot  y^\alpha$

- **`α`**: P3α coefficient `> 0`
- Normalization row-wise (L1) is applied before exponentiation

### RP3β
$s_{xy} = \frac{x^\alpha \cdot y^\alpha}{{pop}(y)^\beta}$

- **`α`**: P3α coefficient `> 0`
- **`β`**: Popularity penalization coefficient `>= 0`
- **`pop(y)`**: popularity of item y, the column sum of `|x|` in `matrix2` before the row normalization (the number of interactions with `binary=True`)
- Normalization row-wise (L1) is applied before exponentiation
- Penalization is applied before the top k selection

### S-Plus

$s_{xy} = \frac{(x \cdot y)^\alpha}{l_1 \left(t_1(|x| - x \cdot y) + t_2(|y| - x \cdot y) + x \cdot y\right) + l_2\left(\sum x_i^2\right)^{c_1} \cdot \left(\sum y_i^2\right)^{c_2} + l_3(pop_1(x)^{\beta_1} \cdot pop_2(y)^{\beta_2})}$

- **`l1`**, **`l2`**: Tversky, Cosine normalization strength `>= 0`
- **`l3`**: Popularity penalization strength `>= 0`
- **`t1`**, **`t2`**: Tversky coefficients `>= 0` (same rule as `tversky` when `l1 != 0`)
- **`c1`**, **`c2`**: Cosine weighting exponents `>= 0`
- **`α`**: Power of the raw overlap `x · y`, `> 0` (different from P3α)
- **`β1`**, **`β2`**: Popularity penalization coefficient for the item x/y `>= 0`
- **`pop(x)`**, **`pop(y)`**: Popularity value for the item x/y
