# ➕➖ Signed Input

Input values can be negative, e.g. mean-centered ratings (adjusted cosine) or
explicit dislikes. Most recommender data (clicks, plays, counts) is
non-negative and needs none of this page.

A score is evidence of similarity: only **positive** scores (`>= threshold`)
are results; negative scores and scores that cancel to exactly `0` are
dropped, so a row with only negative scores comes back empty.

## Example: adjusted cosine

```python
import numpy as np
import scipy.sparse as sps

import similaripy as sim

urm = sps.random_array((1000, 2000), density=0.025, format="csr", dtype=np.float32)

# Center each user's ratings (on a float copy: integer ratings can't be centered in place)
urm = sps.csr_array(urm, dtype=np.float32)
counts = np.diff(urm.indptr)
sums = np.ravel(urm.sum(axis=1))
means = np.divide(sums, counts, out=np.zeros(urm.shape[0]), where=counts > 0)
urm.data -= np.repeat(means, counts)

# Then item-item cosine; a small threshold drops float32 cancellation residues
similarity_matrix = sim.cosine(urm.T, k=50, threshold=1e-6)
```

## Tips

- **Dissimilar items** are not returned. To rank them, negate `matrix2`:
  `sim.cosine(m, -m.T)` keeps the most negative scores, as positives.
- **`threshold > 0`** is a good idea: sums that should cancel can leave a tiny
  float32 residue (e.g. `1e-8`) instead of an exact `0`. On float data in any
  similarity; for `cosine`, `asymmetric_cosine` and `rp3beta` also on integer
  data (they run as a dot product of pre-scaled inputs, whose rounding doesn't
  cancel exactly).
- **Selectors:** `filter_cols` / `target_cols` matrices select the non-zero
  entries. Pass the raw (or binarized) interaction matrix, not a mean-centered
  one: a rating equal to the user's mean centers to `0` and would not be
  filtered. Duplicate entries that cancel (e.g. `+1` and `-1`) select nothing.
- **`binary=True`** makes the input non-signed (every non-zero is `1`), so none
  of the rules below apply.

## Similarities

| Function | On signed input |
|----------|-----------------|
| `dot_product`, `cosine`, `asymmetric_cosine` | Accepted. `shrink_type='bayesian'` raises: `xy / (xy + h)` would flip the sign of negative scores. |
| `jaccard`, `dice`, `tversky` | Accepted; the norms are sums of `x²`. `tversky`'s `alpha`/`beta` pair rule (see [Tversky](similarity.md#tversky)) also binds when `alpha + beta < 1`. Bayesian shrink raises. |
| `p3alpha`, `rp3beta` | Accepted, as an extension (a signed walk): rows are l1-normalized by the sum of `|x|`, so likes and dislikes share the walk's budget, and the power keeps the sign, `sign(p)·|p|^alpha`. RP3β's popularity is the column sum of `|x|`. Bayesian shrink raises, so `p3alpha` takes no shrink on signed input. |
| `s_plus` | Accepted with `alpha = 1` (another power of a negative score is NaN or positive) and no bayesian shrink; the `t1`/`t2` pair rule applies when `l1 != 0`; `pop='sum'` is the sum of `|x|`. |

## Normalizations

| Function | On signed input |
|----------|-----------------|
| `normalize` | Accepted: `'l1'` divides by the sum of `|x|`, `'l2'` by `sqrt(sum of x²)`, `'max'` by the largest `|x|`; signs are kept. |
| `binarize` | Accepted: a negative value is an interaction too and becomes `1.0`. |
| `tfidf`, `bm25`, `bm25plus` | `ValueError`: they need term frequencies (a negative one gives a NaN log or a negative document length). Checked after summing duplicates, so a `+1`/`-1` pair (sum `0`) is just dropped. |
