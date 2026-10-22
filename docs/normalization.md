# 🧮 Normalization Functions

SimilariPy includes several normalization functions designed for sparse matrix pre-processing. All functions return a new CSR matrix (the input never changes): float32 or float64 values as the input (other types become float32), int32 indices, duplicates summed, no explicit zeros.

## Normalizations

| Function        | Description |
|-----------------|-------------|
| `binarize(X)` | Sets non-zero entries to 1.0 and drops explicit zeros, as `binary=True` in the similarity functions, but as a separate step. |
| `normalize(X, norm='l2')` | Standard row or column-wise normalization. Supports `'l1'`, `'l2'`, and `'max'`. |
| `tfidf(X, tf_mode='sqrt', idf_mode='smooth')` | TF-IDF weighting with customizable term-frequency and inverse-document-frequency modes. |
| `bm25(X, k1=1.2, b=0.75)` | BM25 weighting, a standard IR normalization used for relevance scoring. |
| `bm25plus(X, k1=1.2, b=0.75, delta=1.0)` | BM25+ variant with an additional smoothing `delta` parameter. |

## Common Parameters

All normalization functions in SimilariPy share the following parameters:

| Parameter     | Description |
|---------------|-------------|
| `axis`        | `1` for row-wise (default), `0` for column-wise normalization. |
| `logbase`     | Base of the logarithm (e.g. `e`, `2`) for TF-IDF and BM25. |
| `tf_mode`     | Term frequency transformation mode for TF-IDF and BM25 (see TF table). |
| `idf_mode`    | Inverse document frequency mode for TF-IDF and BM25 (see IDF table). |

## TF Modes

| Mode     | Description |
|----------|-------------|
| `'binary'` | 1 if non-zero |
| `'raw'`    | Raw frequency |
| `'sqrt'`   | √(raw frequency) |
| `'freq'`   | Row-normalized frequency |
| `'log'`    | log(1 + frequency) |

## IDF Modes

| Mode     | Description |
|----------|-------------|
| `'unary'`  | No IDF applied |
| `'base'`   | log(N / df) |
| `'smooth'` | log(1 + N / df) |
| `'prob'`   | log((N - df) / df) |
| `'bm25'`   | BM25-style IDF weighting |

## Signed Input

`normalize` and `binarize` accept negative values; `tfidf`, `bm25` and
`bm25plus` raise a `ValueError`: see [Signed Input](signed-input.md#normalizations).

## Notes

- All normalization functions return a new matrix: use the return value, `X = bm25(X)`.
- Input values must be finite (NaN or inf raises a `ValueError`), as in the similarity functions.
- `bm25` and `tfidf` are ideal for text, user-item, or interaction data.
