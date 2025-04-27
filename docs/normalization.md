# 🧮 Normalization Functions

SimilariPy includes several normalization functions designed for sparse matrix pre-processing. All functions are implemented in Cython and support in-place operation for memory efficiency.

## Normalizations

| Function        | Description |
|-----------------|-------------|
| `normalize(X, norm='l2')` | Standard row or column-wise normalization. Supports `'l1'`, `'l2'`, and `'max'`. |
| `tfidf(X, tf_mode='sqrt', idf_mode='smooth')` | TF-IDF weighting with customizable term-frequency and inverse-document-frequency modes. |
| `bm25(X, k1=1.2, b=0.75)` | BM25 weighting, a standard IR normalization used for relevance scoring. |
| `bm25plus(X, k1=1.2, b=0.75, delta=1.0)` | BM25+ variant with an additional smoothing `delta` parameter. |

## Common Parameters

All normalization functions in SimilariPy share the following parameters:

| Parameter     | Description |
|---------------|-------------|
| `axis`        | `1` for row-wise (default), `0` for column-wise normalization. |
| `inplace`     | If `True`, modifies the input matrix in-place. |
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

## Notes

- All normalization functions can operate in-place on **CSR** format to reduce memory overhead.
- `bm25` and `tfidf` are ideal for text, user-item, or interaction data.
