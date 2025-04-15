# 📘 Similaripy - Guide

This is the documentation for using Similaripy.

---

## ⚙️ Common Parameters

All similarity functions in Similaripy accept the following parameters:

| Parameter        | Description |
|------------------|-------------|
| `m1`             | Input sparse matrix for which to calculate the similarity. |
| `m2`             | Optional transpose matrix. If `None`, uses `m1.T`. *(default: `None`)* |
| `k`              | Number of top-k items per row. *(default: `100`)* |
| `h`              | Shrinkage coefficient applied during normalization. |
| `threshold`      | Minimum similarity value to retain. Values below are set to zero. *(default: `0`)* |
| `binary`         | If `True`, binarizes the input matrix. *(default: `False`)* |
| `target_rows`    | List or array of row indices to compute. If `None`, computes for all rows. *(default: `None`)* |
| `target_cols`    | Subset of columns to consider **before** applying top-k. Can be a list (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `filter_cols`    | Subset of columns to filter **before** applying top-k. Can be a list (applied to all rows) or a sparse matrix (row-specific). *(default: `None`)* |
| `verbose`        | If `True`, shows a progress bar. *(default: `True`)* |
| `format_output`  | Output format: `'coo'` or `'csr'`. *(default: `'coo'`)*<br/>*Note: `'csr'` not currently supported on Windows.* |
| `num_threads`    | Number of threads to use. `0` means use all available cores. *(default: `0`)* |

---

## 📈 Similarity Metrics

### 🔹 Core

#### Dot Product

![dot](https://latex.codecogs.com/svg.latex?&space;s_{xy}%20=%20x%20\cdot%20y)

#### Cosine

![cosine](https://latex.codecogs.com/svg.latex?&space;s_{xy}=\frac{xy}{\|x\|\|y\|+h})

#### Asymmetric Cosine

![asymcos](https://latex.codecogs.com/svg.latex?&space;s_{xy}=\frac{xy}{\left(\sum{x_{i}^{2}}\right)^{\alpha}\left(\sum{y_{i}^{2}}\right)^{1-\alpha}+h})

- **`α`**: Asymmetry coefficient ∈ [0, 1]

#### Jaccard

![jaccard](https://latex.codecogs.com/svg.latex?&space;s_{xy}=\frac{xy}{|x|+|y|-xy+h})

#### Dice

![dice](https://latex.codecogs.com/svg.latex?&space;s_{xy}=\frac{2xy}{|x|+|y|+h})

#### Tversky

![tversky](https://latex.codecogs.com/svg.latex?&space;s_{xy}=\frac{xy}{\alpha(|x|-xy)+\beta(|y|-xy)+xy+h})

- **`α`**, **`β`**: Tversky coefficients ∈ [0, 1]

---

### 🔸 Graph-Based

#### P3α

- **`α`**: P3α coefficient ∈ [0, 1]

#### RP3β

- **`α`**: P3α coefficient ∈ [0, 1]  
- **`β`**: Popularity penalization coefficient ∈ [0, 1]

---

### 🧪 Hybrid

#### S-Plus

![splus](https://latex.codecogs.com/svg.latex?&space;s_{xy}=\frac{xy}{l(t_1(|x|-xy)+t_2(|y|-xy)+xy)+(1-l)(\sum{x_{i}^{2}})^{c}(\sum{y_{i}^{2}})^{1-c}+h})

- **`l`**: Balance between Tversky and Cosine parts ∈ [0, 1]  
- **`t1`**, **`t2`**: Tversky coefficients ∈ [0, 1]  
- **`c`**: Cosine weighting exponent ∈ [0, 1]

---

## ⚡ Performance Notes

- All similarity functions are implemented in **Cython + OpenMP** for maximum speed
- Computations are fully **multi-threaded** and scale with CPU cores
- Supports **CSR** and **COO** sparse matrix formats
- Can operate in-place to reduce memory overhead

**Windows note**: use `format_output='coo'` (CSR output is not supported on Windows due to a platform limitation)

---

## 📌 Example Usage

```python
import similaripy as sim
import scipy.sparse as sps

# Create a random user-rating matrix
urm = sps.random(1000, 2000, density=0.025)

# Normalize with BM25
urm = sim.normalization.bm25(urm)

# Fit cosine similarity model
model = sim.cosine(urm.T, k=50)

# Recommend 100 items for users 1, 14, and 8
recommendations = sim.dot_product(urm, model.T, k=100, target_rows=[1, 14, 8], filter_cols=urm)