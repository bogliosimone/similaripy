# <p align="center"><img src="logo.png" alt="SimilariPy Logo" width="300"/></p>

[![PyPI version](https://img.shields.io/pypi/v/similaripy.svg?logo=pypi&logoColor=white)](https://pypi.org/project/similaripy/)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bogliosimone.github.io/similaripy/)
[![Build and Test](https://github.com/bogliosimone/similaripy/actions/workflows/python-package.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/python-package.yml)
[![Publish to PyPI](https://github.com/bogliosimone/similaripy/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/pypi-publish.yml)
[![Docs Status](https://github.com/bogliosimone/similaripy/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/bogliosimone/similaripy/actions/workflows/deploy-docs.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/similaripy.svg?logo=python&logoColor=white)](https://pypi.org/project/similaripy/)
[![License](https://img.shields.io/github/license/bogliosimone/similaripy.svg)](https://github.com/bogliosimone/similaripy/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2583851.svg)](https://doi.org/10.5281/zenodo.2583851)

## Welcome to SimilariPy’s documentation!

**SimilariPy** is a Python package that provides high-performance KNN similarity and normalization functions, optimized for sparse matrices.

---

## 🔥 Why SimilariPy?

- **Fast**: a multi-threaded Rust core, GIL-free
- **Sparse matrix focus**: any SciPy sparse input, CSR or COO output
- **Wide similarity coverage**: Dot Product, Cosine, Tversky, P3α, RP3β, and more
- **Flexible normalization**: includes L1, L2, TF-IDF, BM25, BM25+

---

## 📚 Guide

- [Installation instructions](installation.md)
- [Quickstart guide](guide.md)
- [Quick reference to similarity functions](similarity.md)
- [Quick reference to normalization functions](normalization.md)
- [Migration from 0.7.0](migration.md)
- [Full API documentation](api/similarity.md)

---

## ⚡ Key Performance Optimizations

SimilariPy achieves high performance through several key optimizations:

- **Rust core**: compiled to native code, multi-threaded with rayon
- **GIL-free computation**: releases Python's Global Interpreter Lock during the heavy computation (free-threaded Python supported)
- **CSR-optimized**: direct memory access patterns for the Compressed Sparse Row format
- **Column-blocked accumulation**: tiles the output columns into cache-sized blocks with popularity-based reordering, reducing L2 cache misses on large matrices
- **Pre-allocated buffers**: per-thread accumulators and heaps reused across rows, no allocation per row
- **Float32 precision**: 32-bit values for 2× memory bandwidth (accuracy: see [Similarities](similarity.md#notes))
- **Top-K filtering**: a heap per row keeps only the best `k` candidates

---
