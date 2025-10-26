"""
Performance benchmarks for similaripy.

Run with: make benchmark
Or: pytest tests/benchmarks.py --benchmark-only

Compare versions:
  pytest tests/benchmarks.py --benchmark-compare=baseline --benchmark-save=new_version
"""

import numpy as np
import scipy.sparse as sp
import pytest

from similaripy import normalization, s_plus


def generate_sparse_matrix(n_rows, n_cols, density=0.01, random_state=42):
    """Generate a random sparse matrix for testing."""
    rng = np.random.default_rng(random_state)
    nnz = int(n_rows * n_cols * density)
    row = rng.integers(0, n_rows, nnz)
    col = rng.integers(0, n_cols, nnz)
    data = rng.random(nnz, dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n_rows, n_cols)).tocsr()


# Matrix sizes representing real-world use cases
MATRIX_SIZES = {
    'small': (1_000, 500, 0.05),              # Small dataset, ~25K non-zeros
    'medium': (10_000, 5_000, 0.01),          # Medium dataset, ~500K non-zeros
    'large': (30_000, 140_000, 0.005),        # Large dataset, ~20M non-zeros
    'xlarge': (1_000_000, 2_260_000, 2.9e-5)  # Extra large dataset, ~100M non-zeros
}


# ============================================================================
# NORMALIZATION BENCHMARKS
# ============================================================================

class TestNormalizationPerformance:
    """Benchmark normalization functions."""

    @pytest.mark.perf
    @pytest.mark.parametrize("size_name", ['xlarge'])
    @pytest.mark.parametrize("norm_type", ['l1', 'l2', 'max'])
    def test_basic_normalization(self, benchmark, size_name, norm_type):
        """Benchmark basic L1/L2/max normalization."""
        n_rows, n_cols, density = MATRIX_SIZES[size_name]
        mat = generate_sparse_matrix(n_rows, n_cols, density)

        result = benchmark(normalization.normalize, mat, norm=norm_type, inplace=False)

        assert result.nnz > 0
        assert result.shape == mat.shape

    @pytest.mark.perf
    @pytest.mark.parametrize("size_name", ['xlarge'])
    @pytest.mark.parametrize("tf_mode,idf_mode", [
        ('sqrt', 'smooth')
    ])
    def test_tfidf_normalization(self, benchmark, size_name, tf_mode, idf_mode):
        """Benchmark TF-IDF with different mode combinations."""
        n_rows, n_cols, density = MATRIX_SIZES[size_name]
        mat = generate_sparse_matrix(n_rows, n_cols, density)

        result = benchmark(
            normalization.tfidf,
            mat,
            tf_mode=tf_mode,
            idf_mode=idf_mode,
            inplace=True
        )

        assert result.nnz > 0
        assert result.shape == mat.shape

    @pytest.mark.perf
    @pytest.mark.parametrize("size_name", ['xlarge'])
    def test_bm25_normalization(self, benchmark, size_name):
        """Benchmark BM25 normalization."""
        n_rows, n_cols, density = MATRIX_SIZES[size_name]
        mat = generate_sparse_matrix(n_rows, n_cols, density)

        result = benchmark(
            normalization.bm25,
            mat,
            k1=1.2,
            b=0.75,
            inplace=True
        )

        assert result.nnz > 0
        assert result.shape == mat.shape


# ============================================================================
# S_PLUS BENCHMARKS
# ============================================================================

class TestSPlusPerformance:
    """Benchmark s_plus similarity computation."""

    @pytest.mark.perf
    @pytest.mark.parametrize("n_rows,n_cols,density,k", [
        (6_000, 4_000, 0.04, 50),        # MovieLens1M-like
        (30_000, 140_000, 0.005, 50),    # MovieLens20M-like
        (1_000_000, 2_260_000, 2.9e-5, 50)  # 100M-like
    ])
    def test_s_plus_basic(self, benchmark, n_rows, n_cols, density, k):
        """Benchmark s_plus with realistic matrix sizes."""
        mat = generate_sparse_matrix(n_rows, n_cols, density)

        result = benchmark(
            s_plus,
            mat,
            k=k,
            verbose=False,
            binary=False,
            format_output='csr',
            num_threads=0  # use all available cores
        )

        assert result.nnz > 0
        assert result.shape[0] == n_rows

    @pytest.mark.perf
    @pytest.mark.parametrize("n_rows,n_cols,density,k", [
        (6_000, 4_000, 0.04, 50),        # MovieLens1M-like
        (30_000, 140_000, 0.005, 50),    # MovieLens20M-like
        (1_000_000, 2_260_000, 2.9e-5, 50)  # 100M-like
    ])
    def test_s_plus_complex(self, benchmark, n_rows, n_cols, density, k):
        """Benchmark s_plus with complex normalization parameters."""
        mat = generate_sparse_matrix(n_rows, n_cols, density)

        result = benchmark(
            s_plus,
            mat,
            l1=0.5,
            l2=0.5,
            l3=1.0,
            k=k,
            alpha=1,
            pop2='sum',
            shrink=5,
            shrink_type='bayesian',
            verbose=False,
            binary=False,
            format_output='csr',
            num_threads=0
        )

        assert result.nnz > 0
        assert result.shape[0] == n_rows
