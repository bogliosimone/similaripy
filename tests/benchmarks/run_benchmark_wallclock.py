"""
Single cosine benchmark on Yamba 50M for wall-clock comparison.
"""
import sys
import time
from pathlib import Path

from dataset_loaders import load_URM
import similaripy as sim

print("="*70)
print("SINGLE COSINE BENCHMARK - Yamba 50M")
print("="*70)

# Load dataset
print("\nLoading dataset...")
URM, metadata = load_URM('yambda', version='50m', verbose=False)
print(f"Dataset: {URM.shape}, NNZ: {URM.nnz:,}")

# Transpose for item-item
item_matrix = URM.T

# Run benchmark
print("\nRunning cosine similarity...")
start = time.perf_counter()

similarity_matrix = sim.cosine(
    item_matrix,
    k=100,
    shrink=0,
    threshold=0,
    verbose=False,
    format_output='csr',
    num_threads=0
)

end = time.perf_counter()
elapsed = end - start

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Wall-clock time:     {elapsed:.2f}s")
print(f"Throughput:          {item_matrix.shape[0] / elapsed:.1f} items/s")
print(f"Result shape:        {similarity_matrix.shape}")
print(f"Result NNZ:          {similarity_matrix.nnz:,}")
print("="*70)
