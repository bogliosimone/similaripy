"""
Minimal script for C++ profiling with py-spy.
Run: sudo py-spy record -o flamegraph.svg --native -- uv run python profile_cpp.py
"""

import sys
from pathlib import Path

# Add tests/benchmarks to path for dataset_loaders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmarks'))

from dataset_loaders import load_URM
import similaripy as sim

print("Loading Yamba 50M dataset...")
URM, metadata = load_URM('yambda', version='50m', verbose=False)
print(f"Dataset loaded: {URM.shape}, NNZ: {URM.nnz:,}")

# Transpose for item-item similarity
item_matrix = URM.T
print(f"Item matrix: {item_matrix.shape}")

print("\nRunning cosine similarity (this will be profiled)...")
print("=" * 70)

# Run cosine similarity - this is what gets profiled
similarity_matrix = sim.cosine(
    item_matrix,
    k=100,
    shrink=0,
    threshold=0,
    verbose=True,
    format_output='csr',
    num_threads=0
)

print("=" * 70)
print(f"Done! Similarity matrix: {similarity_matrix.shape}, NNZ: {similarity_matrix.nnz:,}")
