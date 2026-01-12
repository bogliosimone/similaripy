"""
Profile cosine similarity on Yamba 50M dataset.

This script profiles a single run of cosine similarity to identify performance bottlenecks.
"""

import cProfile
import pstats
import sys
from pathlib import Path

# Add tests/benchmarks to path for dataset_loaders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmarks'))

from dataset_loaders import load_URM
import similaripy as sim


def profile_cosine_yamba50m():
    """Profile cosine similarity on Yamba 50M."""
    print("="*70)
    print("PROFILING: Cosine Similarity on Yamba 50M")
    print("="*70)

    # Load dataset
    print("\nLoading Yamba 50M dataset...")
    URM, metadata = load_URM('yambda', version='50m', verbose=True)

    print(f"\nDataset loaded:")
    print(f"  Shape: {URM.shape}")
    print(f"  NNZ: {URM.nnz:,}")
    print(f"  Density: {URM.nnz / (URM.shape[0] * URM.shape[1]):.6%}")

    # Transpose for item-item similarity
    item_matrix = URM.T

    print(f"\nItem matrix (transposed):")
    print(f"  Shape: {item_matrix.shape}")

    # Profile the cosine similarity computation
    print("\nStarting profiling...")
    print("="*70)

    profiler = cProfile.Profile()
    profiler.enable()

    # Run cosine similarity
    similarity_matrix = sim.cosine(
        item_matrix,
        k=100,
        shrink=0,
        threshold=0,
        verbose=True,
        format_output='csr',
        num_threads=0  # use all available cores
    )

    profiler.disable()

    print("\n" + "="*70)
    print("Profiling complete!")
    print("="*70)

    # Print statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')

    print("\n" + "="*70)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*70)
    stats.print_stats(30)

    print("\n" + "="*70)
    print("TOP 30 FUNCTIONS BY INTERNAL TIME (excluding subcalls)")
    print("="*70)
    stats.sort_stats('tottime')
    stats.print_stats(30)

    # Save detailed profile to file
    output_file = 'profile_cosine_yamba50m.prof'
    profiler.dump_stats(output_file)
    print(f"\n\nDetailed profile saved to: {output_file}")
    print("To view: python -m pstats profile_cosine_yamba50m.prof")

    # Also save human-readable report
    report_file = 'profile_cosine_yamba50m.txt'
    with open(report_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()

        f.write("="*70 + "\n")
        f.write("PROFILE REPORT: Cosine Similarity on Yamba 50M\n")
        f.write("="*70 + "\n\n")

        f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
        f.write("-"*70 + "\n")
        stats.sort_stats('cumulative')
        stats.print_stats(50)

        f.write("\n" + "="*70 + "\n")
        f.write("TOP 50 FUNCTIONS BY INTERNAL TIME\n")
        f.write("-"*70 + "\n")
        stats.sort_stats('tottime')
        stats.print_stats(50)

    print(f"Human-readable report saved to: {report_file}")

    # Print summary
    print("\n" + "="*70)
    print("RESULT SUMMARY")
    print("="*70)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix NNZ: {similarity_matrix.nnz:,}")
    print(f"Similarity matrix density: {similarity_matrix.nnz / (similarity_matrix.shape[0] * similarity_matrix.shape[1]):.6%}")
    print("="*70)


if __name__ == '__main__':
    profile_cosine_yamba50m()
