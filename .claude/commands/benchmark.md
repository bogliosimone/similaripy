---
description: Run similarity benchmarks on dataset
---

Run the similarity benchmarks and analyze the results.

**Default**: Use `make benchmark-similarity` (Movielens 32M dataset - quick regression test)

After running the benchmarks:
1. Show execution times for each similarity function
2. Identify any significant performance changes compared to previous runs (if available)
4. Report any warnings, errors, or unexpected behavior
5. Provide insights on performance characteristics

**Note:** For more comprehensive benchmarking, the user can specify `make benchmark-similarity-medium` (Yambda 50M) for larger dataset testing.
