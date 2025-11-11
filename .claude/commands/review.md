---
description: Review recent code changes for correctness and performance
---

Review the recent code changes in the repository focusing on:

1. **Correctness**: Check for bugs, edge cases, and logic errors

2. **Performance**: Identify optimization opportunities (following CLAUDE.md guidelines)
   - Flag any performance changes without explaining benefit/trade-offs first
   - Look for unnecessary allocations, copies, or computations

3. **Cython Best Practices**:
   - Proper GIL handling (`with nogil:` for parallel sections)
   - Appropriate use of memory views vs NumPy arrays
   - Correct type declarations (`cdef` for C variables, `def` for Python-callable)
   - Pre-allocated buffers instead of repeated allocations
   - No shared mutable state in OpenMP parallel regions
   - Float32 vs Float64 usage

5. **Testing**: Suggest additional test cases for edge cases if needed

Provide specific, actionable feedback with file and line references.
