/*
    @author: Simone Boglio
    @mail: bogliosimone@gmail.com
*/

#ifndef SPLUS_H_
#define SPLUS_H_

#include <algorithm>
#include <vector>
#include <utility>
#include <functional>
#include <cmath>
#include "progress_bar.h"

// Include prefetch intrinsics for MSVC
#ifdef _MSC_VER
#include <xmmintrin.h>
#endif

namespace s_plus {

// Column selection mode for filtering and targeting
enum SelectionMode {
    SELECTION_NONE = 0,    // no selection (filter: none, target: all)
    SELECTION_ARRAY = 1,   // use single array of indices
    SELECTION_MATRIX = 2   // use per-row matrix indices
};

/*
    Functor that stores the Top K (Value/Index) pairs
    passed to it in its results member
 */
template <typename Index, typename Value>
struct TopK {
    explicit TopK(size_t K) : K(K) {
        results.reserve(K);  // pre-allocate to avoid reallocations
    }

    void operator()(Index index, Value score) {
        size_t size = results.size();

        if (size < K) {
            // still filling up - just add and maintain heap property
            results.push_back(std::make_pair(score, index));
            std::push_heap(results.begin(), results.end(), heap_order);
        } else if (score > results[0].first) {
            // heap is full and we found a better element
            // replace root and sift down (O(log K) vs O(K) for make_heap)
            std::pop_heap(results.begin(), results.end(), heap_order);
            results.back() = std::make_pair(score, index);
            std::push_heap(results.begin(), results.end(), heap_order);
        }
    }

    size_t K;
    std::vector<std::pair<Value, Index> > results;
    std::greater<std::pair<Value, Index> > heap_order;
};

/*
    Sparse matrix multiplication algorithm described
    in the paper 'Sparse Matrix Multiplication Package (SMMP)'
*/
template <typename Index, typename Value>
class SparseMatrixMultiplier {
 public:
    explicit SparseMatrixMultiplier(Index column_count,
                                    const Value * Xtversky, const Value * Ytversky, //normalization terms tversky
                                    const Value * Xcosine, const Value * Ycosine, //normalization terms cosine
                                    const Value * Xdepop, const Value * Ydepop, //depop terms tversky
                                    Value a1, // power weight for product term
                                    Value l1, Value l2, Value l3, // weights tversky and cosine and depop
                                    Value t1, Value t2, // tversky coefficients
                                    Value stabilized_shrink,
                                    Value bayesian_shrink,
                                    Value threshold,
                                    Index filter_mode,
                                    Index * filter_m_indptr, Index * filter_m_indices,
                                    Index target_col_mode,
                                    Index * target_col_m_indptr, Index * target_col_m_indices
                                    )
        :
        sums(column_count, 0),
        Xtversky(Xtversky), Ytversky(Ytversky),
        Xcosine(Xcosine), Ycosine(Ycosine),
        Xdepop(Xdepop), Ydepop(Ydepop),
        a1(a1),
        l1(l1), l2(l2), l3(l3),
        t1(t1), t2(t2),
        stabilized_shrink(stabilized_shrink),
        bayesian_shrink(bayesian_shrink),
        threshold(threshold),
        filter_mode(filter_mode),
        filter_m_indptr(filter_m_indptr),
        filter_m_indices(filter_m_indices),
        target_col_mode(target_col_mode),
        target_col_m_indptr(target_col_m_indptr),
        target_col_m_indices(target_col_m_indices) {
        nonzero_cols.reserve(1024);  // Pre-allocate to reduce reallocation overhead
    }

    /* Adds value to the item at index */
    void add(Index index, Value value) {
        if (sums[index] == 0) {
            nonzero_cols.push_back(index);
        }
        sums[index] += value;
    }

    void setIndexRow(Index index_row) {
        row = index_row;
    }

 private:
    /* Compute similarity value with normalization and shrinkage */
    Value computeSimilarity(Index col, Value xy) const {
        Value valTversky = 0, valCosine = 0, valDepop = 0;
        Value val = xy;

        // compute normalization terms
        if (l1 != 0)  // tversky
            valTversky = l1 * (t1 * (Xtversky[row] - xy) + t2 * (Ytversky[col] - xy) + xy);
        if (l2 != 0)  // cosine
            valCosine = l2 * (Xcosine[row] * Ycosine[col]);
        if (l3 != 0)  // depop
            valDepop = l3 * (Xdepop[row] * Ydepop[col]);
        if (a1 != 1)  // power product
            xy = std::pow(xy, a1);

        // compute similarity value
        if (l1 != 0 || l2 != 0 || l3 != 0 || stabilized_shrink != 0 || bayesian_shrink != 0) {
            Value denominator = valTversky + valCosine + valDepop + stabilized_shrink;
            if (denominator != 0)
                val = xy / denominator;
            else
                val = 0;

            if (bayesian_shrink != 0)
                val = val * (xy / (xy + bayesian_shrink));
        }

        return val;
    }

    /* Check if column should be filtered out */
    bool isFiltered(Index col) const {
        // SELECTION_NONE: no filtering (also used when pre-filtered in Python)
        // SELECTION_ARRAY: handled by Python pre-filtering, never reaches here
        if (filter_mode == SELECTION_NONE || filter_mode == SELECTION_ARRAY) return false;

        // SELECTION_MATRIX: per-row filtering (can't pre-filter in Python)
        if (filter_mode == SELECTION_MATRIX) {
            Index start = filter_m_indptr[row];
            Index end = filter_m_indptr[row + 1];
            return std::binary_search(&filter_m_indices[start], &filter_m_indices[end], col);
        }

        return false;
    }

    /* Check if column is in target set */
    bool isTargetColumn(Index col) const {
        // SELECTION_NONE: include all (also used when pre-filtered in Python)
        // SELECTION_ARRAY: handled by Python pre-filtering, never reaches here
        if (target_col_mode == SELECTION_NONE || target_col_mode == SELECTION_ARRAY) return true;

        // SELECTION_MATRIX: per-row targeting (can't pre-filter in Python)
        if (target_col_mode == SELECTION_MATRIX) {
            Index start = target_col_m_indptr[row];
            Index end = target_col_m_indptr[row + 1];
            return std::binary_search(&target_col_m_indices[start], &target_col_m_indices[end], col);
        }

        return true;
    }

 public:
    /* Calls a function once per non-zero entry in the row, also clears entries for the next row */
    template <typename Function>
    void foreach(Function & f) {  // NOLINT(*)
        // Sequential vector iteration (cache-friendly)
        for (size_t i = 0; i < nonzero_cols.size(); ++i) {
            Index col = nonzero_cols[i];
            Value xy = sums[col];

            // skip work for filtered/untargeted columns
            if (!isFiltered(col) && isTargetColumn(col)) {
                // compute similarity value with normalization
                Value val = computeSimilarity(col, xy);

                // apply threshold
                if (val >= threshold) {
                    f(col, val);
                }
            }

            // clear for next row
            sums[col] = 0;
        }
        nonzero_cols.clear();
    }

    Index nnz() const { return nonzero_cols.size(); }

 protected:
    std::vector<Value> sums;
    std::vector<Index> nonzero_cols;  // Sequential storage for cache-friendly access
    const Value * Xtversky;
    const Value * Ytversky;
    const Value * Xcosine;
    const Value * Ycosine;
    const Value * Xdepop;
    const Value * Ydepop;
    Value a1;
    Value l1, l2, l3;
    Value t1, t2;
    Value stabilized_shrink;
    Value bayesian_shrink;
    Value threshold;
    Index row;
    Index filter_mode;
    Index * filter_m_indptr, * filter_m_indices;
    Index target_col_mode;
    Index * target_col_m_indptr, * target_col_m_indices;
};

/*
    Compute top-K similarities for multiple rows in parallel using OpenMP.
    This function encapsulates the entire parallel computation loop.

    Parameters:
    - n_targets: Number of target rows to process
    - targets: Array of row indices to process
    - m1_data, m1_indices, m1_indptr: CSR matrix 1 data
    - m2_data, m2_indices, m2_indptr: CSR matrix 2 data
    - Xtversky, Ytversky: Tversky normalization arrays (can be empty)
    - Xcosine, Ycosine: Cosine normalization arrays (can be empty)
    - Xdepop, Ydepop: Depopularization arrays (can be empty)
    - a1, l1, l2, l3, t1, t2: Algorithm parameters
    - stabilized_shrink, bayesian_shrink, threshold: Shrinkage and threshold parameters
    - k: Number of top results to keep per row
    - n_output_cols: Total number of columns in output
    - filter_mode, filter_m_indptr, filter_m_indices: Column filtering configuration
    - target_col_mode, target_col_m_indptr, target_col_m_indices: Column targeting configuration
    - rows, cols, values: Pre-allocated output arrays (size: n_targets * k)
    - progress: Progress bar for tracking computation
    - num_threads: Number of OpenMP threads (0 = use all available)
*/
template <typename Index, typename Value>
void compute_similarities_parallel(
    Index n_targets,
    const Index* targets,
    const Value* m1_data,
    const Index* m1_indices,
    const Index* m1_indptr,
    const Value* m2_data,
    const Index* m2_indices,
    const Index* m2_indptr,
    const Value* Xtversky,
    const Value* Ytversky,
    const Value* Xcosine,
    const Value* Ycosine,
    const Value* Xdepop,
    const Value* Ydepop,
    Value a1,
    Value l1,
    Value l2,
    Value l3,
    Value t1,
    Value t2,
    Value stabilized_shrink,
    Value bayesian_shrink,
    Value threshold,
    Index k,
    Index n_output_cols,
    Index filter_mode,
    Index* filter_m_indptr,
    Index* filter_m_indices,
    Index target_col_mode,
    Index* target_col_m_indptr,
    Index* target_col_m_indices,
    Index* rows,
    Index* cols,
    Value* values,
    progress::ProgressBar* progress,
    int num_threads
) {
    #pragma omp parallel num_threads(num_threads)
    {
        // Thread-local allocations
        SparseMatrixMultiplier<Index, Value>* neighbours =
            new SparseMatrixMultiplier<Index, Value>(
                n_output_cols,
                Xtversky, Ytversky,
                Xcosine, Ycosine,
                Xdepop, Ydepop,
                a1,
                l1, l2, l3,
                t1, t2,
                stabilized_shrink,
                bayesian_shrink,
                threshold,
                filter_mode,
                filter_m_indptr, filter_m_indices,
                target_col_mode,
                target_col_m_indptr, target_col_m_indices
            );

        TopK<Index, Value>* topk = new TopK<Index, Value>(k);

        // Process rows in parallel with dynamic scheduling
        #pragma omp for schedule(dynamic)
        for (Index i = 0; i < n_targets; ++i) {
            // Update progress (thread-safe, auto-throttled by C++)
            if (progress != nullptr) {
                progress->update(1);
            }

            // Compute row similarity
            const Index t = targets[i];
            neighbours->setIndexRow(t);

            // Sparse matrix multiplication: accumulate products
            // Cache loop bounds to reduce memory accesses
            const Index m1_start = m1_indptr[t];
            const Index m1_end = m1_indptr[t + 1];
            for (Index index1 = m1_start; index1 < m1_end; ++index1) {
                const Index u = m1_indices[index1];
                const Value v1 = m1_data[index1];

                // Prefetch hint: next iteration's indptr (helps with random access pattern)
                #if defined(__GNUC__) || defined(__clang__)
                if (index1 + 1 < m1_end) {
                    __builtin_prefetch(&m2_indptr[m1_indices[index1 + 1]], 0, 1);
                }
                #elif defined(_MSC_VER)
                if (index1 + 1 < m1_end) {
                    _mm_prefetch((const char*)&m2_indptr[m1_indices[index1 + 1]], _MM_HINT_T1);
                }
                #endif

                const Index m2_start = m2_indptr[u];
                const Index m2_end = m2_indptr[u + 1];
                for (Index index2 = m2_start; index2 < m2_end; ++index2) {
                    neighbours->add(m2_indices[index2], m2_data[index2] * v1);
                }
            }

            // Extract top-k results
            topk->results.clear();
            neighbours->foreach(*topk);

            // Write results to output arrays
            Index index3 = k * i;
            for (const auto& result : topk->results) {
                rows[index3] = t;
                cols[index3] = result.second;
                values[index3] = result.first;
                ++index3;
            }
        }

        // Cleanup thread-local allocations
        delete neighbours;
        delete topk;
    }
}

}  // namespace s_plus
#endif  // SPLUS_H_
