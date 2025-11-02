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

namespace s_plus {

// Sentinel values for linked list implementation
static const int UNSET = -1;
static const int LIST_END = -2;

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
                                    Value c1, Value c2, // cosine exponents
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
        nonzeros(column_count, UNSET),
        Xtversky(Xtversky), Ytversky(Ytversky),
        Xcosine(Xcosine), Ycosine(Ycosine),
        Xdepop(Xdepop), Ydepop(Ydepop),
        a1(a1),
        l1(l1), l2(l2), l3(l3),
        t1(t1), t2(t2),
        c1(c1), c2(c2),
        stabilized_shrink(stabilized_shrink),
        bayesian_shrink(bayesian_shrink),
        threshold(threshold),
        filter_mode(filter_mode),
        filter_m_indptr(filter_m_indptr),
        filter_m_indices(filter_m_indices),
        target_col_mode(target_col_mode),
        target_col_m_indptr(target_col_m_indptr),
        target_col_m_indices(target_col_m_indices),
        head(LIST_END), length(0) {
    }

    /* Adds value to the item at index */
    void add(Index index, Value value) {
        sums[index] += value;

        if (nonzeros[index] == UNSET) {
            nonzeros[index] = head;
            head = index;
            length += 1;
        }
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
        for (int i = 0; i < length; ++i) {
            Index col = head;
            Value xy = sums[col];

            // compute similarity value with normalization
            Value val = computeSimilarity(col, xy);

            // apply threshold and filter/target checks
            if (val >= threshold && !isFiltered(col) && isTargetColumn(col)) {
                f(col, val);
            }

            // clear up memory and advance linked list
            head = nonzeros[head];
            sums[col] = 0;
            nonzeros[col] = UNSET;
        }
        length = 0;
        head = LIST_END;
    }

    Index nnz() const { return length; }

 protected:
    std::vector<Value> sums;
    std::vector<Index> nonzeros;
    const Value * Xtversky;
    const Value * Ytversky;
    const Value * Xcosine;
    const Value * Ycosine;
    const Value * Xdepop;
    const Value * Ydepop;
    Value a1;
    Value l1, l2, l3;
    Value t1, t2;
    Value c1, c2;
    Value stabilized_shrink;
    Value bayesian_shrink;
    Value threshold;
    Index row;
    Index filter_mode;
    Index * filter_m_indptr, * filter_m_indices;
    Index target_col_mode;
    Index * target_col_m_indptr, * target_col_m_indices;
    Index head, length;
};

}  // namespace s_plus
#endif  // SPLUS_H_
