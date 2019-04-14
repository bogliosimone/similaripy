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

namespace s_plus {

/* 
    Functor that stores the Top K (Value/Index) pairs
    passed to it in its results member
 */
template <typename Index, typename Value>
struct TopK {
    explicit TopK(size_t K) : K(K) {}

    void operator()(Index index, Value score) {
        if ((results.size() < K) || (score > results[0].first)) {
            if (results.size() >= K) {
                std::pop_heap(results.begin(), results.end(), heap_order);
                results.pop_back();
            }

            results.push_back(std::make_pair(score, index));
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
                                    Value * Xtversky, Value * Ytversky, //normalization terms tversky
                                    Value * Xcosine, Value * Ycosine, //normalization terms cosine
                                    Value * Xdepop, Value *Ydepop, //depop terms tversky
                                    Value a1, // power weight for product term
                                    Value l1, Value l2, Value l3, // weights tversky and cosine and depop
                                    Value t1, Value t2, // tversky coefficients
                                    Value c1, Value c2, // cosine exponents
                                    Value shrink, Value threshold,
                                    Index filter_mode, 
                                    Index * filter_m_indptr, Index * filter_m_indices,
                                    Index target_col_mode,
                                    Index * target_col_m_indptr, Index * target_col_m_indices
                                    )
        :
        sums(column_count, 0),
        nonzeros(column_count, -1),
        Xtversky(Xtversky), Ytversky(Ytversky),
        Xcosine(Xcosine), Ycosine(Ycosine),
        Xdepop(Xdepop), Ydepop(Ydepop),
        a1(a1),
        l1(l1), l2(l2), l3(l3),
        t1(t1), t2(t2), 
        c1(c1), c2(c2),
        shrink(shrink), threshold(threshold),
        filter_mode(filter_mode),
        filter_m_indptr(filter_m_indptr),
        filter_m_indices(filter_m_indices),
        target_col_mode(target_col_mode),
        target_col_m_indptr(target_col_m_indptr),
        target_col_m_indices(target_col_m_indices),
        head(-2), length(0) {
    }

    /* Adds value to the item at index */
    void add(Index index, Value value) {
        sums[index] += value;

        if (nonzeros[index] == -1) {
            nonzeros[index] = head;
            head = index;
            length += 1;
        }
    }

    void setIndexRow(Index index_row) {
        row = index_row;
    }

    /* Calls a function once per non-zero entry in the row, also clears entries for the next row */
    template <typename Function>
    void foreach(Function & f) {  // NOLINT(*)

        for (int i = 0; i < length; ++i) {
            Index col = head;
            Value xy = sums[col];
            Value valTversky=0, valCosine=0, valDepop=0, val=xy;
            bool filter = false;
            bool target_col = true;

            if(l1!=0) // tversky
                valTversky = l1 * (t1 * (Xtversky[row] - xy) + t2 * (Ytversky[col] - xy) + xy);
            if(l2!=0) // cosine
                valCosine = l2 * (Xcosine[row] * Ycosine[col]);
            if(l3!=0) // depop
                valDepop = l3 * (Xdepop[row] * Ydepop[col]);
            if(a1!=1) // power product
                xy = std::pow(xy, a1); // very slow operation
            if(l1!=0 || l2!=0 || l3!=0 ||shrink!=0)
                val = xy/(valTversky + valCosine + valDepop + shrink);

            // filter cols, filter_mode = 0:none, 1:array, 2:matrix
            if (filter_mode !=0 ){
                if(filter_mode == 1){
                    Index start = filter_m_indptr[0];
                    Index end = filter_m_indptr[1];
                    if(std::binary_search(&filter_m_indices[start], &filter_m_indices[end], col))
                        filter = true;
                }
                else{
                    if(filter_mode == 2){
                        Index start = filter_m_indptr[row];
                        Index end = filter_m_indptr[row+1];
                        if(std::binary_search(&filter_m_indices[start], &filter_m_indices[end], col))
                            filter = true;
                    }
                }
            }

            // keep only the target cols, target_mode = 0:all, 1:array, 2:matrix
            if (target_col_mode !=0 ){
                if(target_col_mode == 1){
                    Index start = target_col_m_indptr[0];
                    Index end = target_col_m_indptr[1];
                    if(!std::binary_search(&target_col_m_indices[start], &target_col_m_indices[end], col))
                        target_col = false;
                }
                else{
                    if(target_col_mode == 2){
                        Index start = target_col_m_indptr[row];
                        Index end = target_col_m_indptr[row+1];
                        if(!std::binary_search(&target_col_m_indices[start], &target_col_m_indices[end], col))
                            target_col = false;
                    }
                }
            }

            if (val >= threshold && !filter && target_col)
                f(col, val);
            // clear up memory and advance linked list
            head = nonzeros[head];
            sums[col] = 0;
            nonzeros[col] = -1;
        }
        length = 0;
        head = -2;
    }

    Index nnz() const { return length; }

 protected:
    std::vector<Value> sums;
    std::vector<Index> nonzeros;
    Value * Xtversky, * Ytversky;
    Value * Xcosine, * Ycosine;
    Value * Xdepop, * Ydepop;
    Value a1;
    Value l1, l2,l3;
    Value t1, t2;
    Value c1, c2;
    Value shrink, threshold;
    Index row;
    Index filter_mode;
    Index * filter_m_indptr, * filter_m_indices;
    Index target_col_mode;
    Index * target_col_m_indptr, * target_col_m_indices;
    Index head, length;

};
}  // namespace similarity
#endif  // SPLUS
