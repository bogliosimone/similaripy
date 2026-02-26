#ifndef __COO_TO_CSR_H__
#define __COO_TO_CSR_H__

#include <algorithm>

/*
 * Convert COO matrix A to CSR matrix B.
 *
 * Input Arguments:
 *   n_row        - number of rows in A
 *   nnz          - number of nonzeros in A
 *   Ai[nnz]      - row indices (int)
 *   Aj[nnz]      - column indices (int)
 *   Ax[nnz]      - nonzeros (float)
 *
 * Output Arguments:
 *   Bp[n_row+1]  - row pointer
 *   Bj[nnz]      - column indices
 *   Bx[nnz]      - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated.
 *   Input row and column indices are *not* assumed to be ordered.
 *   Duplicate entries are carried over to the CSR representation.
 *
 *   Complexity: O(nnz + n_row)
 */
template <typename Index>
void coo_to_csr(const int n_row,
                const Index nnz,
                const int Ai[],
                const int Aj[],
                const float Ax[],
                      Index Bp[],
                      Index Bj[],
                      float Bx[])
{
    // Compute number of non-zero entries per row of A
    std::fill(Bp, Bp + n_row, static_cast<Index>(0));

    for (Index n = 0; n < nnz; n++) {
        Bp[Ai[n]]++;
    }

    // Cumsum the nnz per row to get Bp[]
    for (Index i = 0, cumsum = 0; i < static_cast<Index>(n_row); i++) {
        Index temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz;

    // Write Aj, Ax into Bj, Bx
    for (Index n = 0; n < nnz; n++) {
        Index row  = Ai[n];
        Index dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for (Index i = 0, last = 0; i <= static_cast<Index>(n_row); i++) {
        Index temp = Bp[i];
        Bp[i]  = last;
        last   = temp;
    }

    // Now Bp, Bj, Bx form a CSR representation (with possible duplicates)
}

#endif
