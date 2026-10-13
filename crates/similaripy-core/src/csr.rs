//! Validated borrowed CSR views.
//!
//! The constructors check the structure once (indptr monotone, indices in
//! range), so the kernel can read rows without bounds checks. This module
//! holds the crate's only `unsafe` row access.

use crate::Error;

fn check_structure(
    name: &'static str,
    indptr: &[i32],
    indices: &[i32],
    n_cols: usize,
) -> Result<(), Error> {
    let bad = |msg: &str| Err(Error::InvalidInput(format!("{name}: {msg}")));
    if indptr.is_empty() {
        return bad("indptr is empty");
    }
    if indptr[0] != 0 {
        return bad("indptr[0] != 0");
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return bad("indptr is not monotone");
    }
    if indptr[indptr.len() - 1] as usize != indices.len() {
        return bad("indptr[-1] != len(indices)");
    }
    if indices.iter().any(|&c| c < 0 || c as usize >= n_cols) {
        return bad("column index out of range");
    }
    Ok(())
}

fn rows_sorted(indptr: &[i32], indices: &[i32]) -> bool {
    indptr
        .windows(2)
        .all(|w| indices[w[0] as usize..w[1] as usize].is_sorted())
}

/// A CSR matrix with values. Indices within a row need not be sorted.
#[derive(Clone, Copy)]
pub struct CsrView<'a> {
    indptr: &'a [i32],
    indices: &'a [i32],
    data: &'a [f32],
    n_cols: usize,
}

impl<'a> CsrView<'a> {
    pub fn new(
        name: &'static str,
        indptr: &'a [i32],
        indices: &'a [i32],
        data: &'a [f32],
        n_cols: usize,
    ) -> Result<Self, Error> {
        check_structure(name, indptr, indices, n_cols)?;
        if data.len() != indices.len() {
            return Err(Error::InvalidInput(format!(
                "{name}: len(data) != len(indices)"
            )));
        }
        Ok(Self {
            indptr,
            indices,
            data,
            n_cols,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn indices_sorted(&self) -> bool {
        rows_sorted(self.indptr, self.indices)
    }

    /// Column indices and values of row `r`.
    ///
    /// # Safety
    /// `r < self.n_rows()`.
    #[inline(always)]
    pub unsafe fn row_unchecked(&self, r: usize) -> (&'a [i32], &'a [f32]) {
        // SAFETY: caller contract, forwarded.
        unsafe {
            let (start, end) = self.row_span_unchecked(r);
            self.span_unchecked(start, end)
        }
    }

    /// Start and end of row `r` in `indices`/`data`.
    ///
    /// # Safety
    /// `r < self.n_rows()`.
    #[inline(always)]
    pub unsafe fn row_span_unchecked(&self, r: usize) -> (usize, usize) {
        // SAFETY: r + 1 < len(indptr).
        unsafe {
            (
                *self.indptr.get_unchecked(r) as usize,
                *self.indptr.get_unchecked(r + 1) as usize,
            )
        }
    }

    /// Column indices and values at positions `start..end`.
    ///
    /// # Safety
    /// `start <= end <= nnz`, e.g. a sub-span of `row_span_unchecked`.
    #[inline(always)]
    pub unsafe fn span_unchecked(&self, start: usize, end: usize) -> (&'a [i32], &'a [f32]) {
        // SAFETY: caller contract; len(data) == len(indices) (`new`).
        unsafe {
            (
                self.indices.get_unchecked(start..end),
                self.data.get_unchecked(start..end),
            )
        }
    }
}

/// A CSR sparsity pattern (no values) with sorted indices per row,
/// used for the per-row filter/target column selections.
#[derive(Clone, Copy)]
pub struct CsrPattern<'a> {
    indptr: &'a [i32],
    indices: &'a [i32],
    n_cols: usize,
}

impl<'a> CsrPattern<'a> {
    pub fn new(
        name: &'static str,
        indptr: &'a [i32],
        indices: &'a [i32],
        n_cols: usize,
    ) -> Result<Self, Error> {
        check_structure(name, indptr, indices, n_cols)?;
        if !rows_sorted(indptr, indices) {
            return Err(Error::InvalidInput(format!(
                "{name}: indices not sorted within rows"
            )));
        }
        Ok(Self {
            indptr,
            indices,
            n_cols,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Column ids of row `r`, each `< n_cols()`.
    ///
    /// # Safety
    /// `r < self.n_rows()`.
    #[inline(always)]
    pub unsafe fn row_unchecked(&self, r: usize) -> &'a [i32] {
        // SAFETY: same invariant as `CsrView::row_unchecked`.
        unsafe {
            let start = *self.indptr.get_unchecked(r) as usize;
            let end = *self.indptr.get_unchecked(r + 1) as usize;
            self.indices.get_unchecked(start..end)
        }
    }
}
