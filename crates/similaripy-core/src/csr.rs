//! Validated borrowed CSR views.
//!
//! The constructors check the structure once (indptr monotone, indices in
//! range), so the kernel can read rows without bounds checks. This module
//! holds the crate's only `unsafe` row access.

use crate::Error;

/// `indptr` starts at 0, never decreases and ends at `nnz`, so every row span
/// lies in `0..=nnz`. Shared with the prep functions.
pub(crate) fn check_indptr(name: &str, indptr: &[i64], nnz: usize) -> Result<(), Error> {
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
    if indptr[indptr.len() - 1] as usize != nnz {
        return bad("indptr[-1] != len(indices)");
    }
    Ok(())
}

fn check_structure(
    name: &'static str,
    indptr: &[i64],
    indices: &[i32],
    n_cols: usize,
) -> Result<(), Error> {
    let bad = |msg: &str| Err(Error::InvalidInput(format!("{name}: {msg}")));
    check_indptr(name, indptr, indices.len())?;
    if indices.iter().any(|&c| c < 0 || c as usize >= n_cols) {
        return bad("column index out of range");
    }
    Ok(())
}

/// Start and end of row `r`.
///
/// # Safety
/// `r + 1 < indptr.len()`.
#[inline(always)]
unsafe fn row_span(indptr: &[i64], r: usize) -> (usize, usize) {
    // SAFETY: caller contract.
    unsafe {
        (
            *indptr.get_unchecked(r) as usize,
            *indptr.get_unchecked(r + 1) as usize,
        )
    }
}

fn rows_sorted(indptr: &[i64], indices: &[i32]) -> bool {
    indptr
        .windows(2)
        .all(|w| indices[w[0] as usize..w[1] as usize].is_sorted())
}

/// A CSR matrix with values. Indices within a row need not be sorted.
#[derive(Clone, Copy)]
pub struct CsrView<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
    data: &'a [f32],
    n_cols: usize,
}

impl<'a> CsrView<'a> {
    pub fn new(
        name: &'static str,
        indptr: &'a [i64],
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

    /// A view over arrays that passed `new` before (an owner that validated
    /// them once and never changes them).
    ///
    /// # Safety
    /// `new` with the same arguments returned `Ok`.
    pub(crate) unsafe fn new_unchecked(
        indptr: &'a [i64],
        indices: &'a [i32],
        data: &'a [f32],
        n_cols: usize,
    ) -> Self {
        Self {
            indptr,
            indices,
            data,
            n_cols,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// (indptr, indices, data).
    pub(crate) fn arrays(&self) -> (&'a [i64], &'a [i32], &'a [f32]) {
        (self.indptr, self.indices, self.data)
    }

    pub fn indices_sorted(&self) -> bool {
        rows_sorted(self.indptr, self.indices)
    }

    /// Column indices and values of row `r`.
    ///
    /// # Safety
    /// `r < self.n_rows()`.
    #[inline(always)]
    pub(crate) unsafe fn row_unchecked(&self, r: usize) -> (&'a [i32], &'a [f32]) {
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
    pub(crate) unsafe fn row_span_unchecked(&self, r: usize) -> (usize, usize) {
        // SAFETY: r + 1 < len(indptr).
        unsafe { row_span(self.indptr, r) }
    }

    /// Column indices and values at positions `start..end`.
    ///
    /// # Safety
    /// `start <= end <= nnz`, e.g. a sub-span of `row_span_unchecked`.
    #[inline(always)]
    pub(crate) unsafe fn span_unchecked(&self, start: usize, end: usize) -> (&'a [i32], &'a [f32]) {
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
    indptr: &'a [i64],
    indices: &'a [i32],
    n_cols: usize,
}

impl<'a> CsrPattern<'a> {
    pub fn new(
        name: &'static str,
        indptr: &'a [i64],
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
    pub(crate) unsafe fn row_unchecked(&self, r: usize) -> &'a [i32] {
        // SAFETY: same invariant as `CsrView::row_unchecked`.
        unsafe {
            let (start, end) = row_span(self.indptr, r);
            self.indices.get_unchecked(start..end)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (indptr, indices, data, expected message)
    type Case = (&'static [i64], &'static [i32], &'static [f32], &'static str);

    #[test]
    fn rejects_bad_structure() {
        // 3 columns.
        let cases: &[Case] = &[
            (&[], &[], &[], "indptr is empty"),
            (&[1, 1], &[0], &[1.0], "indptr[0] != 0"),
            (&[0, 2, 1], &[0, 1], &[1.0, 1.0], "indptr is not monotone"),
            (&[0, 1], &[0, 1], &[1.0, 1.0], "indptr[-1] != len(indices)"),
            (&[0, 1], &[3], &[1.0], "column index out of range"),
            (&[0, 1], &[-1], &[1.0], "column index out of range"),
            (&[0, 1], &[0], &[], "len(data) != len(indices)"),
        ];
        for &(indptr, indices, data, msg) in cases {
            let err = CsrView::new("m", indptr, indices, data, 3).err();
            assert_eq!(err, Some(Error::InvalidInput(format!("m: {msg}"))));
        }
        let err = CsrPattern::new("sel", &[0, 2], &[2, 0], 3).err();
        let msg = "sel: indices not sorted within rows";
        assert_eq!(err, Some(Error::InvalidInput(msg.into())));
    }

    #[test]
    fn accepts_valid_structure() {
        // Unsorted rows and empty rows are fine for a view.
        let v = CsrView::new("m", &[0, 2, 2, 3], &[2, 0, 1], &[1.0, 2.0, 3.0], 3).unwrap();
        assert!(!v.indices_sorted());
        // SAFETY: rows 0..3 exist.
        unsafe {
            assert_eq!(v.row_unchecked(0), (&[2, 0][..], &[1.0, 2.0][..]));
            assert_eq!(v.row_unchecked(1).0.len(), 0);
            assert_eq!(v.row_unchecked(2), (&[1][..], &[3.0][..]));
        }
        let p = CsrPattern::new("sel", &[0, 2], &[0, 2], 3).unwrap();
        // SAFETY: row 0 exists.
        assert_eq!(unsafe { p.row_unchecked(0) }, &[0, 2]);
    }
}
