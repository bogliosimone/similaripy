//! In-place CSR normalizations (l1, l2, max, TF-IDF, BM25+), the same formulas
//! as the 0.7.0 Cython code. Generic over f32/f64 data and i32/i64 indices,
//! since `similaripy.normalization` keeps the input dtypes.
//!
//! Float math follows the C code it replaces: per-row sums in the data type,
//! `log` and the BM25 terms in f64, rounded back where Cython stores a
//! `floating`. Parity is within rtol 1e-5, not bitwise (C++ is `-ffast-math`).

use std::ops::{Add, Div, Mul, Sub};

use crate::Error;

/// Rows per parallel leaf are grouped until they hold at least this many entries.
const LEAF_NNZ: usize = 1 << 16;

pub trait Value:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    fn wide(self) -> f64;
    fn narrow(x: f64) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
}

macro_rules! impl_value {
    ($t:ty) => {
        impl Value for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn wide(self) -> f64 {
                self as f64
            }
            fn narrow(x: f64) -> Self {
                x as $t
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
        }
    };
}
impl_value!(f32);
impl_value!(f64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Norm {
    L1,
    L2,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TfMode {
    Binary,
    Raw,
    Sqrt,
    Freq,
    Log,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IdfMode {
    Unary,
    Base,
    Smooth,
    Prob,
    Bm25,
}

impl TfMode {
    pub fn parse(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "binary" => Self::Binary,
            "raw" => Self::Raw,
            "sqrt" => Self::Sqrt,
            "freq" => Self::Freq,
            "log" => Self::Log,
            _ => return Err(Error::InvalidInput(format!("unknown tf_mode '{s}'"))),
        })
    }
}

impl IdfMode {
    pub fn parse(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "unary" => Self::Unary,
            "base" => Self::Base,
            "smooth" => Self::Smooth,
            "prob" => Self::Prob,
            "bm25" => Self::Bm25,
            _ => return Err(Error::InvalidInput(format!("unknown idf_mode '{s}'"))),
        })
    }
}

/// Divides each row by its l1 / l2 / max norm. Empty rows, zero norms and
/// rows with max <= 0 are left unchanged.
pub fn normalize_rows<T: Value, I: Copy + Into<i64> + Sync>(
    norm: Norm,
    data: &mut [T],
    indptr: &[I],
) -> Result<(), Error> {
    let (lo, hi) = check_indptr(indptr, data.len())?;
    par_rows(&mut data[lo..hi], indptr, &|_, _, row: &mut [T]| {
        let d = match norm {
            Norm::L1 => row.iter().fold(T::ZERO, |s, &x| s + x.abs()),
            Norm::L2 => row.iter().fold(T::ZERO, |s, &x| s + x * x).sqrt(),
            Norm::Max => match row.split_first() {
                Some((&first, rest)) => rest.iter().fold(first, |m, &x| if x > m { x } else { m }),
                None => return,
            },
        };
        // l1/l2: empty row; max: zero division and negative values
        let skip = match norm {
            Norm::Max => d <= T::ZERO,
            _ => d == T::ZERO,
        };
        if !skip {
            row.iter_mut().for_each(|x| *x = *x / d);
        }
    });
    Ok(())
}

/// TF-IDF in place; rows are documents, columns are terms.
pub fn tfidf<T: Value, I: Copy + Into<i64> + Sync>(
    data: &mut [T],
    indices: &[I],
    indptr: &[I],
    n_cols: usize,
    tf_mode: TfMode,
    idf_mode: IdfMode,
    logbase: f64,
) -> Result<(), Error> {
    let (lo, hi) = check_indptr(indptr, data.len())?;
    check_indices(indices, lo, hi, n_cols)?;
    let lb = log_logbase::<T>(logbase);
    let (idf_, _) = idf_and_doc_len(data, indices, indptr, n_cols, idf_mode, lb, false);
    par_rows(&mut data[lo..hi], indptr, &|_, start, row: &mut [T]| {
        let doc_len = row.iter().fold(T::ZERO, |s, &x| s + x);
        for (x, c) in row.iter_mut().zip(&indices[start..]) {
            *x = tf(*x, doc_len, tf_mode, lb) * idf_[ix(*c)];
        }
    });
    Ok(())
}

/// BM25+ in place (`delta = 0` is plain BM25); rows are documents, columns are terms.
#[allow(clippy::too_many_arguments)]
pub fn bm25plus<T: Value, I: Copy + Into<i64> + Sync>(
    data: &mut [T],
    indices: &[I],
    indptr: &[I],
    n_cols: usize,
    k1: f64,
    b: f64,
    delta: f64,
    tf_mode: TfMode,
    idf_mode: IdfMode,
    logbase: f64,
) -> Result<(), Error> {
    let (lo, hi) = check_indptr(indptr, data.len())?;
    check_indices(indices, lo, hi, n_cols)?;
    let lb = log_logbase::<T>(logbase);
    let n_docs = indptr.len() - 1;
    if n_docs == 0 {
        return Ok(());
    }
    let (idf_, doc_len) = idf_and_doc_len(data, indices, indptr, n_cols, idf_mode, lb, true);
    // Serial f32/f64 sum in row order, as Cython: a parallel sum would round differently.
    let avg = doc_len.iter().fold(T::ZERO, |s, &x| s + x) / T::narrow(n_docs as f64);
    let (k1, b, delta) = (T::narrow(k1), T::narrow(b), T::narrow(delta));
    par_rows(&mut data[lo..hi], indptr, &|r, start, row: &mut [T]| {
        let norm_len = T::narrow((1.0 - b.wide()) + (b * doc_len[r] / avg).wide());
        for (x, c) in row.iter_mut().zip(&indices[start..]) {
            let t = tf(*x, doc_len[r], tf_mode, lb);
            let w = t.wide() * (k1.wide() + 1.0) / (t + k1 * norm_len).wide() + delta.wide();
            *x = T::narrow(idf_[ix(*c)].wide() * w);
        }
    });
    Ok(())
}

/// Per-column IDF (0 for unused columns) and, if asked, per-row document length.
/// Serial: the count is a column histogram over all rows (O(nnz) increments).
/// Counts are integers: Cython's `floating` counter stops at 2^24 in f32.
fn idf_and_doc_len<T: Value, I: Copy + Into<i64>>(
    data: &[T],
    indices: &[I],
    indptr: &[I],
    n_cols: usize,
    mode: IdfMode,
    lb: T,
    with_doc_len: bool,
) -> (Vec<T>, Vec<T>) {
    let n_docs = indptr.len() - 1;
    let mut counts = vec![0usize; n_cols];
    let mut doc_len = Vec::with_capacity(if with_doc_len { n_docs } else { 0 });
    for r in 0..n_docs {
        let (a, z) = (ix(indptr[r]), ix(indptr[r + 1]));
        let mut len = T::ZERO;
        for (&x, &c) in data[a..z].iter().zip(&indices[a..z]) {
            len = len + x;
            if x > T::ZERO {
                counts[ix(c)] += 1;
            }
        }
        if with_doc_len {
            doc_len.push(len);
        }
    }
    let n = T::narrow(n_docs as f64);
    let idf_ = counts
        .iter()
        .map(|&c| {
            if c == 0 {
                T::ZERO
            } else {
                idf(T::narrow(c as f64), n, mode, lb)
            }
        })
        .collect();
    (idf_, doc_len)
}

fn tf<T: Value>(freq: T, doc_len: T, mode: TfMode, lb: T) -> T {
    match mode {
        TfMode::Binary => {
            if freq != T::ZERO {
                T::ONE
            } else {
                T::ZERO
            }
        }
        TfMode::Raw => freq,
        TfMode::Sqrt => freq.sqrt(),
        TfMode::Freq => freq / doc_len,
        TfMode::Log => T::narrow((T::ONE + freq).wide().ln() / lb.wide()),
    }
}

fn idf<T: Value>(df: T, n_docs: T, mode: IdfMode, lb: T) -> T {
    let lb = lb.wide();
    T::narrow(match mode {
        IdfMode::Unary => return T::ONE,
        IdfMode::Base => (n_docs / df).wide().ln() / lb,
        IdfMode::Smooth => (n_docs / (T::ONE + df)).wide().ln() / lb,
        IdfMode::Prob => ((n_docs - df) / df).wide().ln() / lb,
        IdfMode::Bm25 => (((n_docs - df).wide() + 0.5) / (df.wide() + 0.5)).ln() / lb,
    })
}

/// `log(logbase)` rounded to the data type, as Cython's `floating log_logbase`.
fn log_logbase<T: Value>(logbase: f64) -> T {
    T::narrow(T::narrow(logbase).wide().ln())
}

fn ix<I: Into<i64>>(i: I) -> usize {
    i.into() as usize
}

/// indptr: non-empty, non-negative, monotone, last entry within `data_len`.
/// Returns the data range the rows cover.
fn check_indptr<I: Copy + Into<i64>>(
    indptr: &[I],
    data_len: usize,
) -> Result<(usize, usize), Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("indptr: {m}")));
    let (Some(&first), Some(&last)) = (indptr.first(), indptr.last()) else {
        return bad("empty");
    };
    if first.into() < 0 {
        return bad("negative");
    }
    if indptr.windows(2).any(|w| w[0].into() > w[1].into()) {
        return bad("not monotone");
    }
    if last.into() as u64 > data_len as u64 {
        return bad("last entry exceeds data length");
    }
    Ok((ix(first), ix(last)))
}

/// Column indices of entries `lo..hi` are in `0..n_cols`.
fn check_indices<I: Copy + Into<i64>>(
    indices: &[I],
    lo: usize,
    hi: usize,
    n_cols: usize,
) -> Result<(), Error> {
    if indices.len() < hi {
        return Err(Error::InvalidInput(
            "indices shorter than indptr's last entry".into(),
        ));
    }
    if indices[lo..hi]
        .iter()
        .any(|&c| c.into() < 0 || c.into() as u64 >= n_cols as u64)
    {
        return Err(Error::InvalidInput("column index out of range".into()));
    }
    Ok(())
}

/// Runs `f(row, start, row_data)` for every row, in parallel. `data` covers
/// entries `indptr[0]..indptr[last]`; `start` is the row's absolute offset.
/// Splits by entries, not rows, so popularity-ordered rows still balance (D9).
fn par_rows<T, I, F>(data: &mut [T], indptr: &[I], f: &F)
where
    T: Send,
    I: Copy + Into<i64> + Sync,
    F: Fn(usize, usize, &mut [T]) + Sync,
{
    rows_from(data, indptr, 0, LEAF_NNZ, f)
}

fn rows_from<T, I, F>(data: &mut [T], indptr: &[I], first_row: usize, leaf: usize, f: &F)
where
    T: Send,
    I: Copy + Into<i64> + Sync,
    F: Fn(usize, usize, &mut [T]) + Sync,
{
    let n = indptr.len() - 1;
    let base = ix(indptr[0]);
    if n > 1 && data.len() > leaf {
        let half = (base + data.len() / 2) as i64;
        let mid = indptr
            .partition_point(|&p| p.into() <= half)
            .clamp(1, n - 1);
        let (l, r) = data.split_at_mut(ix(indptr[mid]) - base);
        rayon::join(
            || rows_from(l, &indptr[..=mid], first_row, leaf, f),
            || rows_from(r, &indptr[mid..], first_row + mid, leaf, f),
        );
    } else {
        for i in 0..n {
            let (a, z) = (ix(indptr[i]), ix(indptr[i + 1]));
            f(first_row + i, a, &mut data[a - base..z - base]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::Mutex;

    proptest! {
        // Miri runs ~1000x slower: 8 cases (256 made the Miri job 80-105 min).
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // Every row is visited once, with its own offset and entries, however it splits.
        #[test]
        fn par_rows_visits_each_row_once(
            lens in proptest::collection::vec(0usize..6, 1..40),
            leaf in 0usize..8,
            offset in 0usize..3,
        ) {
            let mut indptr = vec![offset as i64];
            for l in &lens {
                indptr.push(indptr.last().unwrap() + *l as i64);
            }
            let total = *indptr.last().unwrap() as usize;
            let mut data: Vec<f32> = (0..total).map(|i| i as f32).collect();
            let seen = Mutex::new(vec![0; lens.len()]);
            rows_from(&mut data[offset..], &indptr, 0, leaf, &|r, start, row: &mut [f32]| {
                assert_eq!(start, indptr[r] as usize);
                assert_eq!(row.len(), lens[r]);
                assert!(row.iter().enumerate().all(|(j, &x)| x == (start + j) as f32));
                seen.lock().unwrap()[r] += 1;
            });
            prop_assert!(seen.into_inner().unwrap().iter().all(|&c| c == 1));
        }
    }
}
