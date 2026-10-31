//! In-place CSR normalizations (l1, l2, max, TF-IDF, BM25+), and the per-event
//! decay of `time_decay` / `rank_decay` (`Decay`, `user_ages`). Generic over
//! f32/f64 data and i32/i64 indices, since `similaripy.normalization` keeps the
//! input dtypes.
//!
//! Float math: per-row sums in the data type, `log` and the BM25 terms in f64,
//! rounded back to the data type. Matches similaripy 0.7.0 within rtol 1e-5
//! (`benchmarks/parity_normalization.py`), not bitwise.

use std::ops::{Add, Div, Mul, Sub};

use rayon::prelude::*;

use crate::Error;
use crate::csr::check_indptr;

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

/// Divides each row by its l1 / l2 / max norm (sum of |x|, sqrt of the sum
/// of x^2, largest |x|: signed rows keep their signs). Empty rows and zero
/// norms are left unchanged.
pub fn normalize_rows<T: Value>(norm: Norm, data: &mut [T], indptr: &[i64]) -> Result<(), Error> {
    check_indptr("normalize", indptr, data.len())?;
    par_rows(data, indptr, &|_, _, row: &mut [T]| {
        let d = match norm {
            Norm::L1 => row.iter().fold(T::ZERO, |s, &x| s + x.abs()),
            Norm::L2 => row.iter().fold(T::ZERO, |s, &x| s + x * x).sqrt(),
            // max(x) on non-negative rows, bit for bit
            Norm::Max => row
                .iter()
                .fold(T::ZERO, |m, &x| if x.abs() > m { x.abs() } else { m }),
        };
        if d != T::ZERO {
            row.iter_mut().for_each(|x| *x = *x / d);
        }
    });
    Ok(())
}

/// TF-IDF in place; rows are documents, columns are terms.
pub fn tfidf<T: Value>(
    data: &mut [T],
    indices: &[i32],
    indptr: &[i64],
    n_cols: usize,
    tf_mode: TfMode,
    idf_mode: IdfMode,
    logbase: f64,
) -> Result<(), Error> {
    check_csr(data, indices, indptr, n_cols)?;
    let lb = log_logbase::<T>(logbase);
    let (idf_, _) = idf_and_doc_len(data, indices, indptr, n_cols, idf_mode, lb, false);
    par_rows(data, indptr, &|_, start, row: &mut [T]| {
        let doc_len = row.iter().fold(T::ZERO, |s, &x| s + x);
        for (x, c) in row.iter_mut().zip(&indices[start..]) {
            *x = tf(*x, doc_len, tf_mode, lb) * idf_[*c as usize];
        }
    });
    Ok(())
}

/// BM25+ in place (`delta = 0` is plain BM25); rows are documents, columns are terms.
#[allow(clippy::too_many_arguments)]
pub fn bm25plus<T: Value>(
    data: &mut [T],
    indices: &[i32],
    indptr: &[i64],
    n_cols: usize,
    k1: f64,
    b: f64,
    delta: f64,
    tf_mode: TfMode,
    idf_mode: IdfMode,
    logbase: f64,
) -> Result<(), Error> {
    check_csr(data, indices, indptr, n_cols)?;
    let lb = log_logbase::<T>(logbase);
    let n_docs = indptr.len() - 1;
    if n_docs == 0 {
        return Ok(());
    }
    let (idf_, doc_len) = idf_and_doc_len(data, indices, indptr, n_cols, idf_mode, lb, true);
    // Serial sum in row order, in the data type: one fixed rounding, O(rows).
    let avg = doc_len.iter().fold(T::ZERO, |s, &x| s + x) / T::narrow(n_docs as f64);
    let (k1, b, delta) = (T::narrow(k1), T::narrow(b), T::narrow(delta));
    par_rows(data, indptr, &|r, start, row: &mut [T]| {
        let norm_len = T::narrow((1.0 - b.wide()) + (b * doc_len[r] / avg).wide());
        for (x, c) in row.iter_mut().zip(&indices[start..]) {
            let t = tf(*x, doc_len[r], tf_mode, lb);
            let w = t.wide() * (k1.wide() + 1.0) / (t + k1 * norm_len).wide() + delta.wide();
            *x = T::narrow(idf_[*c as usize].wide() * w);
        }
    });
    Ok(())
}

/// Per-column IDF (0 for unused columns) and, if asked, per-row document length.
/// Serial: the count is a column histogram over all rows (O(nnz) increments).
/// Counts are integers: an f32 counter would stop at 2^24.
fn idf_and_doc_len<T: Value>(
    data: &[T],
    indices: &[i32],
    indptr: &[i64],
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
                counts[c as usize] += 1;
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

/// A decay curve: the weight of an age (1 at age 0). `scale` > 0 (infinite =
/// every weight 1). One formula for the events of a log (offline) and of a
/// live history (a server), so training and serving weights match.
#[derive(Clone, Copy, Debug)]
pub struct Decay {
    shape: Shape,
    scale: f64,
    /// Power law: 2^(1/p) - 1, so the weight is 0.5 at age = scale for every p.
    c: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shape {
    /// 0.5^(age / scale): scale = half-life.
    Exponential,
    /// (1 + (2^(1/p) - 1) age / scale)^-p: half-life scale, longer tail for a smaller p.
    Power(f64),
    /// max(0, 1 - age / scale): 0 from age = scale.
    Linear,
    /// 1 if age < scale, else 0: a window.
    Step,
}

impl Decay {
    pub fn new(shape: Shape, scale: f64) -> Result<Self, Error> {
        let bad = |m: String| Err(Error::InvalidInput(m));
        if scale.is_nan() || scale <= 0.0 {
            return bad(format!("scale must be > 0 (inf = no decay), got {scale}"));
        }
        let c = match shape {
            Shape::Power(p) if !(p > 0.0 && p.is_finite()) => {
                return bad(format!("exponent must be finite and > 0, got {p}"));
            }
            Shape::Power(p) => (std::f64::consts::LN_2 / p).exp_m1(),
            _ => 0.0,
        };
        Ok(Decay { shape, scale, c })
    }

    /// The f32 weight of each age (one live history or every event of a log),
    /// computed in f64. A negative age counts as 0 (clock skew between
    /// services: never an error on the serving path); NaN / inf is an error.
    pub fn weights(&self, ages: &[f64]) -> Result<Vec<f32>, Error> {
        if ages.iter().any(|a| !a.is_finite()) {
            return Err(Error::InvalidInput("ages must be finite".into()));
        }
        Ok(ages
            .iter()
            .map(|&a| self.weight(a.max(0.0)) as f32)
            .collect())
    }

    /// The weight of `age` (>= 0).
    pub fn weight(&self, age: f64) -> f64 {
        let x = age / self.scale;
        match self.shape {
            Shape::Exponential => (-x).exp2(),
            Shape::Power(p) => (1.0 + self.c * x).powf(-p),
            Shape::Linear => (1.0 - x).max(0.0),
            Shape::Step => {
                if x < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// How `user_ages` measures an event's age within its user's events.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UserAge {
    /// The user's latest timestamp minus the event's (needs timestamps).
    SinceLatest,
    /// The recency rank: 0 = the latest event, by timestamp or, without, by
    /// input order; equal timestamps: the later row is the more recent.
    Rank,
}

/// The age of each event within its user's events (`users`: codes in
/// `0..n_users`, one per event), in input order. A counting sort by user
/// (stable), then each user's events in parallel. Peak: ~3 x 8 bytes per event.
pub fn user_ages(
    users: &[i32],
    n_users: usize,
    t: Option<&[f64]>,
    kind: UserAge,
) -> Result<Vec<f64>, Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("user_ages: {m}")));
    let n = users.len();
    if t.is_some_and(|t| t.len() != n) {
        return bad("one timestamp per event");
    }
    if t.is_some_and(|t| t.iter().any(|x| !x.is_finite())) {
        return bad("timestamps must be finite");
    }
    if users.iter().any(|&u| u < 0 || u as usize >= n_users) {
        return bad("user code out of range");
    }
    let t = match (t, kind) {
        (None, UserAge::SinceLatest) => return bad("SinceLatest needs timestamps"),
        (t, _) => t,
    };
    let mut start = vec![0usize; n_users + 1];
    for &u in users {
        start[u as usize + 1] += 1;
    }
    for u in 0..n_users {
        start[u + 1] += start[u];
    }
    let mut next = start.clone();
    let mut events = vec![0usize; n]; // event indices grouped by user, input order
    for (e, &u) in users.iter().enumerate() {
        events[next[u as usize]] = e;
        next[u as usize] += 1;
    }
    drop(next);
    let mut grouped = vec![0f64; n]; // ages in `events` order
    let mut slices = Vec::with_capacity(n_users);
    let mut rest = grouped.as_mut_slice();
    for u in 0..n_users {
        let (head, tail) = rest.split_at_mut(start[u + 1] - start[u]);
        slices.push((u, head));
        rest = tail;
    }
    slices.into_par_iter().for_each(|(u, out)| {
        let ev = &events[start[u]..start[u + 1]];
        match (kind, t) {
            (UserAge::SinceLatest, Some(t)) => {
                let latest = ev.iter().map(|&e| t[e]).fold(f64::NEG_INFINITY, f64::max);
                for (o, &e) in out.iter_mut().zip(ev) {
                    *o = latest - t[e];
                }
            }
            (UserAge::Rank, None) => {
                for (k, o) in out.iter_mut().enumerate() {
                    *o = (ev.len() - 1 - k) as f64;
                }
            }
            (UserAge::Rank, Some(t)) => {
                let mut order: Vec<usize> = (0..ev.len()).collect();
                order.sort_by(|&a, &b| t[ev[a]].total_cmp(&t[ev[b]])); // stable: ties keep input order
                for (k, &j) in order.iter().enumerate() {
                    out[j] = (ev.len() - 1 - k) as f64;
                }
            }
            (UserAge::SinceLatest, None) => unreachable!(),
        }
    });
    let mut ages = vec![0f64; n];
    for (&e, &a) in events.iter().zip(&grouped) {
        ages[e] = a;
    }
    Ok(ages)
}

/// `log(logbase)` rounded to the data type, like every stored value.
fn log_logbase<T: Value>(logbase: f64) -> T {
    T::narrow(T::narrow(logbase).wide().ln())
}

fn ix(i: i64) -> usize {
    i as usize
}

/// The canonical CSR the Python prep passes (`_canonical_csr`): indptr from 0
/// to nnz, one value per entry, column indices in `0..n_cols`.
fn check_csr<T>(data: &[T], indices: &[i32], indptr: &[i64], n_cols: usize) -> Result<(), Error> {
    check_indptr("normalize", indptr, indices.len())?;
    if data.len() != indices.len() {
        return Err(Error::InvalidInput("len(data) != len(indices)".into()));
    }
    if indices.iter().any(|&c| c < 0 || c as usize >= n_cols) {
        return Err(Error::InvalidInput("column index out of range".into()));
    }
    Ok(())
}

/// Runs `f(row, start, row_data)` for every row, in parallel. `data` covers
/// entries `indptr[0]..indptr[last]` (a split's `indptr` starts past 0);
/// `start` is the row's absolute offset. Splits by entries, not rows, so
/// popularity-ordered rows still balance. Same split as `prep::par_leaves`,
/// kept separate: one data slice (prep's has two).
fn par_rows<T, F>(data: &mut [T], indptr: &[i64], f: &F)
where
    T: Send,
    F: Fn(usize, usize, &mut [T]) + Sync,
{
    rows_from(data, indptr, 0, LEAF_NNZ, f)
}

fn rows_from<T, F>(data: &mut [T], indptr: &[i64], first_row: usize, leaf: usize, f: &F)
where
    T: Send,
    F: Fn(usize, usize, &mut [T]) + Sync,
{
    let n = indptr.len() - 1;
    let base = ix(indptr[0]);
    if n > 1 && data.len() > leaf {
        let half = base + data.len() / 2;
        let mid = indptr.partition_point(|&p| ix(p) <= half).clamp(1, n - 1);
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

    #[test]
    fn decay_curves() {
        for shape in [
            Shape::Exponential,
            Shape::Power(0.5),
            Shape::Power(3.0),
            Shape::Linear,
            Shape::Step,
        ] {
            let d = Decay::new(shape, 10.0).unwrap();
            assert_eq!(d.weight(0.0), 1.0);
            assert_eq!(Decay::new(shape, f64::INFINITY).unwrap().weight(1e9), 1.0);
            if let Shape::Exponential | Shape::Power(_) = shape {
                assert!((d.weight(10.0) - 0.5).abs() < 1e-12);
            } else {
                assert_eq!(d.weight(10.0), 0.0); // exclusive horizon
            }
        }
        assert!(Decay::new(Shape::Linear, 0.0).is_err());
        assert!(Decay::new(Shape::Linear, f64::NAN).is_err());
        assert!(Decay::new(Shape::Power(0.0), 1.0).is_err());
    }

    #[test]
    fn user_ages_latest_and_rank() {
        // users 1, 0, 1, 1; t 5, 7, 9, 9 (a tie in user 1)
        let (users, t) = ([1, 0, 1, 1], [5.0, 7.0, 9.0, 9.0]);
        let since = user_ages(&users, 2, Some(&t), UserAge::SinceLatest).unwrap();
        assert_eq!(since, [4.0, 0.0, 0.0, 0.0]);
        let rank = user_ages(&users, 2, Some(&t), UserAge::Rank).unwrap();
        assert_eq!(rank, [2.0, 0.0, 1.0, 0.0]); // tie: the later row is the more recent
        let order = user_ages(&users, 2, None, UserAge::Rank).unwrap();
        assert_eq!(order, [2.0, 0.0, 1.0, 0.0]);
        assert!(user_ages(&users, 2, None, UserAge::SinceLatest).is_err());
        assert!(user_ages(&[2], 2, None, UserAge::Rank).is_err());
    }

    #[test]
    fn decay_weights_clamp_and_reject() {
        let d = Decay::new(Shape::Exponential, 10.0).unwrap();
        assert_eq!(d.weights(&[0.0, 10.0, -5.0]).unwrap(), [1.0, 0.5, 1.0]);
        assert!(d.weights(&[f64::NAN]).is_err());
        assert!(d.weights(&[f64::INFINITY]).is_err());
    }

    #[test]
    fn max_norm_divides_by_largest_abs() {
        // rows: mixed signs, all negative, all positive, empty, zeros
        let mut data = vec![2.0f32, -4.0, 1.0, -1.0, -0.5, 3.0, 6.0, 0.0, 0.0];
        let indptr = [0i64, 3, 5, 7, 7, 9];
        normalize_rows(Norm::Max, &mut data, &indptr).unwrap();
        assert_eq!(data, [0.5, -1.0, 0.25, -1.0, -0.5, 0.5, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn check_csr_rejects_non_canonical_input() {
        let ok = |d: &[f32], ix: &[i32], p: &[i64]| check_csr(d, ix, p, 3).is_ok();
        assert!(ok(&[1.0, 2.0], &[0, 2], &[0, 1, 2]));
        assert!(!ok(&[1.0, 2.0], &[0, 2], &[1, 1, 2])); // indptr[0] != 0
        assert!(!ok(&[1.0, 2.0], &[0, 2], &[0, 1, 1])); // indptr[-1] != nnz
        assert!(!ok(&[1.0], &[0, 2], &[0, 1, 2])); // one value per entry
        assert!(!ok(&[1.0, 2.0], &[0, 3], &[0, 1, 2])); // index out of range
    }
}
