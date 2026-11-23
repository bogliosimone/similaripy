//! In-place CSR normalizations (l1, l2, max, TF-IDF, BM25+) of each row or
//! each column, and the per-event decay of `time_decay` / `rank_decay`
//! (`Decay`, `user_ages`). Values are f32 (the similarities' type; indices
//! i32, indptr i64).
//!
//! Float math: sums (norms, document lengths, their average) in f64, in an
//! order fixed by the data (row order, so a column's sum adds in the same
//! order as the transpose's row would); `log`, IDF and the BM25 terms in f64;
//! one rounding to f32 per stored value. Matches similaripy 0.7.0 within rtol
//! 1e-5 (`benchmarks/parity_normalization.py`), not bitwise; BM25 changed on
//! purpose (always-positive IDF, TF without `(k1 + 1)`: see `bm25plus`).

use rayon::prelude::*;

use crate::Error;
use crate::csr::check_structure;
use crate::prep::{LEAF_NNZ, par_leaves_one};

/// What a normalization treats as one vector (a document, for the term
/// weightings): each row (Python's `axis=1`) or each column (`axis=0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Axis {
    Rows,
    Cols,
}

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
    /// `ln(1 + (N - df + 0.5) / (df + 0.5))`, always > 0 (Lucene, bm25s).
    Bm25,
    /// Robertson's `ln((N - df + 0.5) / (df + 0.5))`: negative for df > N/2.
    Bm25Robertson,
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
            "bm25_robertson" => Self::Bm25Robertson,
            _ => return Err(Error::InvalidInput(format!("unknown idf_mode '{s}'"))),
        })
    }
}

/// Divides each row or column by its l1 / l2 / max norm (sum of |x|, sqrt of
/// the sum of x^2, largest |x|: signed values keep their signs). Empty
/// vectors and zero norms are left unchanged.
pub fn normalize(
    norm: Norm,
    axis: Axis,
    data: &mut [f32],
    indices: &[i32],
    indptr: &[i64],
    n_cols: usize,
) -> Result<(), Error> {
    check_csr(data, indices, indptr, n_cols)?;
    normalize_checked(norm, axis, data, indices, indptr, n_cols);
    Ok(())
}

/// `normalize` on a structure the caller has checked (the similarity driver's
/// walk: `CsrView::new` ran first), without a second O(nnz) scan.
pub(crate) fn normalize_checked(
    norm: Norm,
    axis: Axis,
    data: &mut [f32],
    indices: &[i32],
    indptr: &[i64],
    n_cols: usize,
) {
    let add = |s: f64, x: f32| {
        let x = x.abs() as f64;
        match norm {
            Norm::L1 => s + x,
            Norm::L2 => s + x * x,
            Norm::Max => s.max(x),
        }
    };
    let finish = |s: f64| if norm == Norm::L2 { s.sqrt() } else { s };
    let scale = |x: &mut f32, d: f64| {
        if d != 0.0 {
            *x = (*x as f64 / d) as f32;
        }
    };
    match axis {
        Axis::Rows => par_rows(data, indptr, &|_, _, row| {
            let d = finish(row.iter().fold(0.0, |s, &x| add(s, x)));
            row.iter_mut().for_each(|x| scale(x, d));
        }),
        Axis::Cols => {
            // Serial, in row order: O(nnz) adds into O(cols) sums
            let mut d = vec![0f64; n_cols];
            for (&x, &c) in data.iter().zip(indices) {
                d[c as usize] = add(d[c as usize], x);
            }
            d.iter_mut().for_each(|s| *s = finish(*s));
            par_entries(data, indices, indptr, &|_, c, x| scale(x, d[c]));
        }
    }
}

/// TF-IDF in place; the documents are the rows (`Axis::Rows`) or the columns,
/// the terms the other side.
#[allow(clippy::too_many_arguments)]
pub fn tfidf(
    axis: Axis,
    data: &mut [f32],
    indices: &[i32],
    indptr: &[i64],
    n_cols: usize,
    tf_mode: TfMode,
    idf_mode: IdfMode,
    logbase: f64,
) -> Result<(), Error> {
    check_csr(data, indices, indptr, n_cols)?;
    let lb = logbase.ln();
    let s = Stats::new(axis, data, indices, indptr, n_cols, idf_mode, lb);
    par_entries(data, indices, indptr, &|r, c, x| {
        let (doc, term) = s.doc_term(r, c);
        *x = (tf(*x as f64, s.doc_len[doc], tf_mode, lb) * s.idf[term]) as f32;
    });
    Ok(())
}

/// BM25+ in place (`delta = 0` is plain BM25); the documents are the rows
/// (`Axis::Rows`) or the columns, the terms the other side.
/// Weight `idf * (tf / (tf + k1 * (1 - b + b * len / avg_len)) + delta)`: the TF
/// part without Robertson's `(k1 + 1)` numerator (Lucene, bm25s), so it is in
/// [0, 1); 0.7.0's values are `(k1 + 1)` times these with `delta / (k1 + 1)`.
#[allow(clippy::too_many_arguments)]
pub fn bm25plus(
    axis: Axis,
    data: &mut [f32],
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
    if data.is_empty() {
        return Ok(()); // no document has a length: the average would be 0 / 0
    }
    let lb = logbase.ln();
    let s = Stats::new(axis, data, indices, indptr, n_cols, idf_mode, lb);
    // Serial sum in document order: one fixed rounding, O(documents).
    let avg = s.doc_len.iter().sum::<f64>() / s.doc_len.len() as f64;
    par_entries(data, indices, indptr, &|r, c, x| {
        let (doc, term) = s.doc_term(r, c);
        let norm_len = (1.0 - b) + b * s.doc_len[doc] / avg;
        let t = tf(*x as f64, s.doc_len[doc], tf_mode, lb);
        *x = (s.idf[term] * (t / (t + k1 * norm_len) + delta)) as f32;
    });
    Ok(())
}

/// The term weightings' statistics: each term's IDF (0 for unused terms) and
/// each document's length (the sum of its values).
struct Stats {
    axis: Axis,
    idf: Vec<f64>,
    doc_len: Vec<f64>,
}

impl Stats {
    /// One serial pass in row order (O(nnz) increments into O(rows + cols)
    /// counters). Document frequencies are integers: an f32 counter would
    /// stop at 2^24.
    fn new(
        axis: Axis,
        data: &[f32],
        indices: &[i32],
        indptr: &[i64],
        n_cols: usize,
        mode: IdfMode,
        lb: f64,
    ) -> Self {
        let n_rows = indptr.len() - 1;
        let (n_docs, n_terms) = match axis {
            Axis::Rows => (n_rows, n_cols),
            Axis::Cols => (n_cols, n_rows),
        };
        let mut s = Stats {
            axis,
            idf: Vec::new(),
            doc_len: vec![0.0; n_docs],
        };
        let mut df = vec![0usize; n_terms];
        for r in 0..n_rows {
            let (a, z) = (indptr[r] as usize, indptr[r + 1] as usize);
            for (&x, &c) in data[a..z].iter().zip(&indices[a..z]) {
                let (doc, term) = s.doc_term(r, c as usize);
                s.doc_len[doc] += x as f64;
                if x > 0.0 {
                    df[term] += 1;
                }
            }
        }
        let n = n_docs as f64;
        s.idf = df
            .iter()
            .map(|&d| {
                if d == 0 {
                    0.0
                } else {
                    idf(d as f64, n, mode, lb)
                }
            })
            .collect();
        s
    }

    /// (document, term) of the entry at (row, col).
    fn doc_term(&self, r: usize, c: usize) -> (usize, usize) {
        match self.axis {
            Axis::Rows => (r, c),
            Axis::Cols => (c, r),
        }
    }
}

fn tf(freq: f64, doc_len: f64, mode: TfMode, lb: f64) -> f64 {
    match mode {
        TfMode::Binary => {
            if freq != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        TfMode::Raw => freq,
        TfMode::Sqrt => freq.sqrt(),
        TfMode::Freq => freq / doc_len,
        TfMode::Log => freq.ln_1p() / lb,
    }
}

fn idf(df: f64, n_docs: f64, mode: IdfMode, lb: f64) -> f64 {
    match mode {
        IdfMode::Unary => 1.0,
        IdfMode::Base => (n_docs / df).ln() / lb,
        IdfMode::Smooth => (n_docs / (1.0 + df)).ln() / lb,
        // A term in every document (df == N) weighs 0, as `Base` does: ln(0) would be -inf.
        IdfMode::Prob if df >= n_docs => 0.0,
        IdfMode::Prob => ((n_docs - df) / df).ln() / lb,
        IdfMode::Bm25 => ((n_docs - df + 0.5) / (df + 0.5)).ln_1p() / lb,
        IdfMode::Bm25Robertson => ((n_docs - df + 0.5) / (df + 0.5)).ln() / lb,
    }
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
    if t.is_none() && kind == UserAge::SinceLatest {
        return bad("SinceLatest needs timestamps");
    }
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

/// The canonical CSR the Python prep passes (`_canonical_csr`): indptr from 0
/// to nnz, one value per entry, column indices in `0..n_cols`.
fn check_csr(data: &[f32], indices: &[i32], indptr: &[i64], n_cols: usize) -> Result<(), Error> {
    check_structure("normalize", indptr, indices, n_cols)?;
    if data.len() != indices.len() {
        return Err(Error::InvalidInput(
            "normalize: len(data) != len(indices)".into(),
        ));
    }
    Ok(())
}

/// Runs `f(row, start, row_data)` for every row, in parallel, on
/// `prep::par_leaves_one`'s leaves (split by entries, so popularity-ordered
/// rows still balance); `start` is the row's offset in `data`.
fn par_rows<F: Fn(usize, usize, &mut [f32]) + Sync>(data: &mut [f32], indptr: &[i64], f: &F) {
    par_leaves_one(indptr, data, LEAF_NNZ, &|row0, ip, d| {
        let base = ip[0] as usize;
        for (i, w) in ip.windows(2).enumerate() {
            let (a, z) = (w[0] as usize, w[1] as usize);
            f(row0 + i, a, &mut d[a - base..z - base]);
        }
    });
}

/// Runs `f(row, col, value)` for every entry, rows in parallel (`par_rows`).
fn par_entries<F: Fn(usize, usize, &mut f32) + Sync>(
    data: &mut [f32],
    indices: &[i32],
    indptr: &[i64],
    f: &F,
) {
    par_rows(data, indptr, &|r, start, row| {
        for (x, &c) in row.iter_mut().zip(&indices[start..]) {
            f(r, c as usize, x);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (indices, indptr) = ([0, 1, 2, 0, 1, 0, 1, 0, 1], [0i64, 3, 5, 7, 7, 9]);
        normalize(Norm::Max, Axis::Rows, &mut data, &indices, &indptr, 3).unwrap();
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

    /// The transpose of a canonical CSR (rows ascending within each column).
    fn transpose(
        m: &(Vec<i64>, Vec<i32>, Vec<f32>),
        n_cols: usize,
    ) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        let (indptr, indices, data) = m;
        let mut out = vec![Vec::new(); n_cols];
        for r in 0..indptr.len() - 1 {
            for j in indptr[r] as usize..indptr[r + 1] as usize {
                out[indices[j] as usize].push((r as i32, data[j]));
            }
        }
        let mut t = (vec![0i64], vec![], vec![]);
        for col in out {
            for (r, x) in col {
                t.1.push(r);
                t.2.push(x);
            }
            t.0.push(t.1.len() as i64);
        }
        t
    }

    type Op = Box<dyn Fn(Axis, &mut [f32], &[i32], &[i64], usize)>;

    fn ops() -> Vec<Op> {
        let mut ops: Vec<Op> = Vec::new();
        for norm in [Norm::L1, Norm::L2, Norm::Max] {
            ops.push(Box::new(move |a, d, i, p, n| {
                normalize(norm, a, d, i, p, n).unwrap()
            }));
        }
        let tfs = [
            TfMode::Binary,
            TfMode::Raw,
            TfMode::Sqrt,
            TfMode::Freq,
            TfMode::Log,
        ];
        let idfs = [
            IdfMode::Unary,
            IdfMode::Base,
            IdfMode::Smooth,
            IdfMode::Prob,
            IdfMode::Bm25,
            IdfMode::Bm25Robertson,
        ];
        for tf in tfs {
            for idf in idfs {
                ops.push(Box::new(move |a, d, i, p, n| {
                    tfidf(a, d, i, p, n, tf, idf, 2.0).unwrap()
                }));
                ops.push(Box::new(move |a, d, i, p, n| {
                    bm25plus(a, d, i, p, n, 1.2, 0.75, 0.5, tf, idf, std::f64::consts::E).unwrap()
                }));
            }
        }
        ops
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 64 }))]
        // Axis::Cols on a matrix == Axis::Rows on its transpose, transposed back, bit for bit: the
        // column sums add in row order, as the transpose's rows do.
        #[test]
        fn cols_is_rows_of_the_transpose(
            dense in proptest::collection::vec(
                proptest::collection::vec(proptest::prop_oneof![3 => proptest::strategy::Just(0.0f32), 1 => 0.1f32..9.0], 0..6),
                0..7,
            ),
        ) {
            let n_cols = dense.iter().map(Vec::len).max().unwrap_or(0);
            let mut m = (vec![0i64], vec![], vec![]);
            for row in &dense {
                for (c, &x) in row.iter().enumerate().filter(|(_, x)| **x != 0.0) {
                    m.1.push(c as i32);
                    m.2.push(x);
                }
                m.0.push(m.1.len() as i64);
            }
            for op in ops() {
                let mut cols = m.clone();
                op(Axis::Cols, &mut cols.2, &cols.1, &cols.0, n_cols);
                let mut t = transpose(&m, n_cols);
                op(Axis::Rows, &mut t.2, &t.1, &t.0, dense.len());
                let back = transpose(&t, dense.len());
                proptest::prop_assert_eq!(cols.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    back.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            }
        }
    }
}
