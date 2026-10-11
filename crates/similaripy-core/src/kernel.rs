//! Top-K similarity kernel: the 0.7.0 algorithm (dense accumulator + list of
//! touched columns, heap top-K), rows in parallel with rayon.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::Error;
use crate::csr::{CsrPattern, CsrView};
use crate::topk::TopK;

/// Per-row (`x_*`, indexed by target row) and per-column (`y_*`, indexed by
/// output column) normalization terms. A pair is only read when its weight
/// in `Params` is non-zero, and may be empty otherwise.
#[derive(Clone, Copy, Default)]
pub struct Norms<'a> {
    pub x_tversky: &'a [f32],
    pub y_tversky: &'a [f32],
    pub x_cosine: &'a [f32],
    pub y_cosine: &'a [f32],
    pub x_depop: &'a [f32],
    pub y_depop: &'a [f32],
}

/// Scoring weights, same meaning as the `s_plus` arguments.
#[derive(Clone, Copy)]
pub struct Params {
    pub a1: f32,
    pub l1: f32,
    pub l2: f32,
    pub l3: f32,
    pub t1: f32,
    pub t2: f32,
    pub stabilized_shrink: f32,
    pub bayesian_shrink: f32,
    pub threshold: f32,
}

impl Default for Params {
    /// Plain dot product.
    fn default() -> Self {
        Self {
            a1: 1.0,
            l1: 0.0,
            l2: 0.0,
            l3: 0.0,
            t1: 1.0,
            t2: 1.0,
            stabilized_shrink: 0.0,
            bayesian_shrink: 0.0,
            threshold: 0.0,
        }
    }
}

/// Everything the kernel reads. `m1` rows are targets, `m1` columns index
/// `m2` rows, `m2` columns are the output columns.
#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    pub m1: CsrView<'a>,
    pub m2: CsrView<'a>,
    pub targets: &'a [i32],
    pub norms: Norms<'a>,
    pub params: Params,
    /// Per-row columns to exclude.
    pub filter: Option<CsrPattern<'a>>,
    /// Per-row columns to keep (all others are excluded).
    pub target_cols: Option<CsrPattern<'a>>,
}

/// Output buffers, `len(targets) * k` each, zeroed by the caller. Row `i`
/// owns slots `[i*k, (i+1)*k)`; slots past its result count stay untouched.
pub struct Output<'a> {
    pub rows: &'a mut [i32],
    pub cols: &'a mut [i32],
    pub values: &'a mut [f32],
}

/// How rows are spread over the rayon pool (D9: both are measured, the loser is deleted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// `par_chunks_mut` + `for_each_init`: rayon's adaptive splitting.
    Adaptive,
    /// `broadcast` + an atomic row counter: OpenMP's `schedule(dynamic, 1)`.
    Dynamic,
}

// ponytail: fixed guess for Schedule::Adaptive, only tuned if it wins D9.
const MIN_ROWS_PER_JOB: usize = 16;

/// Computes the top-`k` scores for every target row on the current rayon pool.
/// `block_size == 0` disables column blocking.
pub fn compute(
    inp: &Inputs,
    k: usize,
    block_size: usize,
    schedule: Schedule,
    out: Output,
) -> Result<(), Error> {
    validate(inp, k, &out)?;
    if k == 0 || inp.targets.is_empty() {
        return Ok(());
    }
    let n_cols = inp.m2.n_cols();
    if block_size > 0 && n_cols > block_size {
        return Err(Error::Unsupported("column blocking (slice 2)".into()));
    }

    // SAFETY (both arms): `validate` checked every target against all row-indexed inputs.
    let row = |sc: &mut Scratch, t: i32, rows: &mut [i32], cols: &mut [i32], values: &mut [f32]| {
        let n = unsafe { sc.single_block_row(inp, t as usize, cols, values) };
        rows[..n].fill(t);
    };
    match schedule {
        Schedule::Adaptive => inp
            .targets
            .par_iter()
            .zip(out.rows.par_chunks_mut(k))
            .zip(out.cols.par_chunks_mut(k))
            .zip(out.values.par_chunks_mut(k))
            .with_min_len(MIN_ROWS_PER_JOB)
            .for_each_init(
                || Scratch::new(n_cols, k),
                |sc, (((&t, rows), cols), values)| row(sc, t, rows, cols, values),
            ),
        Schedule::Dynamic => {
            let next = AtomicUsize::new(0);
            let shared = SharedOutput::new(out, k);
            rayon::broadcast(|_| {
                let mut sc = Scratch::new(n_cols, k);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&t) = inp.targets.get(i) else { break };
                    // SAFETY: the counter hands out each i exactly once, and i < len(targets).
                    let (rows, cols, values) = unsafe { shared.chunk(i) };
                    row(&mut sc, t, rows, cols, values);
                }
            });
        }
    }
    Ok(())
}

/// Output buffers shared by the `Dynamic` workers. Each row index is claimed
/// by one worker only, so the `k`-slot chunks they write never overlap.
struct SharedOutput<'a> {
    rows: *mut i32,
    cols: *mut i32,
    values: *mut f32,
    k: usize,
    _out: PhantomData<Output<'a>>,
}

// SAFETY: workers only write disjoint chunks (see `chunk`).
unsafe impl Sync for SharedOutput<'_> {}

impl<'a> SharedOutput<'a> {
    fn new(out: Output<'a>, k: usize) -> Self {
        Self {
            rows: out.rows.as_mut_ptr(),
            cols: out.cols.as_mut_ptr(),
            values: out.values.as_mut_ptr(),
            k,
            _out: PhantomData,
        }
    }

    /// # Safety
    /// `i < len(targets)`, and no other live reference to chunk `i` exists.
    #[allow(clippy::mut_from_ref)]
    unsafe fn chunk(&self, i: usize) -> (&mut [i32], &mut [i32], &mut [f32]) {
        // SAFETY: buffers are len(targets) * k long (validate), chunk i is exclusive (caller).
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.rows.add(i * self.k), self.k),
                std::slice::from_raw_parts_mut(self.cols.add(i * self.k), self.k),
                std::slice::from_raw_parts_mut(self.values.add(i * self.k), self.k),
            )
        }
    }
}

fn validate(inp: &Inputs, k: usize, out: &Output) -> Result<(), Error> {
    let bad = |msg: &str| Err(Error::InvalidInput(msg.into()));
    let (m1, m2, p, n) = (&inp.m1, &inp.m2, &inp.params, &inp.norms);
    if m1.n_cols() != m2.n_rows() {
        return bad("m1 columns != m2 rows");
    }
    let n_out = inp.targets.len() * k;
    if out.rows.len() != n_out || out.cols.len() != n_out || out.values.len() != n_out {
        return bad("output length != len(targets) * k");
    }
    // Rows needed by every row-indexed input: max target + 1.
    let rows_needed = match inp.targets.iter().min() {
        Some(&t) if t < 0 => return bad("negative target row"),
        Some(_) => *inp.targets.iter().max().unwrap() as usize + 1,
        None => 0,
    };
    if rows_needed > m1.n_rows() {
        return bad("target row out of range");
    }
    let n_cols = m2.n_cols();
    for (w, x, y, name) in [
        (p.l1, n.x_tversky, n.y_tversky, "tversky"),
        (p.l2, n.x_cosine, n.y_cosine, "cosine"),
        (p.l3, n.x_depop, n.y_depop, "depop"),
    ] {
        if w != 0.0 && (x.len() < rows_needed || y.len() < n_cols) {
            return Err(Error::InvalidInput(format!("{name} norms too short")));
        }
    }
    for sel in [inp.filter, inp.target_cols].into_iter().flatten() {
        if sel.n_rows() < rows_needed {
            return bad("column selection has fewer rows than targets need");
        }
    }
    Ok(())
}

impl Params {
    fn normalizes(&self) -> bool {
        self.l1 != 0.0
            || self.l2 != 0.0
            || self.l3 != 0.0
            || self.stabilized_shrink != 0.0
            || self.bayesian_shrink != 0.0
    }

    /// Same arithmetic, in the same order, as the C++ `computeSimilarity`.
    ///
    /// # Safety
    /// `row` and `col` are in range of every norm pair whose weight is non-zero.
    #[inline(always)]
    unsafe fn score(&self, n: &Norms, row: usize, col: usize, xy: f32) -> f32 {
        if !self.normalizes() {
            return xy; // a1 is ignored without normalization, as in 0.7.0
        }
        let (mut tv, mut cs, mut dp) = (0.0, 0.0, 0.0);
        // SAFETY: caller contract, checked in `validate`.
        unsafe {
            if self.l1 != 0.0 {
                let (x, y) = (
                    *n.x_tversky.get_unchecked(row),
                    *n.y_tversky.get_unchecked(col),
                );
                tv = self.l1 * (self.t1 * (x - xy) + self.t2 * (y - xy) + xy);
            }
            if self.l2 != 0.0 {
                cs = self.l2 * (n.x_cosine.get_unchecked(row) * n.y_cosine.get_unchecked(col));
            }
            if self.l3 != 0.0 {
                dp = self.l3 * (n.x_depop.get_unchecked(row) * n.y_depop.get_unchecked(col));
            }
        }
        let xy_pow = if self.a1 != 1.0 { xy.powf(self.a1) } else { xy };
        let den = tv + cs + dp + self.stabilized_shrink;
        let mut val = if den != 0.0 { xy_pow / den } else { 0.0 };
        if self.bayesian_shrink != 0.0 {
            val *= xy_pow / (xy_pow + self.bayesian_shrink);
        }
        val
    }
}

/// Per-worker state, reused across rows.
struct Scratch {
    sums: Vec<f32>,
    touched: Vec<u32>,
    topk: TopK,
}

impl Scratch {
    fn new(acc_len: usize, k: usize) -> Self {
        Self {
            sums: vec![0.0; acc_len],
            touched: Vec::with_capacity(1024),
            topk: TopK::new(k),
        }
    }

    /// Accumulates row `t` of `m1 · m2` over all columns, then drains it
    /// through scoring, selection and top-K. Returns the result count.
    ///
    /// # Safety
    /// `t` passed `validate`, and `self.sums.len() == m2.n_cols()`.
    unsafe fn single_block_row(
        &mut self,
        inp: &Inputs,
        t: usize,
        cols: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        // SAFETY: t < m1.n_rows; m1 indices < m1.n_cols == m2.n_rows;
        // m2 indices < m2.n_cols == sums.len() (all checked at construction/validate).
        unsafe {
            let (u_cols, u_vals) = inp.m1.row_unchecked(t);
            for (&u, &v1) in u_cols.iter().zip(u_vals) {
                let (c_cols, c_vals) = inp.m2.row_unchecked(u as usize);
                for (&c, &v2) in c_cols.iter().zip(c_vals) {
                    let s = self.sums.get_unchecked_mut(c as usize);
                    if *s == 0.0 {
                        self.touched.push(c as u32);
                    }
                    *s += v2 * v1;
                }
            }
        }
        // SAFETY: as above; block offset 0, so local col == real col.
        unsafe { self.drain(inp, t, 0) };
        self.topk.drain_into(cols, values)
    }

    /// Scores every touched column, pushes the survivors to top-K and
    /// clears the accumulator for the next row/block.
    ///
    /// # Safety
    /// `t` passed `validate`; every touched column + `offset` is a valid output column.
    #[inline]
    unsafe fn drain(&mut self, inp: &Inputs, t: usize, offset: usize) {
        let p = &inp.params;
        for &local in &self.touched {
            // SAFETY: touched columns are < sums.len().
            let s = unsafe { self.sums.get_unchecked_mut(local as usize) };
            let xy = std::mem::take(s);
            let col = offset + local as usize;
            let col_i = col as i32;
            // SAFETY: t < rows of each selection (validate).
            let keep = unsafe {
                !inp.filter.is_some_and(|f| f.contains_unchecked(t, col_i))
                    && inp
                        .target_cols
                        .is_none_or(|g| g.contains_unchecked(t, col_i))
            };
            if keep {
                // SAFETY: t and col in range of the active norms (validate).
                let val = unsafe { p.score(&inp.norms, t, col, xy) };
                if val >= p.threshold {
                    self.topk.push(col_i, val);
                }
            }
        }
        self.touched.clear();
    }
}
