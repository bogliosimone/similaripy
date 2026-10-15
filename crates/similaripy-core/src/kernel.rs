//! Top-K similarity kernel: the 0.7.0 algorithm (dense accumulator + list of
//! touched columns, heap top-K), rows in parallel with rayon.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    /// Output column ids: kernel column `c` is written as `col_map[c]`. A
    /// permutation of `0..m2.n_cols()` (m2 columns renumbered by the caller), or
    /// `None` to keep the ids.
    pub col_map: Option<&'a [i32]>,
}

/// The results, rows in `targets` order: row `i` has `counts[i]` entries,
/// sorted by column, packed one after another in `cols`/`values` (no padding).
pub struct Output {
    pub cols: Vec<i32>,
    pub values: Vec<f32>,
    pub counts: Vec<u32>,
}

/// Computes the top-`k` scores for every target row on the current rayon pool.
/// `block_size == 0` disables column blocking. `on_row` runs on the worker
/// after each finished row (progress reporting; pass `&|| {}` for none).
pub fn compute<F: Fn() + Sync>(
    inp: &Inputs,
    k: usize,
    block_size: usize,
    on_row: &F,
) -> Result<Output, Error> {
    let n_slots = validate(inp, k)?;
    let counts = vec![0; inp.targets.len()];
    if n_slots == 0 {
        let (cols, values) = (Vec::new(), Vec::new());
        return Ok(Output {
            cols,
            values,
            counts,
        });
    }
    let n_cols = inp.m2.n_cols();
    let blocked = block_size > 0 && n_cols > block_size;
    // The blocked path binary-searches m2 rows. 0.7.0 silently gives wrong
    // results here; the Step 2 prep will always sort instead.
    if blocked && !inp.m2.indices_sorted() {
        return Err(Error::InvalidInput(
            "m2: indices not sorted within rows (needed for column blocking)".into(),
        ));
    }
    let acc_len = if blocked { block_size } else { n_cols };
    // Padded: row i owns slots [i*k, (i+1)*k) until `compact`. Zeroed = calloc,
    // so pages a row never reaches are never touched.
    let mut out = Output {
        cols: vec![0; n_slots],
        values: vec![0.0; n_slots],
        counts,
    };

    // SAFETY: `validate` checked every target against all row-indexed inputs and
    // the selection widths, the scratch accumulator is `acc_len` long, its marks
    // are built for these selections, and m2 rows are sorted when blocked.
    let row = |sc: &mut Scratch, t: i32, cols: &mut [i32], values: &mut [f32]| unsafe {
        sc.mark_selections(inp, t as usize, true);
        let n = if blocked {
            sc.blocked_row(inp, t as usize, block_size, cols, values)
        } else {
            sc.single_block_row(inp, t as usize, cols, values)
        };
        sc.mark_selections(inp, t as usize, false);
        n as u32 // n <= k <= n_cols <= i32::MAX (validate)
    };
    // `broadcast` + an atomic row counter = OpenMP's `schedule(dynamic, 1)`, one
    // scratch per worker. Rayon's adaptive splitting was +65..86% on ML-32M:
    // popularity-ordered rows put most of the work in the first few chunks (D9).
    let next = AtomicUsize::new(0);
    let shared = SharedOutput::new(&mut out, k);
    rayon::broadcast(|_| {
        let mut sc = Scratch::new(acc_len, k, inp);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&t) = inp.targets.get(i) else { break };
            // SAFETY: the counter hands out each i exactly once, and i < len(targets).
            let (cols, values, count) = unsafe { shared.chunk(i) };
            *count = row(&mut sc, t, cols, values);
            on_row();
        }
    });
    compact(&mut out, k);
    Ok(out)
}

/// Packs each row's results right after the previous row's, dropping the
/// padding, then gives the unused tail back to the allocator. One serial
/// pass; rows already in place (all rows before them full) are skipped.
fn compact(out: &mut Output, k: usize) {
    let mut end = 0;
    for (i, &n) in out.counts.iter().enumerate() {
        let (start, n) = (i * k, n as usize);
        if start != end {
            out.cols.copy_within(start..start + n, end);
            out.values.copy_within(start..start + n, end);
        }
        end += n;
    }
    // A shrinking realloc: in place for large blocks on macOS (Step 3c part b
    // review), so the peak stays the padded size, never padded + packed.
    out.cols.truncate(end);
    out.values.truncate(end);
    out.cols.shrink_to_fit();
    out.values.shrink_to_fit();
}

/// Output buffers shared by the workers. Each row index is claimed
/// by one worker only, so the `k`-slot chunks they write never overlap.
struct SharedOutput<'a> {
    cols: *mut i32,
    values: *mut f32,
    counts: *mut u32,
    k: usize,
    _out: PhantomData<&'a mut Output>,
}

// SAFETY: workers only write disjoint chunks (see `chunk`).
unsafe impl Sync for SharedOutput<'_> {}

impl<'a> SharedOutput<'a> {
    fn new(out: &'a mut Output, k: usize) -> Self {
        Self {
            cols: out.cols.as_mut_ptr(),
            values: out.values.as_mut_ptr(),
            counts: out.counts.as_mut_ptr(),
            k,
            _out: PhantomData,
        }
    }

    /// # Safety
    /// `i < len(targets)`, and no other live reference to chunk `i` exists.
    #[allow(clippy::mut_from_ref)]
    unsafe fn chunk(&self, i: usize) -> (&mut [i32], &mut [f32], &mut u32) {
        // SAFETY: cols/values are len(targets) * k long, counts len(targets)
        // (`compute`), chunk i is exclusive (caller).
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.cols.add(i * self.k), self.k),
                std::slice::from_raw_parts_mut(self.values.add(i * self.k), self.k),
                &mut *self.counts.add(i),
            )
        }
    }
}

/// Checks the inputs; returns the output slot count, `len(targets) * k`.
fn validate(inp: &Inputs, k: usize) -> Result<usize, Error> {
    let bad = |msg: &str| Err(Error::InvalidInput(msg.into()));
    let (m1, m2, p, n) = (&inp.m1, &inp.m2, &inp.params, &inp.norms);
    if m1.n_cols() != m2.n_rows() {
        return bad("m1 columns != m2 rows");
    }
    let Some(n_slots) = inp.targets.len().checked_mul(k) else {
        return bad("output too large: len(targets) * k overflows usize");
    };
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
    // Output column ids are i32 (D11).
    if n_cols > i32::MAX as usize {
        return bad("more than i32::MAX output columns");
    }
    if inp.col_map.is_some_and(|m| m.len() != n_cols) {
        return bad("col_map length != m2 columns");
    }
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
        if sel.n_cols() != n_cols {
            return bad("column selection columns != m2 columns");
        }
    }
    Ok(n_slots)
}

/// Row `t`'s norm terms, read once per drain instead of once per column (the
/// sums reset in the column loop could alias them, so LLVM reloaded them). Only
/// the terms whose weight is non-zero are read; the others stay 0 and unused.
#[derive(Clone, Copy, Default)]
struct RowTerms {
    tversky: f32,
    cosine: f32,
    depop: f32,
}

// Scoring modes for `Params::score`, chosen once per drain (`Params::mode`) so
// the per-column weight tests drop out of the common workloads. A `u8`: enum
// const generics aren't stable.

/// Raw `xy_pow`: no denominator term, no shrink (dot product).
const DOT: u8 = 0;
/// `l2` only, stabilized shrink allowed (cosine, asymmetric cosine).
const COSINE: u8 = 1;
/// Any weights: the general formula, and the reference the others match bit for bit.
const FULL: u8 = 2;

impl Params {
    /// `DOT` without any term or shrink, `COSINE` with `l2` and neither `l1`,
    /// `l3` nor a bayesian shrink, `FULL` otherwise. A bayesian shrink is `FULL`
    /// on purpose: in the small `COSINE` loop LLVM if-converts its test and runs
    /// the second fdiv on every column, even with the shrink at 0;
    /// in `DOT` it would cost plain dot a branch per column.
    fn mode(&self) -> u8 {
        if !self.normalizes() {
            DOT
        } else if self.l1 == 0.0 && self.l3 == 0.0 && self.bayesian_shrink == 0.0 && self.l2 != 0.0
        {
            COSINE
        } else {
            FULL
        }
    }

    /// # Safety
    /// `row` is in range of every norm pair whose weight is non-zero.
    unsafe fn row_terms(&self, n: &Norms, row: usize) -> RowTerms {
        // SAFETY: caller contract, checked in `validate`.
        unsafe {
            RowTerms {
                tversky: if self.l1 != 0.0 {
                    *n.x_tversky.get_unchecked(row)
                } else {
                    0.0
                },
                cosine: if self.l2 != 0.0 {
                    *n.x_cosine.get_unchecked(row)
                } else {
                    0.0
                },
                depop: if self.l3 != 0.0 {
                    *n.x_depop.get_unchecked(row)
                } else {
                    0.0
                },
            }
        }
    }

    fn normalizes(&self) -> bool {
        self.l1 != 0.0
            || self.l2 != 0.0
            || self.l3 != 0.0
            || self.stabilized_shrink != 0.0
            || self.bayesian_shrink != 0.0
    }

    /// The C++ `computeSimilarity` arithmetic, in the same order, except two 0.7.0
    /// bugs: `a1` applies without normalization too (F14), and bayesian shrink
    /// without a denominator term scales xy_pow instead of returning 0 (F1).
    ///
    /// # Safety
    /// `col` is in range of every norm pair whose weight is non-zero; `x` is
    /// `row_terms` of the row.
    #[inline(always)]
    /// `POW` must equal `a1 != 1.0` (see `Scratch::drain`), `MODE` must be
    /// `self.mode()` or `FULL`. `DOT` and `COSINE` give the same bits as `FULL`:
    /// they only skip terms that are +0.0 (which flips a -0.0 den to +0.0, both
    /// take den == 0).
    unsafe fn score<const POW: bool, const MODE: u8>(
        &self,
        n: &Norms,
        x: &RowTerms,
        col: usize,
        xy: f32,
    ) -> f32 {
        let xy_pow = if POW { xy.powf(self.a1) } else { xy };
        if MODE == DOT {
            return xy_pow;
        }
        if MODE == COSINE {
            // SAFETY: caller contract, checked in `validate`.
            let cs = self.l2 * (x.cosine * unsafe { n.y_cosine.get_unchecked(col) });
            let den = cs + self.stabilized_shrink;
            // den == 0 with l2 != 0: 0, as FULL.
            return if den != 0.0 { xy_pow / den } else { 0.0 };
        }
        if !self.normalizes() {
            return xy_pow;
        }
        let (mut tv, mut cs, mut dp) = (0.0, 0.0, 0.0);
        // SAFETY: caller contract, checked in `validate`.
        unsafe {
            if self.l1 != 0.0 {
                let y = *n.y_tversky.get_unchecked(col);
                tv = self.l1 * (self.t1 * (x.tversky - xy) + self.t2 * (y - xy) + xy);
            }
            if self.l2 != 0.0 {
                cs = self.l2 * (x.cosine * n.y_cosine.get_unchecked(col));
            }
            if self.l3 != 0.0 {
                dp = self.l3 * (x.depop * n.y_depop.get_unchecked(col));
            }
        }
        // den != 0 first: the hot path stays the 0.7.0 one. den == 0 means either
        // no denominator term (only bayesian shrink: the raw xy_pow) or terms
        // summing to 0: 0. The Python layer keeps every weight >= 0 and, on any
        // input sign, den > 0 (tversky rule, D45), bar f32 underflow; the guard
        // stays for direct core callers.
        let den = tv + cs + dp + self.stabilized_shrink;
        let mut val = if den != 0.0 {
            xy_pow / den
        } else if self.l1 == 0.0
            && self.l2 == 0.0
            && self.l3 == 0.0
            && self.stabilized_shrink == 0.0
        {
            xy_pow
        } else {
            0.0
        };
        if self.bayesian_shrink != 0.0 {
            val *= xy_pow / (xy_pow + self.bayesian_shrink);
        }
        val
    }
}

/// One row of a column selection (filter or target_cols) as a bitset over the
/// output columns, set before the row and cleared after it: one bit test per
/// candidate instead of a binary search of the selection row, which cost +40%
/// kernel time with a seen-items filter (ML-32M, Yambda 50M; Step 3). Empty,
/// and never read, without a selection.
struct RowMarks {
    bits: Vec<u64>,
}

impl RowMarks {
    fn new(sel: Option<CsrPattern>, n_cols: usize) -> Self {
        let words = if sel.is_some() {
            n_cols.div_ceil(64)
        } else {
            0
        };
        Self {
            bits: vec![0; words],
        }
    }

    /// Sets (`on`) or clears the bits of row `t`. Clearing zeroes whole words:
    /// only this row's bits are ever set.
    ///
    /// # Safety
    /// `t < sel.n_rows()`, and `self` was built for `sel` (`sel.n_cols()` columns).
    #[inline]
    unsafe fn mark(&mut self, sel: &CsrPattern, t: usize, on: bool) {
        // SAFETY: t in range (caller); columns < n_cols, so c / 64 < bits.len().
        unsafe {
            for &c in sel.row_unchecked(t) {
                let word = self.bits.get_unchecked_mut(c as usize / 64);
                *word = if on { *word | (1 << (c % 64)) } else { 0 };
            }
        }
    }

    /// # Safety
    /// `col < n_cols` of the selection `self` was built for.
    #[inline(always)]
    unsafe fn contains(&self, col: usize) -> bool {
        // SAFETY: caller contract.
        unsafe { (self.bits.get_unchecked(col / 64) >> (col % 64)) & 1 != 0 }
    }
}

/// Per-worker state, reused across rows.
struct Scratch {
    sums: Vec<f32>,
    touched: Vec<u32>,
    /// Blocked path: `(pos, end)` in m2 per entry of the current m1 row.
    cursor: Vec<(u32, u32)>,
    topk: TopK,
    filter: RowMarks,
    target_cols: RowMarks,
}

impl Scratch {
    fn new(acc_len: usize, k: usize, inp: &Inputs) -> Self {
        let n_cols = inp.m2.n_cols();
        Self {
            sums: vec![0.0; acc_len],
            touched: Vec::with_capacity(1024),
            cursor: Vec::new(),
            topk: TopK::new(k),
            filter: RowMarks::new(inp.filter, n_cols),
            target_cols: RowMarks::new(inp.target_cols, n_cols),
        }
    }

    /// Sets (`on`) or clears row `t` of both selections.
    ///
    /// # Safety
    /// `t` passed `validate`, and `self` was built for `inp`.
    unsafe fn mark_selections(&mut self, inp: &Inputs, t: usize, on: bool) {
        // SAFETY: t < rows of each selection (validate), marks sized for them (new).
        unsafe {
            if let Some(f) = &inp.filter {
                self.filter.mark(f, t, on);
            }
            if let Some(g) = &inp.target_cols {
                self.target_cols.mark(g, t, on);
            }
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
                accumulate(&mut self.sums, &mut self.touched, c_cols, c_vals, v1, 0);
            }
            // block offset 0, so local col == real col.
            self.drain(inp, t, 0);
        }
        self.topk.drain_into(inp.col_map, cols, values)
    }

    /// Same as `single_block_row`, but over column blocks of `block` columns:
    /// m1 row `t` is re-read once per block, and only the m2 entries inside
    /// the block are accumulated, so the accumulator stays cache-sized.
    ///
    /// # Safety
    /// `t` passed `validate`, `self.sums.len() == block`, and m2 rows are sorted.
    unsafe fn blocked_row(
        &mut self,
        inp: &Inputs,
        t: usize,
        block: usize,
        cols: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        let n_cols = inp.m2.n_cols();
        // SAFETY: as in `single_block_row`; m1 row t is in range.
        let (u_cols, u_vals) = unsafe { inp.m1.row_unchecked(t) };
        // Per m1 entry, the unread rest of its m2 row. Blocks run in column order
        // and rows are sorted, so each block's span starts at the cursor: no `lo`
        // search, and each m2 row is read once, front to back (decisions D34).
        // Positions fit u32: indptr is i32, so nnz <= i32::MAX (`CsrView::new`).
        self.cursor.clear();
        self.cursor.extend(u_cols.iter().map(|&u| {
            // SAFETY: m1 indices < m2.n_rows.
            let (a, b) = unsafe { inp.m2.row_span_unchecked(u as usize) };
            (a as u32, b as u32)
        }));
        for start in (0..n_cols).step_by(block) {
            let hi_col = (start + block).min(n_cols) as i32;
            for (cur, &v1) in self.cursor.iter_mut().zip(u_vals) {
                let (pos, end) = (cur.0 as usize, cur.1 as usize);
                // SAFETY: pos..end is a suffix of an m2 row span.
                let (c_cols, c_vals) = unsafe { inp.m2.span_unchecked(pos, end) };
                let (Some(&first), Some(&last)) = (c_cols.first(), c_cols.last()) else {
                    continue; // row done
                };
                if first >= hi_col {
                    continue; // row resumes in a later block
                }
                // All of c_cols is >= start (cursor), so only the block end is searched.
                let n = if last < hi_col {
                    c_cols.len()
                } else {
                    lower_bound(c_cols, hi_col)
                };
                // SAFETY: start <= c < hi_col, so c - start < block == sums.len().
                unsafe {
                    accumulate(
                        &mut self.sums,
                        &mut self.touched,
                        &c_cols[..n],
                        &c_vals[..n],
                        v1,
                        start,
                    )
                };
                cur.0 = (pos + n) as u32;
            }
            if !self.touched.is_empty() {
                // SAFETY: touched + start < hi_col <= n_cols.
                unsafe { self.drain(inp, t, start) };
            }
        }
        self.topk.drain_into(inp.col_map, cols, values)
    }

    /// Scores every touched column, pushes the survivors to top-K and
    /// clears the accumulator for the next row/block.
    ///
    /// # Safety
    /// `t` passed `validate`; every touched column + `offset` is a valid output column.
    // Out of line on purpose: inlined into the row functions, the drain loops ran
    // out of registers and reloaded the norm pointers from the stack per column.
    // One call per drain is noise.
    #[inline(never)]
    unsafe fn drain(&mut self, inp: &Inputs, t: usize, offset: usize) {
        macro_rules! drain_with {
            ($pow:literal, $mode:ident) => {
                drain_impl::<$pow, $mode>(
                    &mut self.sums,
                    &self.touched,
                    &mut self.topk,
                    (&self.filter, &self.target_cols),
                    inp,
                    t,
                    offset,
                )
            };
        }
        // Branch once per drain: inside the loop LLVM speculates `powf` (a pure
        // intrinsic) and calls it for every column even when a1 == 1. The mode
        // takes the weight tests out of the loop for dot and cosine.
        // SAFETY: caller contract, forwarded.
        unsafe {
            match (inp.params.a1 != 1.0, inp.params.mode()) {
                (true, DOT) => drain_with!(true, DOT),
                (true, COSINE) => drain_with!(true, COSINE),
                (true, _) => drain_with!(true, FULL),
                (false, DOT) => drain_with!(false, DOT),
                (false, COSINE) => drain_with!(false, COSINE),
                (false, _) => drain_with!(false, FULL),
            }
        }
        self.touched.clear();
    }
}

/// `Scratch::drain`'s loop. A free function over slices and locals, like
/// `accumulate`: as a `&mut self` method LLVM reloaded `sums`' pointer every
/// column (the reset store might alias `Scratch`), and the top-K reject read
/// the heap's length, pointer and root from memory. Here the row's norm terms
/// and the lowest acceptable score (`floor`) stay in registers.
///
/// # Safety
/// As `Scratch::drain`; `MODE` = `inp.params.mode()`, `sel` = (filter,
/// target_cols) marked for row `t`.
#[inline(always)]
unsafe fn drain_impl<const POW: bool, const MODE: u8>(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    sel: (&RowMarks, &RowMarks),
    inp: &Inputs,
    t: usize,
    offset: usize,
) {
    let p = &inp.params;
    // SAFETY: t in range of the active norms (validate).
    let x = unsafe { p.row_terms(&inp.norms, t) };
    // `val >= floor` is the threshold test while the heap fills and push's
    // root reject once it is full (the root passed the threshold): same pushes.
    let mut floor = topk.floor(p.threshold);
    for &local in touched {
        // SAFETY: touched columns are < sums.len().
        let s = unsafe { sums.get_unchecked_mut(local as usize) };
        let xy = std::mem::take(s);
        let col = offset + local as usize;
        // SAFETY: col < n_cols (caller), row t marked (`mark_selections`).
        let keep = unsafe {
            !(inp.filter.is_some() && sel.0.contains(col))
                && (inp.target_cols.is_none() || sel.1.contains(col))
        };
        if keep {
            // SAFETY: col in range of the active norms (validate).
            let val = unsafe { p.score::<POW, MODE>(&inp.norms, &x, col, xy) };
            if val >= floor {
                topk.push(col as i32, val);
                floor = topk.floor(p.threshold);
            }
        }
    }
}

/// Adds `v1 * v2` for each `(c, v2)` into `sums[c - offset]`, recording first
/// touches. A free function over a slice and a local write cursor so LLVM keeps
/// the pointers in registers: as a `&mut self` method it reloaded `sums` and the
/// `touched` header from the stack every element (decisions D14).
///
/// A column whose sum is exactly 0.0 (an underflowed product) is pushed again on
/// its next hit, as in 0.7.0; drain then scores the duplicate with xy = 0 and
/// `TopK::drain_into` drops that 0.0 from the output. There is at most one write
/// per element, so `reserve` covers it.
///
/// # Safety
/// Every `c - offset` is `< sums.len()`.
#[inline(always)]
unsafe fn accumulate(
    sums: &mut [f32],
    touched: &mut Vec<u32>,
    c_cols: &[i32],
    c_vals: &[f32],
    v1: f32,
    offset: usize,
) {
    // At most one push per element, so one reserve covers the whole segment.
    touched.reserve(c_cols.len());
    let buf = touched.as_mut_ptr();
    let mut n = touched.len();
    for (&c, &v2) in c_cols.iter().zip(c_vals) {
        let local = c as usize - offset;
        // SAFETY: caller contract; n < capacity (reserve above).
        unsafe {
            let s = sums.get_unchecked_mut(local);
            let old = *s; // reuse: the touched store may alias, re-reading costs a load
            // Branchless first touch: always write the slot, keep it only if new.
            // The branch was near-random at Yambda shape (39% first touches,
            // mispredicted; Step 3c profile); the extra store is sequential.
            buf.add(n).write(local as u32);
            n += (old == 0.0) as usize;
            *s = old + v2 * v1;
        }
    }
    // SAFETY: entries [len, n) were written above, n <= capacity.
    unsafe { touched.set_len(n) };
}

/// First index `i` with `s[i] >= x` (`s` sorted). Branchy on purpose, like
/// libstdc++ `lower_bound`: `partition_point` compiles branchless (`csel`), which
/// serializes the probes' cache misses; with a branch the CPU speculates the next
/// probe's load. Matters once m2 rows miss to DRAM (Yambda 500M, decisions D14).
#[inline]
fn lower_bound(s: &[i32], x: i32) -> usize {
    let (mut lo, mut len) = (0, s.len());
    while len > 0 {
        let half = len / 2;
        // SAFETY: lo + half < lo + len <= s.len().
        if unsafe { *s.get_unchecked(lo + half) } < x {
            lo += half + 1;
            len -= half + 1;
        } else {
            // Empty asm = optimization barrier, so LLVM can't if-convert this into
            // `csel`/`cmov` again; std has no stable "keep this branch" hint. Only on
            // targets with stable `asm!` (the wheel matrix), and not under Miri (no
            // inline asm support); a plain loop elsewhere.
            #[cfg(all(
                not(miri),
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "arm",
                    target_arch = "aarch64"
                )
            ))]
            // SAFETY: emits no instructions.
            unsafe {
                std::arch::asm!("", options(nomem, nostack, preserves_flags))
            };
            len = half;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn lower_bound_matches_partition_point(
            mut s in proptest::collection::vec(-50i32..50, 0..64),
            x in -60i32..60,
        ) {
            s.sort_unstable();
            prop_assert_eq!(lower_bound(&s, x), s.partition_point(|&c| c < x));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // Branchless first touch == the branchy 0.7.0 loop, including sums that
        // cancel back to exactly 0.0 (signed data, re-pushed on the next hit).
        #[test]
        fn accumulate_matches_branchy_push(
            segs in proptest::collection::vec(
                (proptest::collection::vec((0i32..8, -3i8..=3), 0..12), -2i8..=2),
                0..6,
            ),
            offset in 0usize..3,
        ) {
            let (mut sums, mut touched) = (vec![0.0f32; 8], Vec::new());
            let (mut ref_sums, mut ref_touched) = (vec![0.0f32; 8], Vec::new());
            for (seg, v1) in &segs {
                let cols: Vec<i32> = seg.iter().map(|&(c, _)| c + offset as i32).collect();
                let vals: Vec<f32> = seg.iter().map(|&(_, v)| f32::from(v)).collect();
                let v1 = f32::from(*v1);
                // SAFETY: c + offset - offset = c < 8 = sums.len().
                unsafe { accumulate(&mut sums, &mut touched, &cols, &vals, v1, offset) };
                for (&c, &v2) in cols.iter().zip(&vals) {
                    let local = c as usize - offset;
                    if ref_sums[local] == 0.0 {
                        ref_touched.push(local as u32);
                    }
                    ref_sums[local] += v2 * v1;
                }
            }
            prop_assert_eq!(touched, ref_touched);
            prop_assert_eq!(sums, ref_sums);
        }
    }

    #[test]
    fn accumulate_repushes_a_sum_that_cancels_to_zero() {
        let (mut sums, mut touched) = (vec![0.0f32; 4], Vec::new());
        // SAFETY: all columns < 4, offset 0.
        unsafe {
            accumulate(&mut sums, &mut touched, &[1, 3], &[2.0, 1.0], 1.0, 0);
            accumulate(&mut sums, &mut touched, &[1, 2], &[-2.0, 1.0], 1.0, 0); // col 1 back to 0
            accumulate(&mut sums, &mut touched, &[1], &[5.0], 1.0, 0); // pushed again
        }
        assert_eq!(touched, [1, 3, 2, 1]);
        assert_eq!(sums, [0.0, 5.0, 1.0, 1.0]);
    }

    /// `score` in `p.mode()` vs `FULL`, as bits.
    fn mode_and_full_bits<const POW: bool>(p: &Params, n: &Norms, xy: f32) -> (u32, u32) {
        // SAFETY: row 0 and col 0 exist in every norm slice.
        unsafe {
            let x = p.row_terms(n, 0);
            let m = match p.mode() {
                DOT => p.score::<POW, DOT>(n, &x, 0, xy),
                COSINE => p.score::<POW, COSINE>(n, &x, 0, xy),
                _ => p.score::<POW, FULL>(n, &x, 0, xy),
            };
            (m.to_bits(), p.score::<POW, FULL>(n, &x, 0, xy).to_bits())
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // Bit-identical output across modes: the mode only skips +0.0 terms.
        #[test]
        fn score_mode_matches_full_bitwise(
            w in proptest::array::uniform5(prop_oneof![Just(0.0f32), Just(-0.0f32), 0.1f32..2.0]),
            t in proptest::array::uniform2(0.0f32..1.0),
            pow in any::<bool>(),
            norms in proptest::array::uniform6(prop_oneof![Just(0.0f32), 0.1f32..3.0]),
            xy in prop_oneof![Just(0.0f32), -5.0f32..5.0],
        ) {
            // Not under Miri: it adds random error to powf (and random NaN bits),
            // so two equal calls can differ there. Natively powf is deterministic.
            let pow = pow && !cfg!(miri);
            let p = Params {
                a1: if pow { 0.5 } else { 1.0 },
                l1: w[0], l2: w[1], l3: w[2], t1: t[0], t2: t[1],
                stabilized_shrink: w[3], bayesian_shrink: w[4],
                threshold: 0.0,
            };
            let v: Vec<[f32; 1]> = norms.iter().map(|&x| [x]).collect();
            let n = Norms {
                x_tversky: &v[0], y_tversky: &v[1], x_cosine: &v[2],
                y_cosine: &v[3], x_depop: &v[4], y_depop: &v[5],
            };
            let (m, f) = if pow {
                mode_and_full_bits::<true>(&p, &n, xy)
            } else {
                mode_and_full_bits::<false>(&p, &n, xy)
            };
            prop_assert_eq!(m, f, "mode {} xy {}", p.mode(), xy);
        }
    }

    #[test]
    fn mode_per_similarity() {
        let p = |l1: f32, l2: f32, l3: f32, ss: f32, bs: f32| Params {
            l1,
            l2,
            l3,
            stabilized_shrink: ss,
            bayesian_shrink: bs,
            ..Params::default()
        };
        assert_eq!(p(0.0, 0.0, 0.0, 0.0, 0.0).mode(), DOT); // dot product
        assert_eq!(p(0.0, 1.0, 0.0, 0.0, 0.0).mode(), COSINE); // cosine, asymmetric cosine
        assert_eq!(p(0.0, 1.0, 0.0, 2.0, 0.0).mode(), COSINE); // + stabilized shrink
        assert_eq!(p(0.0, 1.0, 0.0, 0.0, 2.0).mode(), FULL); // + bayesian shrink
        assert_eq!(p(0.0, 0.0, 1.0, 0.0, 0.0).mode(), FULL); // rp3beta
        assert_eq!(p(1.0, 0.0, 0.0, 0.0, 0.0).mode(), FULL); // tversky, jaccard, dice
        assert_eq!(p(0.0, 0.0, 0.0, 0.0, 2.0).mode(), FULL); // dot + bayesian: FULL on purpose
        assert_eq!(p(-0.0, 1.0, -0.0, 0.0, 0.0).mode(), COSINE); // -0.0 weights = 0, as FULL's tests
    }
}
