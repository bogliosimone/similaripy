//! Top-K similarity kernel: dense accumulator + list of touched columns, top-K
//! buffer; rows in parallel with rayon.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::Error;
use crate::csr::{CsrPattern, CsrView};
use crate::topk::TopK;

/// Per-row (`x_*`, indexed by target row) and per-column (`y_*`, indexed by
/// output column) normalization terms. A pair is needed when its weight in
/// `Params` is non-zero, and ignored otherwise (None: unused).
#[derive(Clone, Copy, Default)]
pub struct Norms<'a> {
    pub x_tversky: Option<&'a [f32]>,
    pub y_tversky: Option<&'a [f32]>,
    pub x_cosine: Option<&'a [f32]>,
    pub y_cosine: Option<&'a [f32]>,
    pub x_depop: Option<&'a [f32]>,
    pub y_depop: Option<&'a [f32]>,
}

/// `Norms` resolved once, before the row loop (`Inputs::resolved`): plain
/// slices, empty when unused, so the hot loop reads no `Option`.
#[derive(Clone, Copy)]
struct NormSlices<'a> {
    x_tversky: &'a [f32],
    y_tversky: &'a [f32],
    x_cosine: &'a [f32],
    y_cosine: &'a [f32],
    x_depop: &'a [f32],
    y_depop: &'a [f32],
}

impl<'a> Norms<'a> {
    fn resolved(&self) -> NormSlices<'a> {
        let s = |x: Option<&'a [f32]>| x.unwrap_or(&[]);
        NormSlices {
            x_tversky: s(self.x_tversky),
            y_tversky: s(self.y_tversky),
            x_cosine: s(self.x_cosine),
            y_cosine: s(self.y_cosine),
            x_depop: s(self.x_depop),
            y_depop: s(self.y_depop),
        }
    }
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
pub struct Inputs<'a, N = Norms<'a>> {
    pub m1: CsrView<'a>,
    pub m2: CsrView<'a>,
    pub targets: &'a [i32],
    pub norms: N,
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

/// `Inputs` as the kernel reads them: norms resolved (`NormSlices`).
type Ready<'a> = Inputs<'a, NormSlices<'a>>;

impl<'a> Inputs<'a> {
    fn resolved(&self) -> Ready<'a> {
        Inputs {
            m1: self.m1,
            m2: self.m2,
            targets: self.targets,
            norms: self.norms.resolved(),
            params: self.params,
            filter: self.filter,
            target_cols: self.target_cols,
            col_map: self.col_map,
        }
    }
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
    // At most one result per column: a larger k would only pad the output.
    let k = k.min(inp.m2.n_cols());
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
    let inp = &inp.resolved();
    let n_cols = inp.m2.n_cols();
    let blocked = block_size > 0 && n_cols > block_size;
    // The blocked path stops each m2 row at the first column past the block end,
    // so unsorted rows would give wrong results silently. The Python prep sorts
    // them; this catches direct callers.
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
    // popularity-ordered rows put most of the work in the first few chunks.
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
    // A shrinking realloc: in place for large blocks (checked on macOS: same
    // pointer, footprint drops), so the peak stays the padded size, never
    // padded + packed.
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
    // Output column ids are i32.
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
        if w == 0.0 {
            continue;
        }
        let (Some(x), Some(y)) = (x, y) else {
            return Err(Error::InvalidInput(format!("{name} norms missing")));
        };
        if x.len() < rows_needed || y.len() < n_cols {
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
    /// in `DOT` it would cost plain dot a branch per column. The Python prep
    /// sends cosine and rp3beta without shrink here as `DOT` (pre-scaled inputs,
    /// `_product_form`) and depop-only weights through the cosine slots
    /// (`_depop_as_cosine`).
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
    unsafe fn row_terms(&self, n: &NormSlices, row: usize) -> RowTerms {
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

    /// The s_plus score of one column. `a1` applies with or without a
    /// denominator; with no denominator term the value is `xy_pow`, times the
    /// bayesian factor if set.
    ///
    /// K1: `bound` is `k1_bound(floor)`. With a denominator (`COSINE`,
    /// `FULL`) a column whose `xy_pow < bound * den` (den > 0) scores below
    /// `floor` for sure and returns NaN before any division; every other
    /// column takes the exact path. The caller's `val >= floor` drops the NaN
    /// as it would have dropped the score (and LLVM folds that compare: the
    /// reject branches straight to the next column): same pushes, same bits.
    /// NaN `bound`: never rejects.
    ///
    /// # Safety
    /// `col` is in range of every norm pair whose weight is non-zero; `x` is
    /// `row_terms` of the row. `POW` must equal `a1 != 1.0` (see
    /// `Scratch::drain`), `MODE` must be `self.mode()` or `FULL`. `DOT` and
    /// `COSINE` give the same bits as `FULL`: they only skip terms that are +0.0
    /// (which flips a -0.0 den to +0.0, both take den == 0).
    #[inline(always)]
    unsafe fn score<const POW: bool, const MODE: u8>(
        &self,
        n: &NormSlices,
        x: &RowTerms,
        col: usize,
        xy: f32,
        bound: f32,
    ) -> f32 {
        let xy_pow = if POW { xy.powf(self.a1) } else { xy };
        if MODE == DOT {
            return xy_pow;
        }
        if MODE == COSINE {
            // SAFETY: caller contract, checked in `validate`.
            let cs = self.l2 * (x.cosine * unsafe { n.y_cosine.get_unchecked(col) });
            let den = cs + self.stabilized_shrink;
            if den > 0.0 && xy_pow < bound * den {
                return f32::NAN;
            }
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
        // den != 0 first: the common case. den == 0 means either no denominator
        // term (only bayesian shrink: the raw xy_pow) or terms summing to 0: 0.
        // The Python layer keeps every weight >= 0 and, on any input sign,
        // den > 0 (signed-input tversky rule, `_check_score_rules` in
        // `_s_plus.py`), bar f32 underflow; the guard stays for direct core
        // callers.
        let den = tv + cs + dp + self.stabilized_shrink;
        // With the bayesian factor x / (x + s) (x = xy_pow >= 0, s >= 0; 0 when
        // off): val = x^2 / (den (x + s)), so reject x^2 < bound den (x + s). Most
        // bayesian rejects come from the factor, not the quotient: a test on
        // the quotient alone missed them (rp3beta_bayesian +7..+15%, df1).
        // In f64: the f32 products are exact there and can't over/underflow,
        // so the margin argument of `k1_bound` holds unchanged; s = 0 is the
        // quotient test (x > 0). x < 0 (signed input, where the factor can
        // exceed 1) takes the exact path.
        if den > 0.0 && xy_pow >= 0.0 && {
            let x = f64::from(xy_pow);
            x * x < f64::from(bound) * f64::from(den) * (x + f64::from(self.bayesian_shrink))
        } {
            return f32::NAN;
        }
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
/// kernel time with a seen-items filter (measured on ML-32M and Yambda 50M).
/// Empty, and never read, without a selection.
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
    cursor: Vec<(usize, usize, usize)>,
    topk: TopK,
    filter: RowMarks,
    target_cols: RowMarks,
}

impl Scratch {
    fn new(acc_len: usize, k: usize, inp: &Ready) -> Self {
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
    unsafe fn mark_selections(&mut self, inp: &Ready, t: usize, on: bool) {
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
        inp: &Ready,
        t: usize,
        cols: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        // SAFETY: t < m1.n_rows; m1 indices < m1.n_cols == m2.n_rows;
        // m2 indices < m2.n_cols == sums.len() (all checked at construction/validate).
        unsafe {
            let (u_cols, u_vals) = inp.m1.row_unchecked(t);
            for i in 0..u_cols.len() {
                // Prefetch the head of the m2 row PREFETCH_ROWS ahead: each row starts at an address the
                // hardware prefetcher can't predict; long rows stream after that. M2 Pro vs no prefetch
                // (night13, bits unchanged): ML-32M −4..−5%, Y500M −2..−4%, Y50M flat; short rows gain most.
                if let Some(&un) = u_cols.get(i + PREFETCH_ROWS) {
                    let (pc, pv) = inp.m2.row_unchecked(un as usize);
                    prefetch_span(pc, pv);
                }
                let (u, v1) = (*u_cols.get_unchecked(i), *u_vals.get_unchecked(i));
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
        inp: &Ready,
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
        // search, and each m2 row is read once, front to back.
        self.cursor.clear();
        self.cursor.extend(u_cols.iter().map(|&u| {
            // SAFETY: m1 indices < m2.n_rows.
            unsafe {
                let (pos, end) = inp.m2.row_span_unchecked(u as usize);
                (pos, end, next_col(inp.m2.span_unchecked(pos, end).0, 0))
            }
        }));
        for start in (0..n_cols).step_by(block) {
            let hi = (start + block).min(n_cols);
            for i in 0..self.cursor.len() {
                // Prefetch as in `single_block_row`, the in-block part of the row ahead.
                if let Some(&(ppos, pend, pnext)) = self.cursor.get(i + PREFETCH_ROWS)
                    && pnext < hi
                {
                    // SAFETY: ppos..pend is a suffix of an m2 row span.
                    let (pc, pv) = unsafe { inp.m2.span_unchecked(ppos, pend) };
                    prefetch_span(pc, pv);
                }
                // SAFETY: i < cursor.len() == u_vals.len().
                let (cur, v1) =
                    unsafe { (self.cursor.get_unchecked_mut(i), *u_vals.get_unchecked(i)) };
                let (pos, end, next) = *cur;
                // Cached (as serving's request loop): a row with nothing in this
                // block costs no load from m2.
                if next >= hi {
                    continue;
                }
                // SAFETY: pos..end is a suffix of an m2 row span.
                let (c_cols, c_vals) = unsafe { inp.m2.span_unchecked(pos, end) };
                // Not empty (next < hi). A row inside the block is taken whole; one
                // past its end stops inside the accumulate loop, no search.
                // SAFETY: start <= c < hi, so c - start < block == sums.len().
                let n = unsafe {
                    if hi == n_cols || (c_cols[c_cols.len() - 1] as usize) < hi {
                        accumulate(&mut self.sums, &mut self.touched, c_cols, c_vals, v1, start);
                        c_cols.len()
                    } else {
                        accumulate_below(
                            &mut self.sums,
                            &mut self.touched,
                            c_cols,
                            c_vals,
                            v1,
                            start,
                            hi,
                        )
                    }
                };
                *cur = (pos + n, end, next_col(c_cols, n));
            }
            if !self.touched.is_empty() {
                // SAFETY: touched + start < hi <= n_cols.
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
    unsafe fn drain(&mut self, inp: &Ready, t: usize, offset: usize) {
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
/// the top-K's state from memory. Here the row's norm terms
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
    inp: &Ready,
    t: usize,
    offset: usize,
) {
    let p = &inp.params;
    // SAFETY: t in range of the active norms (validate).
    let x = unsafe { p.row_terms(&inp.norms, t) };
    // `val >= floor` is the threshold test until the top-K's first cut and
    // push's floor reject after it (the k-th best passed the threshold): same pushes.
    let mut floor = topk.floor(p.threshold);
    let k1 = p.bayesian_shrink >= 0.0;
    let mut bound = k1_bound(floor, k1);
    // Back to front: the floor rises sooner, 35-73% fewer top-K replacements
    // (perf gates T0 counters: ML-32M, Y50M, Y500M; cosine, rp3beta). Same
    // result: the top-K keeps a strict total order. Not in the serving drain:
    // there it doubled the batch time (ML-32M, perf gates night 2).
    for &local in touched.iter().rev() {
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
            let val = unsafe { p.score::<POW, MODE>(&inp.norms, &x, col, xy, bound) };
            if val >= floor {
                topk.push(col as i32, val);
                // The floor moves only at a cut: the bound's call is as rare.
                let f = topk.floor(p.threshold);
                if f != floor {
                    floor = f;
                    bound = k1_bound(floor, k1);
                }
            }
        }
    }
}

/// K1's reject bound for `Params::score`: `floor * (1 - 2^-10)` for a normal
/// `floor > 0`, NaN (no reject) otherwise or when `!ok` (a negative or NaN
/// bayesian shrink, which can lift a score past its quotient). Why exact: the
/// bound and its product with den round by <= 2^-23 relative (or below the f32
/// grid that `xy_pow` sits on), so `xy_pow < bound * den` means
/// `xy_pow / den < floor * (1 - 2^-11)`: thousands of ulps below `floor`, which
/// the division's rounding can't cross. The 2^-10 margin only sends that band's
/// columns down the exact path. Not for +inf (k = 0): an xy whose score
/// overflows to +inf still passes.
///
/// Out of line and cold (once per drain and per top-K cut), like `cut_to_k`:
/// its constants stay out of the drain loops' registers.
#[cold]
#[inline(never)]
fn k1_bound(floor: f32, ok: bool) -> f32 {
    if ok && floor > 0.0 && floor.is_normal() {
        floor * (1.0 - 1.0 / 1024.0)
    } else {
        f32::NAN
    }
}

/// Adds `v1 * v2` for each `(c, v2)` into `sums[c - offset]`, recording first
/// touches. A free function over a slice and a local write cursor so LLVM keeps
/// the pointers in registers: as a `&mut self` method it reloaded `sums` and the
/// `touched` header from the stack every element (see benchmarks/README.md,
/// "Codegen-sensitive hot loops").
///
/// A column whose sum is exactly 0.0 (an underflowed product) is pushed again on
/// its next hit; drain then scores the duplicate with xy = 0 and
/// `TopK::drain_into` drops that 0.0 from the output. There is at most one
/// write per element, so `reserve` covers it.
///
/// # Safety
/// Every `c - offset` is `< sums.len()`.
#[inline(always)]
pub(crate) unsafe fn accumulate(
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
            // mispredicted; ~20% of kernel time on Yambda 50M); the extra store is
            // sequential.
            buf.add(n).write(local as u32);
            n += (old == 0.0) as usize;
            *s = old + v2 * v1;
        }
    }
    // SAFETY: entries [len, n) were written above, n <= capacity.
    unsafe { touched.set_len(n) };
}

/// `accumulate` over the leading columns `< hi` of a sorted segment, in one
/// pass, no search first; returns how many it took. Same body and addition
/// order as `accumulate` (same bits).
///
/// # Safety
/// As `accumulate`, for the columns `< hi`.
#[inline(always)]
pub(crate) unsafe fn accumulate_below(
    sums: &mut [f32],
    touched: &mut Vec<u32>,
    c_cols: &[i32],
    c_vals: &[f32],
    v1: f32,
    offset: usize,
    hi: usize,
) -> usize {
    touched.reserve(c_cols.len());
    let buf = touched.as_mut_ptr();
    let mut n = touched.len();
    let mut i = 0;
    for (&c, &v2) in c_cols.iter().zip(c_vals) {
        if c as usize >= hi {
            break;
        }
        let local = c as usize - offset;
        // SAFETY: caller contract; n < capacity (reserve above).
        unsafe {
            let s = sums.get_unchecked_mut(local);
            let old = *s;
            buf.add(n).write(local as u32);
            n += (old == 0.0) as usize;
            *s = old + v2 * v1;
        }
        i += 1;
    }
    // SAFETY: entries [len, n) were written above, n <= capacity.
    unsafe { touched.set_len(n) };
    i
}

/// The column at `cols[n]`, `usize::MAX` past the end (never in a block).
#[inline(always)]
pub(crate) fn next_col(cols: &[i32], n: usize) -> usize {
    cols.get(n).map_or(usize::MAX, |&c| c as usize)
}

/// m2 rows ahead to prefetch, and the most lines prefetched per array (8 lines = 128 entries; longer rows
/// stream on the hardware prefetcher after that).
const PREFETCH_ROWS: usize = 8;
const PREFETCH_LINES: usize = 8;

/// Hints the head of an m2 row span (both arrays) into L1.
#[inline(always)]
fn prefetch_span(cols: &[i32], vals: &[f32]) {
    let bytes = cols.len().min(PREFETCH_LINES * 16) * 4;
    for off in (0..bytes).step_by(64) {
        prefetch(cols.as_ptr().cast::<u8>().wrapping_add(off));
        prefetch(vals.as_ptr().cast::<u8>().wrapping_add(off));
    }
}

/// A read prefetch hint into L1: never faults, no-op where the target has none here.
// ponytail: aarch64 and x86_64 only (the release targets); others skip the hint, same results.
#[inline(always)]
pub(crate) fn prefetch(p: *const u8) {
    #[cfg(all(target_arch = "aarch64", not(miri)))] // Miri runs no inline asm
    // SAFETY: a prefetch hint never faults and writes nothing.
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    // SAFETY: as above.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(p.cast::<i8>());
    }
    #[cfg(any(miri, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
    let _ = p;
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // Branchless first touch == the plain branchy loop, including sums that
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
    fn mode_and_full_bits<const POW: bool>(p: &Params, n: &NormSlices, xy: f32) -> (u32, u32) {
        // SAFETY: row 0 and col 0 exist in every norm slice.
        unsafe {
            let x = p.row_terms(n, 0);
            let m = match p.mode() {
                DOT => p.score::<POW, DOT>(n, &x, 0, xy, f32::NAN),
                COSINE => p.score::<POW, COSINE>(n, &x, 0, xy, f32::NAN),
                _ => p.score::<POW, FULL>(n, &x, 0, xy, f32::NAN),
            };
            (
                m.to_bits(),
                p.score::<POW, FULL>(n, &x, 0, xy, f32::NAN).to_bits(),
            )
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // Positive float ranges start at 0.125, a multiple of ulp(3.0): from 0.1,
        // proptest 1.12's sampler can overshoot `low` and trip its own (inverted)
        // debug_assert, ~1 in 4e7 draws in debug builds.
        // Bit-identical output across modes: the mode only skips +0.0 terms.
        #[test]
        fn score_mode_matches_full_bitwise(
            w in proptest::array::uniform5(prop_oneof![Just(0.0f32), Just(-0.0f32), 0.125f32..2.0]),
            t in proptest::array::uniform2(0.0f32..1.0),
            pow in any::<bool>(),
            norms in proptest::array::uniform6(prop_oneof![Just(0.0f32), 0.125f32..3.0]),
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
            let n = NormSlices {
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // `_depop_as_cosine` (Python prep): depop-only weights moved to the cosine
        // slots score, in their mode, as FULL with the depop slots.
        #[test]
        fn depop_in_cosine_slot_matches_full_bitwise(
            w in 0.125f32..2.0,
            shrink in proptest::array::uniform2(prop_oneof![Just(0.0f32), 0.125f32..2.0]),
            pow in any::<bool>(),
            xd in prop_oneof![Just(0.0f32), 0.125f32..3.0],
            yd in prop_oneof![Just(0.0f32), 0.125f32..3.0],
            xy in prop_oneof![Just(0.0f32), -5.0f32..5.0],
        ) {
            let pow = pow && !cfg!(miri); // see score_mode_matches_full_bitwise
            let p = |l2: f32, l3: f32| Params {
                a1: if pow { 0.5 } else { 1.0 },
                l2, l3,
                stabilized_shrink: shrink[0], bayesian_shrink: shrink[1],
                ..Params::default()
            };
            let (e, x, y) = ([0.0f32], [xd], [yd]);
            let depop = NormSlices {
                x_tversky: &e, y_tversky: &e, x_cosine: &e, y_cosine: &e, x_depop: &x, y_depop: &y,
            };
            let cosine = NormSlices {
                x_tversky: &e, y_tversky: &e, x_cosine: &x, y_cosine: &y, x_depop: &e, y_depop: &e,
            };
            let (full, swapped) = if pow {
                (mode_and_full_bits::<true>(&p(0.0, w), &depop, xy).1,
                 mode_and_full_bits::<true>(&p(w, 0.0), &cosine, xy).0)
            } else {
                (mode_and_full_bits::<false>(&p(0.0, w), &depop, xy).1,
                 mode_and_full_bits::<false>(&p(w, 0.0), &cosine, xy).0)
            };
            prop_assert_eq!(swapped, full, "mode {} xy {}", p(w, 0.0).mode(), xy);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 4096 }))]
        // K1: `score` with `k1_bound` either returns the exact
        // path's bits or NaN for a score the exact path puts below `floor`,
        // in every mode, so the drain pushes the same columns with the same
        // values. Three quarters of the cases put xy a few ulps from
        // `floor * den`, within ~2^-16 of the reject edge (`floor * (1 - 2^-10)
        // * den`) or at the bayesian edge (den as COSINE computes it), signed
        // xy, floors from subnormal to +inf.
        #[test]
        fn k1_bound_keeps_the_exact_pushes(
            w in proptest::array::uniform5(prop_oneof![Just(0.0f32), 0.125f32..2.0]),
            t in proptest::array::uniform2(0.0f32..1.0),
            pow in any::<bool>(),
            norms in proptest::array::uniform6(prop_oneof![Just(0.0f32), 0.125f32..3.0]),
            floor in prop_oneof![
                Just(0.0f32), Just(-1.0f32), Just(f32::INFINITY), Just(f32::MIN_POSITIVE),
                Just(1e-40f32), 1e-30f32..1e-25, 0.001f32..10.0
            ],
            edge in 0u8..4,
            rel in -1.0f32..1.0,
            ulps in -4i32..=4,
            xy_free in prop_oneof![Just(0.0f32), -5.0f32..5.0, 1e-30f32..1e-25],
            neg_shrink in any::<bool>(),
        ) {
            let pow = pow && !cfg!(miri); // see score_mode_matches_full_bitwise
            let p = Params {
                a1: if pow { 0.5 } else { 1.0 },
                l1: w[0], l2: w[1], l3: w[2], t1: t[0], t2: t[1],
                stabilized_shrink: w[3], bayesian_shrink: if neg_shrink { -w[4] } else { w[4] },
                threshold: 0.0,
            };
            let xy = if edge > 0 {
                let den = p.l2 * (norms[2] * norms[3]) + p.stabilized_shrink;
                // 1: a few ulps around the floor itself; 2: around the reject
                // edge; 3: around the bayesian edge x^2 = floor den (x + s)
                let m = if edge == 1 { 1.0 } else { (1.0 - 1.0 / 1024.0) * (1.0 + rel / 65536.0) };
                let q = if edge == 3 {
                    let (fd, s) = (f64::from(floor) * f64::from(den), f64::from(p.bayesian_shrink));
                    ((fd + (fd * fd + 4.0 * fd * s).sqrt()) / 2.0) as f32
                } else {
                    floor * den * m
                };
                let q = if pow { q * q } else { q }; // xy_pow = sqrt(xy)
                f32::from_bits((q.to_bits() as i32).wrapping_add(ulps) as u32)
            } else {
                xy_free
            };
            let xy = if xy.is_finite() { xy } else { xy_free }; // floor = inf
            let v: Vec<[f32; 1]> = norms.iter().map(|&x| [x]).collect();
            let n = NormSlices {
                x_tversky: &v[0], y_tversky: &v[1], x_cosine: &v[2],
                y_cosine: &v[3], x_depop: &v[4], y_depop: &v[5],
            };
            let bound = k1_bound(floor, p.bayesian_shrink >= 0.0);
            for mode in [p.mode(), FULL] {
                // SAFETY: row 0 and col 0 exist in every norm slice.
                let (fast, exact) = unsafe {
                    let x = p.row_terms(&n, 0);
                    macro_rules! both {
                        ($pow:literal, $m:ident) => {
                            (p.score::<$pow, $m>(&n, &x, 0, xy, bound), p.score::<$pow, $m>(&n, &x, 0, xy, f32::NAN))
                        };
                    }
                    match (pow, mode) {
                        (true, DOT) => both!(true, DOT),
                        (true, COSINE) => both!(true, COSINE),
                        (true, _) => both!(true, FULL),
                        (false, DOT) => both!(false, DOT),
                        (false, COSINE) => both!(false, COSINE),
                        (false, _) => both!(false, FULL),
                    }
                };
                if fast.is_nan() && !exact.is_nan() {
                    prop_assert!(floor > 0.0 && exact < floor, "mode {} xy {} floor {} exact {}", mode, xy, floor, exact);
                } else {
                    prop_assert_eq!(fast.to_bits(), exact.to_bits(), "mode {} xy {} floor {}", mode, xy, floor);
                }
            }
        }
    }

    #[test]
    fn k1_bound_is_off_without_a_positive_normal_floor() {
        for f in [0.0, -0.0, -1.0, 1e-40, f32::INFINITY, f32::NAN] {
            assert!(k1_bound(f, true).is_nan(), "{f}");
        }
        assert_eq!(k1_bound(1.0, true), 1.0 - 1.0 / 1024.0);
        assert!(k1_bound(1.0, false).is_nan());
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
