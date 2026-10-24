//! One history (`recommend`) and a batch of them (`recommend_batch`: rayon
//! over users calling the same function).

use std::cell::Cell;

use rayon::prelude::*;

use super::ids::{Ids, ItemMask, Unknown};
use super::model::Model;
use super::{bit, invalid};
use crate::Error;
use crate::kernel::{accumulate, lower_bound};
use crate::topk::TopK;

/// As the similarity functions' default threshold 0: negative scores never
/// enter the top-K, 0.0 ones are dropped by `TopK`.
const THRESHOLD: f32 = 0.0;

/// One history.
pub struct Request<'a> {
    pub ids: Ids<'a>,
    /// One finite weight per id; None = all 1.0.
    pub weights: Option<&'a [f32]>,
    pub k: usize,
    /// Never recommend a history item.
    pub exclude_seen: bool,
    /// From `Model::mask` of this model or of one sharing its build
    /// (views, `trimmed`); None = all allowed.
    pub allowed: Option<&'a ItemMask>,
    pub unknown: Unknown,
}

/// Per-thread state reused across requests; between requests `sums` is all
/// 0.0 and `touched` empty. Grows to the largest model it served.
#[derive(Default)]
struct Scratch {
    sums: Vec<f32>,
    touched: Vec<u32>,
    /// Blocked path: `(pos, end)` in S per history item.
    cursor: Vec<(u32, u32)>,
    topk: TopK,
    hist: Vec<(u32, f32)>,
    cols: Vec<i32>,
    vals: Vec<f32>,
}

thread_local! {
    /// Taken out for a request, put back only on a normal return: a panic
    /// mid-request drops the dirty scratch instead of reusing it (G4).
    static SCRATCH: Cell<Option<Scratch>> = const { Cell::new(None) };
}

/// The top `req.k` items of one history, best first (score desc, then the
/// more popular item), appended as (internal id, score).
fn recommend_internal(
    model: &Model,
    req: &Request,
    out: &mut Vec<(u32, f32)>,
) -> Result<(), Error> {
    if let Some(w) = req.weights {
        if w.len() != req.ids.len() {
            return invalid("len(weights) != len(ids)");
        }
        if !w.iter().all(|x| x.is_finite()) {
            return invalid("weights must be finite");
        }
    }
    if let Some(m) = req.allowed {
        model.check_mask(m)?;
    }
    let mut s = SCRATCH.take().unwrap_or_default();
    let r = s.run(model, req, out);
    // An error comes from the lookup, before the scratch is touched: clean.
    SCRATCH.set(Some(s));
    r
}

/// The top `req.k` items of one history, best first (score desc, then the
/// more popular item), appended to the outputs; ids as the request's
/// (external ids, or S's column indices). Fewer than k when fewer items
/// score > 0.
pub fn recommend(
    model: &Model,
    req: &Request,
    ids_out: &mut Vec<i64>,
    scores_out: &mut Vec<f32>,
) -> Result<(), Error> {
    let mut out = Vec::new();
    recommend_internal(model, req, &mut out)?;
    let ext = match (req.ids, &model.ext) {
        (Ids::External(_), Some(e)) => Some(&e.out),
        _ => None,
    };
    let new_to_old: &[i32] = &model.new_to_old;
    for (c, v) in out {
        ids_out.push(ext.map_or(i64::from(new_to_old[c as usize]), |e| e[c as usize]));
        scores_out.push(v);
    }
    Ok(())
}

/// Batch results as CSR over S's original columns: (indptr, columns, scores).
pub type Batch = (Vec<i64>, Vec<i32>, Vec<f32>);

/// One history per row of H (CSR over S's original columns: `indptr` into
/// `cols` / `weights`), each through the single-request path on the current
/// rayon pool, so row u equals `recommend` of that history bit for bit.
/// Columns of dropped items are skipped; out of range is an error.
pub fn recommend_batch(
    model: &Model,
    indptr: &[i64],
    cols: &[i32],
    weights: Option<&[f32]>,
    k: usize,
    exclude_seen: bool,
    allowed: Option<&ItemMask>,
) -> Result<Batch, Error> {
    if indptr.first() != Some(&0)
        || indptr.windows(2).any(|w| w[0] > w[1])
        || indptr.last().map(|&e| e as usize) != Some(cols.len())
        || weights.is_some_and(|w| w.len() != cols.len())
    {
        return invalid("history: bad CSR structure");
    }
    if cols.iter().any(|&c| c < 0 || c as usize >= model.n_cols()) {
        return invalid("history: column index out of range");
    }
    let rows: Vec<Vec<(u32, f32)>> = (0..indptr.len() - 1)
        .into_par_iter()
        // Small tasks: request cost varies 100x across users.
        .with_max_len(4)
        .map(|u| {
            let (a, b) = (indptr[u] as usize, indptr[u + 1] as usize);
            let req = Request {
                ids: Ids::Columns(&cols[a..b]),
                weights: weights.map(|w| &w[a..b]),
                k,
                exclude_seen,
                allowed,
                unknown: Unknown::Skip,
            };
            // Sized by the row's one exact-size `extend`, not k: every row
            // stays alive until the batch ends (k ~ n_items x many users).
            let mut out = Vec::new();
            recommend_internal(model, &req, &mut out).map(|()| out)
        })
        .collect::<Result<_, _>>()?;
    let total: usize = rows.iter().map(Vec::len).sum();
    let mut out_indptr = Vec::with_capacity(rows.len() + 1);
    out_indptr.push(0i64);
    let (mut out_cols, mut out_scores) = (Vec::with_capacity(total), Vec::with_capacity(total));
    let new_to_old: &[i32] = &model.new_to_old;
    for row in rows {
        for (c, v) in row {
            out_cols.push(new_to_old[c as usize]);
            out_scores.push(v);
        }
        out_indptr.push(out_cols.len() as i64);
    }
    Ok((out_indptr, out_cols, out_scores))
}

impl Scratch {
    fn run(&mut self, m: &Model, req: &Request, out: &mut Vec<(u32, f32)>) -> Result<(), Error> {
        self.hist.clear();
        m.lookup_into(req.ids, req.weights, req.unknown, &mut self.hist)?;
        canonicalize(&mut self.hist);
        self.topk.set_k(req.k.min(m.n_active));
        let block = if m.block == 0 || m.block >= m.n_active {
            m.n_active
        } else {
            m.block
        };
        if self.sums.len() < block {
            self.sums.resize(block, 0.0);
        }
        let allowed = Masks(m.allowed.as_deref(), req.allowed.map(|a| a.bits.as_slice()));
        // SAFETY: hist ids < n_items (lookup); sums.len() >= block; masks of
        // this layout (checked by the caller / `view`), so sized for n_items.
        unsafe { self.accumulate_all(m, block, req.exclude_seen, allowed) };
        self.cols.clear();
        self.vals.clear();
        self.topk.drain_by_score(&mut self.cols, &mut self.vals);
        out.extend(
            self.cols
                .iter()
                .zip(&self.vals)
                .map(|(&c, &v)| (c as u32, v)),
        );
        Ok(())
    }

    /// # Safety
    /// See the call in `run`.
    unsafe fn accumulate_all(
        &mut self,
        m: &Model,
        block: usize,
        exclude_seen: bool,
        allowed: Masks,
    ) {
        let s = m.s();
        let n_active = m.n_active;
        if block == n_active {
            self.preset(0, n_active, exclude_seen, f32::NEG_INFINITY);
            for &(id, w) in &self.hist {
                // SAFETY: id < n_items; S columns < n_active == block <= sums.len().
                unsafe {
                    let (c, vals) = s.row_unchecked(id as usize);
                    accumulate(&mut self.sums, &mut self.touched, c, vals, w, 0);
                }
            }
            // SAFETY: forwarded.
            unsafe { self.drain(0, allowed) };
            self.preset(0, n_active, exclude_seen, 0.0);
            return;
        }
        // Per-row column blocks, as the kernel's `blocked_row`.
        self.cursor.clear();
        self.cursor.extend(self.hist.iter().map(|&(id, _)| {
            // SAFETY: id < n_items.
            let (a, b) = unsafe { s.row_span_unchecked(id as usize) };
            (a as u32, b as u32)
        }));
        let mut start = 0;
        while start < n_active {
            let hi = (start + block).min(n_active);
            self.preset(start, hi, exclude_seen, f32::NEG_INFINITY);
            for (cur, &(_, w)) in self.cursor.iter_mut().zip(&self.hist) {
                let (pos, end) = (cur.0 as usize, cur.1 as usize);
                // SAFETY: a suffix of a row span.
                let (c_cols, c_vals) = unsafe { s.span_unchecked(pos, end) };
                let (Some(&first), Some(&last)) = (c_cols.first(), c_cols.last()) else {
                    continue;
                };
                if first >= hi as i32 {
                    continue;
                }
                let n = if last < hi as i32 {
                    c_cols.len()
                } else {
                    lower_bound(c_cols, hi as i32)
                };
                // SAFETY: start <= c < hi, so c - start < block <= sums.len().
                unsafe {
                    accumulate(
                        &mut self.sums,
                        &mut self.touched,
                        &c_cols[..n],
                        &c_vals[..n],
                        w,
                        start,
                    )
                };
                cur.0 = (pos + n) as u32;
            }
            if !self.touched.is_empty() {
                // SAFETY: forwarded.
                unsafe { self.drain(start, allowed) };
            }
            // Even when the drain was skipped: the slots were set.
            self.preset(start, hi, exclude_seen, 0.0);
            start = hi;
        }
    }

    /// Sets the history's slots inside `[lo, hi)` to `val`: `-inf` before
    /// accumulating (`accumulate` never records a `-inf` slot as touched, so
    /// it is never drained), 0.0 after the drain. Ids `>= n_active` (no
    /// column) have no slot (G2).
    #[inline]
    fn preset(&mut self, lo: usize, hi: usize, on: bool, val: f32) {
        if on {
            for &(id, _) in &self.hist {
                let id = id as usize;
                if (lo..hi).contains(&id) {
                    self.sums[id - lo] = val;
                }
            }
        }
    }

    /// # Safety
    /// Touched + `offset` < n_items; `allowed` sized for n_items.
    // Out of line, as the kernel's drain (tests/benchmarks/README.md).
    #[inline(never)]
    unsafe fn drain(&mut self, offset: usize, allowed: Masks) {
        let (sums, touched, topk) = (&mut self.sums, &self.touched, &mut self.topk);
        // SAFETY: forwarded.
        unsafe {
            match allowed {
                Masks(None, None) => drain_impl::<0>(sums, touched, topk, offset, &[], &[]),
                Masks(Some(a), None) | Masks(None, Some(a)) => {
                    drain_impl::<1>(sums, touched, topk, offset, a, &[])
                }
                Masks(Some(a), Some(b)) => drain_impl::<2>(sums, touched, topk, offset, a, b),
            }
        }
        self.touched.clear();
    }
}

/// The default mask and the per-call mask (each None = all allowed).
#[derive(Clone, Copy)]
struct Masks<'a>(Option<&'a [u64]>, Option<&'a [u64]>);

/// DOT drain: score = sum; `MASKS` masks (0, 1 in `a`, 2 in `a` and `b`)
/// are tested only for scores that pass the floor (most never do: <= 1%
/// cost, D64).
///
/// # Safety
/// As `Scratch::drain`.
#[inline(always)]
unsafe fn drain_impl<const MASKS: u8>(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    offset: usize,
    a: &[u64],
    b: &[u64],
) {
    let mut floor = topk.floor(THRESHOLD);
    for &local in touched {
        // SAFETY: touched < sums.len(); col < n_items (caller).
        unsafe {
            let val = std::mem::take(sums.get_unchecked_mut(local as usize));
            let col = offset + local as usize;
            if val >= floor && (MASKS == 0 || (bit(a, col) && (MASKS == 1 || bit(b, col)))) {
                topk.push(col as i32, val);
                floor = topk.floor(THRESHOLD);
            }
        }
    }
}

/// Sorts by internal id and merges duplicates (weights summed), so the
/// addition order depends on the item set, not the history order (G3).
/// Exception: 3+ copies of one id with different weights sum in an order the
/// sort picks from the input (f32 addition isn't associative); the same input
/// still gives the same bits.
fn canonicalize(hist: &mut Vec<(u32, f32)>) {
    hist.sort_unstable_by_key(|&(i, _)| i);
    hist.dedup_by(|b, a| {
        let dup = a.0 == b.0;
        if dup {
            a.1 += b.1;
        }
        dup
    });
}
