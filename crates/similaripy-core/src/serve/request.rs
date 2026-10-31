//! One history (`recommend`) and a batch of them (`recommend_batch`: rayon
//! over users calling the same function), in the caller's ids.

use std::cell::Cell;

use rayon::prelude::*;

use super::model::Model;
use super::{bit, invalid, words};
use crate::Error;
use crate::kernel::{accumulate, lower_bound};
use crate::topk::TopK;

/// As the similarity functions' default threshold 0: negative scores never
/// enter the top-K, 0.0 ones are dropped by `TopK`.
const THRESHOLD: f32 = 0.0;

/// One history. Ids are the caller's (the model's external ids, or S's column
/// indices without them); ids the model doesn't have are ignored everywhere.
pub struct Request<'a> {
    pub ids: &'a [i64],
    /// One finite weight per id; None = all 1.0.
    pub weights: Option<&'a [f32]>,
    pub k: usize,
    /// Never recommend a history item.
    pub exclude_seen: bool,
    /// Only these items can be recommended (ANDed with the view's mask);
    /// None = all.
    pub include: Option<&'a [i64]>,
    /// Never recommend these items.
    pub exclude: Option<&'a [i64]>,
    /// Best first; false = the same items in heap order (no sort).
    pub sorted: bool,
}

/// Per-thread state reused across requests; between requests `sums` is all
/// 0.0, `include` all zero and `touched` empty. Grows to the largest model it
/// served.
#[derive(Default)]
struct Scratch {
    sums: Vec<f32>,
    touched: Vec<u32>,
    /// Blocked path: `(pos, end)` in S per history item.
    cursor: Vec<(usize, usize)>,
    topk: TopK,
    hist: Vec<(u32, f32)>,
    /// Internal ids never recommended in this request (seen, excluded).
    blocked: Vec<u32>,
    /// The request's include bits, and the ids set in them (to clear).
    include: Vec<u64>,
    included: Vec<u32>,
    cols: Vec<i32>,
    vals: Vec<f32>,
}

thread_local! {
    /// Taken out for a request, put back only on a normal return: a panic
    /// mid-request drops the dirty scratch instead of reusing it (G4).
    static SCRATCH: Cell<Option<Scratch>> = const { Cell::new(None) };
}

/// The top `req.k` items of one history (score desc, then the more popular
/// item, unless unsorted), appended as (internal id, score).
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
    let mut s = SCRATCH.take().unwrap_or_default();
    s.run(model, req, out);
    SCRATCH.set(Some(s));
    Ok(())
}

/// The internal → caller's id map of a model's results.
fn caller_id(model: &Model) -> impl Fn(u32) -> i64 + '_ {
    let ext = model.ext.as_ref().map(|e| &e.out[..]);
    let new_to_old: &[i32] = &model.new_to_old;
    move |c| ext.map_or(i64::from(new_to_old[c as usize]), |e| e[c as usize])
}

/// The top `req.k` items of one history, best first (score desc, then the
/// more popular item; unless `!req.sorted`), appended to the outputs as the
/// caller's ids. Fewer than k when fewer items score > 0.
pub fn recommend(
    model: &Model,
    req: &Request,
    ids_out: &mut Vec<i64>,
    scores_out: &mut Vec<f32>,
) -> Result<(), Error> {
    let mut out = Vec::new();
    recommend_internal(model, req, &mut out)?;
    let id = caller_id(model);
    for (c, v) in out {
        ids_out.push(id(c));
        scores_out.push(v);
    }
    Ok(())
}

/// One list per user, flat: user u's ids are `ids[offsets[u]..offsets[u + 1]]`.
#[derive(Clone, Copy)]
pub struct Lists<'a> {
    pub offsets: &'a [i64],
    pub ids: &'a [i64],
}

impl<'a> Lists<'a> {
    fn check(&self, name: &str, users: usize) -> Result<(), Error> {
        if self.offsets.len() != users + 1
            || self.offsets.first() != Some(&0)
            || self.offsets.windows(2).any(|w| w[0] > w[1])
            || self.offsets.last().map(|&e| e as usize) != Some(self.ids.len())
        {
            return invalid(format!("{name}: one list per user expected"));
        }
        Ok(())
    }

    fn get(&self, u: usize) -> &'a [i64] {
        &self.ids[self.offsets[u] as usize..self.offsets[u + 1] as usize]
    }
}

/// Many histories, each answered as a `Request` with the same options:
/// `weights` aligned with `histories.ids`, `include` / `exclude` one list per
/// user (`no_include[u]`: user u has no include rule).
pub struct BatchRequest<'a> {
    pub histories: Lists<'a>,
    pub weights: Option<&'a [f32]>,
    pub k: usize,
    pub exclude_seen: bool,
    pub include: Option<Lists<'a>>,
    pub no_include: Option<&'a [bool]>,
    pub exclude: Option<Lists<'a>>,
    pub sorted: bool,
}

/// Batch results, one list per user: (offsets, the caller's ids, scores).
pub type Batch = (Vec<i64>, Vec<i64>, Vec<f32>);

/// Each history through the single-request path on the current rayon pool,
/// so user u's results equal `recommend` of that history bit for bit.
pub fn recommend_batch(model: &Model, b: &BatchRequest) -> Result<Batch, Error> {
    let users = b.histories.offsets.len().saturating_sub(1);
    b.histories.check("histories", users)?;
    if b.weights.is_some_and(|w| w.len() != b.histories.ids.len()) {
        return invalid("weights: one weight per history id expected");
    }
    if let Some(l) = b.include {
        l.check("include", users)?;
    }
    if b.no_include.is_some_and(|n| n.len() != users) {
        return invalid("include: one entry per user expected");
    }
    if let Some(l) = b.exclude {
        l.check("exclude", users)?;
    }
    let rows: Vec<Vec<(u32, f32)>> = (0..users)
        .into_par_iter()
        // Small tasks: request cost varies 100x across users.
        .with_max_len(4)
        .map(|u| {
            let (a, z) = (
                b.histories.offsets[u] as usize,
                b.histories.offsets[u + 1] as usize,
            );
            let req = Request {
                ids: &b.histories.ids[a..z],
                weights: b.weights.map(|w| &w[a..z]),
                k: b.k,
                exclude_seen: b.exclude_seen,
                include: b
                    .include
                    .filter(|_| !b.no_include.is_some_and(|n| n[u]))
                    .map(|l| l.get(u)),
                exclude: b.exclude.map(|l| l.get(u)),
                sorted: b.sorted,
            };
            // Sized by the row's one exact-size `extend`, not k: every row
            // stays alive until the batch ends (k ~ n_items x many users).
            let mut out = Vec::new();
            recommend_internal(model, &req, &mut out).map(|()| out)
        })
        .collect::<Result<_, _>>()?;
    let total: usize = rows.iter().map(Vec::len).sum();
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0i64);
    let (mut ids, mut scores) = (Vec::with_capacity(total), Vec::with_capacity(total));
    let id = caller_id(model);
    for row in rows {
        for (c, v) in row {
            ids.push(id(c));
            scores.push(v);
        }
        offsets.push(ids.len() as i64);
    }
    Ok((offsets, ids, scores))
}

impl Scratch {
    fn run(&mut self, m: &Model, req: &Request, out: &mut Vec<(u32, f32)>) {
        self.hist.clear();
        m.lookup_into(req.ids, req.weights, &mut self.hist);
        canonicalize(&mut self.hist);
        self.blocked.clear();
        if req.exclude_seen {
            self.blocked.extend(self.hist.iter().map(|&(id, _)| id));
        }
        self.blocked.extend(
            req.exclude
                .unwrap_or(&[])
                .iter()
                .filter_map(|&id| m.internal(id)),
        );
        // Taken out while `accumulate_all` borrows the scratch; all zero again
        // (only the set bits cleared) before it goes back.
        let mut include = std::mem::take(&mut self.include);
        if let Some(ids) = req.include {
            let n = words(m.n_items());
            if include.len() < n {
                include.resize(n, 0);
            }
            self.included.clear();
            for i in ids.iter().filter_map(|&id| m.internal(id)) {
                include[i as usize / 64] |= 1 << (i % 64);
                self.included.push(i);
            }
        }
        self.topk.set_k(req.k.min(m.n_active));
        let block = if m.block == 0 || m.block >= m.n_active {
            m.n_active
        } else {
            m.block
        };
        if self.sums.len() < block {
            self.sums.resize(block, 0.0);
        }
        let allowed = Masks(m.allowed.as_deref(), req.include.map(|_| &include[..]));
        // SAFETY: hist / blocked ids < n_items (lookup); sums.len() >= block;
        // the view's mask and `include` hold >= words(n_items) words.
        unsafe { self.accumulate_all(m, block, allowed) };
        if req.include.is_some() {
            for &i in &self.included {
                include[i as usize / 64] = 0;
            }
        }
        self.include = include;
        self.cols.clear();
        self.vals.clear();
        self.topk
            .drain_by_score(&mut self.cols, &mut self.vals, req.sorted);
        out.extend(
            self.cols
                .iter()
                .zip(&self.vals)
                .map(|(&c, &v)| (c as u32, v)),
        );
    }

    /// # Safety
    /// See the call in `run`.
    unsafe fn accumulate_all(&mut self, m: &Model, block: usize, allowed: Masks) {
        let s = m.s();
        let n_active = m.n_active;
        if block == n_active {
            self.preset(0, n_active, f32::NEG_INFINITY);
            for &(id, w) in &self.hist {
                // SAFETY: id < n_items; S columns < n_active == block <= sums.len().
                unsafe {
                    let (c, vals) = s.row_unchecked(id as usize);
                    accumulate(&mut self.sums, &mut self.touched, c, vals, w, 0);
                }
            }
            // SAFETY: forwarded.
            unsafe { self.drain(0, allowed) };
            self.preset(0, n_active, 0.0);
            return;
        }
        // Per-row column blocks, as the kernel's `blocked_row`.
        self.cursor.clear();
        self.cursor.extend(self.hist.iter().map(|&(id, _)| {
            // SAFETY: id < n_items.
            unsafe { s.row_span_unchecked(id as usize) }
        }));
        let mut start = 0;
        while start < n_active {
            let hi = (start + block).min(n_active);
            self.preset(start, hi, f32::NEG_INFINITY);
            for (cur, &(_, w)) in self.cursor.iter_mut().zip(&self.hist) {
                let (pos, end) = *cur;
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
                cur.0 = pos + n;
            }
            if !self.touched.is_empty() {
                // SAFETY: forwarded.
                unsafe { self.drain(start, allowed) };
            }
            // Even when the drain was skipped: the slots were set.
            self.preset(start, hi, 0.0);
            start = hi;
        }
    }

    /// Sets the blocked items' slots inside `[lo, hi)` to `val`: `-inf`
    /// before accumulating (`accumulate` never records a `-inf` slot as
    /// touched, so it is never drained), 0.0 after the drain. Ids
    /// `>= n_active` (no column) have no slot (G2).
    #[inline]
    fn preset(&mut self, lo: usize, hi: usize, val: f32) {
        for &id in &self.blocked {
            let id = id as usize;
            if (lo..hi).contains(&id) {
                self.sums[id - lo] = val;
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

/// The view's mask and the request's include bits (each None = all allowed).
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
