//! One history (`recommend`) and a batch of them (`recommend_batch`: rayon
//! over users calling the same function), in the caller's ids.

use std::cell::Cell;

use rayon::prelude::*;

use super::model::{BlockSize, Entry, Model};
use super::{bit, invalid, words};
use crate::Error;
use crate::kernel::prefetch;
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
    /// Best first; false = the same items in any order (no sort).
    pub sorted: bool,
}

/// Per-thread state reused across requests; between requests `sums` is all
/// 0.0, `include` all zero and `touched` empty. Grows to the largest model it
/// served.
#[derive(Default)]
struct Scratch {
    sums: Vec<f32>,
    touched: Vec<u32>,
    /// `(pos, end, column at pos)` in S per history item.
    cursor: Vec<(usize, usize, usize)>,
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

/// A worker's counters, updated after each request (never inside its loops).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Calls, failed ones included.
    pub requests: u64,
    /// Calls rejected as invalid (weights).
    pub errors: u64,
    /// History ids as sent.
    pub ids: u64,
    /// History ids the model has, among those looked up (`ids - known_ids`
    /// = unknown, ignored; except a `score` whose items have no column: no
    /// lookup, none counted).
    pub known_ids: u64,
    // ponytail: no count of S entries (the work's size): summing the history
    // rows' lengths after the request re-read indptr, cold by then on Y500M
    // full histories: batch / concurrent serving +3% (s1gate, 2026-11-13).
    // Count it at the server's edge if a metric needs it.
}

/// One serving thread's state over immutable, shared `Model`s (what to
/// compute): the scratch, the resolved block size, counters and reused
/// buffers (how this thread computes). Owned by one thread, never shared;
/// once warm a request allocates nothing but what the caller's outputs grow.
/// Any model can be served (buffers grow to the largest one); the block is
/// resolved once, for the model given to `new` (a new model = new workers, W3).
pub struct Worker {
    /// Taken out for a request, put back only on a normal return: a panic
    /// mid-request drops the dirty scratch instead of reusing it (G4).
    scratch: Option<Scratch>,
    /// Column block size, 0 = one block (clamped to the model's columns).
    block: usize,
    stats: Stats,
    /// `recommend`'s results in internal ids, before the id map.
    out: Vec<(u32, f32)>,
    /// `score`'s candidates: (internal id, position in `items`).
    cand: Vec<(u32, u32)>,
}

impl Worker {
    /// A worker for `model` served by `threads` threads at once (each with
    /// its own worker): the block size resolved here, once (`BlockSize::resolve`).
    /// No buffers until the first request.
    pub fn new(model: &Model, block_size: BlockSize, threads: usize) -> Result<Self, Error> {
        Ok(Self::with_block(
            block_size.resolve(model.n_active, threads)?,
        ))
    }

    const fn with_block(block: usize) -> Self {
        Self {
            scratch: None,
            block,
            stats: Stats {
                requests: 0,
                errors: 0,
                ids: 0,
                known_ids: 0,
            },
            out: Vec::new(),
            cand: Vec::new(),
        }
    }

    /// The column block size of its requests, 0 = one block.
    pub fn block(&self) -> usize {
        self.block
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The top `req.k` items of one history, best first (score desc, then
    /// the lower internal id: the `ItemOrder`'s tie rule; unless
    /// `!req.sorted`), appended to the outputs as the caller's ids. Fewer
    /// than k when fewer items score > 0.
    pub fn recommend(
        &mut self,
        model: &Model,
        req: &Request,
        ids_out: &mut Vec<i64>,
        scores_out: &mut Vec<f32>,
    ) -> Result<(), Error> {
        let mut out = std::mem::take(&mut self.out);
        out.clear();
        // Sized once: grown push by push, the Vecs reallocated ~6 times each
        // at k = 100 (allocation ~10% of a cap-10 request, 4d/profile.md).
        out.reserve(req.k.min(model.n_active));
        let r = self.recommend_internal(model, req, &mut out);
        if r.is_ok() {
            let id = caller_id(model);
            ids_out.reserve(out.len());
            scores_out.reserve(out.len());
            for &(c, v) in &out {
                ids_out.push(id(c));
                scores_out.push(v);
            }
        }
        self.out = out;
        r
    }

    /// The top `req.k` items of one history (score desc, then the lower
    /// internal id, unless unsorted), appended as (internal id, score).
    fn recommend_internal(
        &mut self,
        model: &Model,
        req: &Request,
        out: &mut Vec<(u32, f32)>,
    ) -> Result<(), Error> {
        self.stats.requests += 1;
        self.stats.ids += req.ids.len() as u64;
        if let Some(w) = req.weights {
            if w.len() != req.ids.len() {
                self.stats.errors += 1;
                return invalid("len(weights) != len(ids)");
            }
            if !w.iter().all(|x| x.is_finite()) {
                self.stats.errors += 1;
                return invalid("weights must be finite");
            }
        }
        let mut s = self.scratch.take().unwrap_or_default();
        let known = s.run(model, self.block, req, out);
        self.stats.known_ids += known as u64;
        self.scratch = Some(s);
        Ok(())
    }

    /// The scores of `items` for one history into `out` (replaced), aligned
    /// with `items` (duplicates repeated): `sum_i w_i * S[i, j]`, the same
    /// bits as `recommend`'s score of j (same accumulation; the block size
    /// changes no column's addition order). No top-K, no threshold. 0.0 for
    /// items the model doesn't have, items with no column (e.g. outside a
    /// trimmed build's candidates) and items the view's mask excludes; the
    /// request has no include / exclude (availability is retrieval's).
    pub fn score(
        &mut self,
        model: &Model,
        ids: &[i64],
        weights: Option<&[f32]>,
        items: &[i64],
        out: &mut Vec<f32>,
    ) -> Result<(), Error> {
        self.stats.requests += 1;
        self.stats.ids += ids.len() as u64;
        if let Err(e) = check_weights(ids, weights) {
            self.stats.errors += 1;
            return Err(e);
        }
        out.clear();
        out.resize(items.len(), 0.0);
        let mut s = self.scratch.take().unwrap_or_default();
        let known = s.score(model, self.block, ids, weights, items, &mut self.cand, out);
        self.stats.known_ids += known as u64;
        self.scratch = Some(s);
        Ok(())
    }
}

thread_local! {
    /// The calling thread's implicit worker (the free functions: Python,
    /// batches); its block is set per call by the caller's resolved one.
    static WORKER: Cell<Option<Worker>> = const { Cell::new(None) };
}

/// `f` on the calling thread's worker, with `block`. Taken out and put back
/// only on a normal return (G4, as the worker's own scratch).
fn with_worker<R>(block: usize, f: impl FnOnce(&mut Worker) -> R) -> R {
    let mut w = WORKER.take().unwrap_or_else(|| Worker::with_block(block));
    w.block = block;
    let r = f(&mut w);
    WORKER.set(Some(w));
    r
}

// ponytail: `recommend_internal` keeps its own copy: calling this moved its
// register allocation (asm check, score session).
fn check_weights(ids: &[i64], weights: Option<&[f32]>) -> Result<(), Error> {
    if let Some(w) = weights {
        if w.len() != ids.len() {
            return invalid("len(weights) != len(ids)");
        }
        if !w.iter().all(|x| x.is_finite()) {
            return invalid("weights must be finite");
        }
    }
    Ok(())
}

/// `Worker::score` on the calling thread's implicit worker, with a resolved
/// `block` (`BlockSize::resolve`).
pub fn score(
    model: &Model,
    block: usize,
    ids: &[i64],
    weights: Option<&[f32]>,
    items: &[i64],
) -> Result<Vec<f32>, Error> {
    let mut out = Vec::new();
    with_worker(block, |w| w.score(model, ids, weights, items, &mut out))?;
    Ok(out)
}

/// `score` per user on the current rayon pool (user u's scores equal
/// `score` of that history and items bit for bit), flat: aligned with
/// `items.ids` (user u's at `items.offsets[u]..items.offsets[u + 1]`). The
/// block is resolved once, for the pool's threads.
pub fn score_batch(
    model: &Model,
    block_size: BlockSize,
    histories: Lists,
    weights: Option<&[f32]>,
    items: Lists,
) -> Result<Vec<f32>, Error> {
    let users = histories.offsets.len().saturating_sub(1);
    histories.check("histories", users)?;
    items.check("items", users)?;
    if weights.is_some_and(|w| w.len() != histories.ids.len()) {
        return invalid("weights: one weight per history id expected");
    }
    let block = block_size.resolve(model.n_active, rayon::current_num_threads())?;
    let rows: Vec<Vec<f32>> = (0..users)
        .into_par_iter()
        .with_max_len(4)
        .map(|u| {
            let (a, z) = (
                histories.offsets[u] as usize,
                histories.offsets[u + 1] as usize,
            );
            score(
                model,
                block,
                &histories.ids[a..z],
                weights.map(|w| &w[a..z]),
                items.get(u),
            )
        })
        .collect::<Result<_, _>>()?;
    Ok(rows.concat())
}

/// The internal → caller's id map of a model's results.
fn caller_id(model: &Model) -> impl Fn(u32) -> i64 + '_ {
    let ext = model.ext.as_ref().map(|e| &e.out[..]);
    let new_to_old: &[i32] = &model.new_to_old;
    move |c| ext.map_or(i64::from(new_to_old[c as usize]), |e| e[c as usize])
}

/// `Worker::recommend` on the calling thread's implicit worker, with a
/// resolved `block` (`BlockSize::resolve`).
pub fn recommend(
    model: &Model,
    block: usize,
    req: &Request,
    ids_out: &mut Vec<i64>,
    scores_out: &mut Vec<f32>,
) -> Result<(), Error> {
    with_worker(block, |w| w.recommend(model, req, ids_out, scores_out))
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
/// the block resolved once for the pool's threads, so user u's results
/// equal `recommend` of that history bit for bit when sorted (unsorted: the
/// same items and scores, maybe in another order when the blocks differ).
pub fn recommend_batch(
    model: &Model,
    block_size: BlockSize,
    b: &BatchRequest,
) -> Result<Batch, Error> {
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
    let block = block_size.resolve(model.n_active, rayon::current_num_threads())?;
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
            with_worker(block, |w| w.recommend_internal(model, &req, &mut out)).map(|()| out)
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
    /// Returns how many history ids the model has.
    fn run(&mut self, m: &Model, block: usize, req: &Request, out: &mut Vec<(u32, f32)>) -> usize {
        self.hist.clear();
        m.lookup_into::<false>(req.ids, req.weights, &mut self.hist);
        let known = self.hist.len();
        canonicalize::<false>(&mut self.hist);
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
        let block = self.block(m, block);
        let allowed = Masks(m.allowed.as_deref(), req.include.map(|_| &include[..]));
        // SAFETY: hist / blocked ids < n_items (lookup); sums.len() >= block;
        // the view's mask and `include` hold >= words(n_items) words.
        unsafe { self.accumulate_all(m, block, allowed) };
        #[cfg(test)]
        tests::panic_if_asked();
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
        known
    }

    /// The worker's `block` for this model (in `1..=n_active`), `sums`
    /// grown to it.
    #[inline(always)]
    fn block(&mut self, m: &Model, block: usize) -> usize {
        let block = if block == 0 || block >= m.n_active {
            m.n_active
        } else {
            block
        };
        if self.sums.len() < block {
            self.sums.resize(block, 0.0);
        }
        block
    }

    /// Scores of `items` into `out` (zeros, one per item): see
    /// `Worker::score`. Returns how many history ids the model has (0 when
    /// no item has a column: no lookup).
    #[allow(clippy::too_many_arguments)]
    fn score(
        &mut self,
        m: &Model,
        block: usize,
        ids: &[i64],
        weights: Option<&[f32]>,
        items: &[i64],
        cand: &mut Vec<(u32, u32)>,
        out: &mut [f32],
    ) -> usize {
        // (internal id, position in `items`) of the candidates with a slot the
        // view allows, by internal id: the gather walks it block by block.
        // The worker's, not a scratch field (recommend's Scratch layout, and
        // so its asm, stays as it is).
        cand.clear();
        cand.extend(items.iter().enumerate().filter_map(|(pos, &id)| {
            let i = m.internal(id)?;
            // SAFETY: i < n_items; the mask holds words(n_items) words.
            let allowed = m
                .allowed
                .as_deref()
                .is_none_or(|a| unsafe { bit(a, i as usize) });
            ((i as usize) < m.n_active && allowed).then_some((i, pos as u32))
        }));
        cand.sort_unstable();
        self.hist.clear();
        let Some(&(last, _)) = cand.last() else {
            return 0;
        };
        m.lookup_into::<true>(ids, weights, &mut self.hist);
        let known = self.hist.len();
        canonicalize::<true>(&mut self.hist);
        let block = self.block(m, block);
        let pairs: &[Entry] = &m.pairs;
        // Blocks past the last candidate don't matter (left unaccumulated).
        let end = last as usize + 1;
        // SAFETY: hist ids < n_items (lookup).
        unsafe { self.set_cursors(m) };
        let (mut start, mut next) = (0, 0);
        while start < end {
            let hi = (start + block).min(m.n_active);
            // SAFETY: as `accumulate_all` (sums.len() >= block >= hi - start).
            unsafe { self.accumulate_block(pairs, start, hi, m.n_active) };
            while let Some(&(c, pos)) = cand.get(next)
                && (c as usize) < hi
            {
                out[pos as usize] = self.sums[c as usize - start];
                next += 1;
            }
            for &local in &self.touched {
                // SAFETY: touched < block <= sums.len().
                unsafe { *self.sums.get_unchecked_mut(local as usize) = 0.0 };
            }
            self.touched.clear();
            start = hi;
        }
        known
    }

    /// One cursor per history row: `(pos, end, column at pos)`.
    ///
    /// # Safety
    /// Every `hist` id < n_items.
    #[inline(always)]
    unsafe fn set_cursors(&mut self, m: &Model) {
        let pairs: &[Entry] = &m.pairs;
        self.cursor.clear();
        self.cursor.extend(self.hist.iter().map(|&(id, _)| {
            // SAFETY: id < n_items; the span is the row's.
            unsafe {
                let (pos, end) = m.row_span_unchecked(id as usize);
                (pos, end, next_col(pairs.get_unchecked(pos..end), 0))
            }
        }));
    }

    /// # Safety
    /// See the call in `run`.
    unsafe fn accumulate_all(&mut self, m: &Model, block: usize, allowed: Masks) {
        let pairs: &[Entry] = &m.pairs;
        let n_active = m.n_active;
        // Per-row column blocks, as the kernel's `blocked_row`; one block (every
        // catalog <= the block size) is this loop once.
        // SAFETY: forwarded.
        unsafe { self.set_cursors(m) };
        let mut start = 0;
        while start < n_active {
            let hi = (start + block).min(n_active);
            self.preset(start, hi, f32::NEG_INFINITY);
            // SAFETY: forwarded.
            unsafe { self.accumulate_block(pairs, start, hi, n_active) };
            if !self.touched.is_empty() {
                // SAFETY: forwarded.
                unsafe { self.drain(start, allowed) };
            }
            // Even when the drain was skipped: the slots were set.
            self.preset(start, hi, 0.0);
            start = hi;
        }
    }

    /// Every history row's segment in the column block `[start, hi)` into
    /// `sums` (shared by `recommend` and `score`: one copy of the tuned loop).
    ///
    /// # Safety
    /// The cursors are `set_cursors`' moved block by block; `hi - start <= sums.len()`.
    #[inline(always)]
    unsafe fn accumulate_block(
        &mut self,
        pairs: &[Entry],
        start: usize,
        hi: usize,
        n_active: usize,
    ) {
        {
            for i in 0..self.cursor.len() {
                // Prefetch the in-block part of a row PREFETCH_ROWS ahead: each history row of S starts at an
                // unpredictable address the hardware prefetcher can't reach, and a k = 100 row is ~13 lines.
                // Overlapping those misses across rows (M2 Pro vs no prefetch, bits unchanged): Y500M latency
                // −13..−37%, 6 concurrent callers −43..−46%, batch −30..−48%, ML-32M −8..−42%; 2..16 rows ahead
                // all within a few %.
                if let Some(&(ppos, pend, pnext)) = self.cursor.get(i + PREFETCH_ROWS)
                    && pnext < hi
                {
                    let bytes =
                        (pend - ppos).min(PREFETCH_LINES * 8) * std::mem::size_of::<Entry>();
                    // SAFETY: ppos < pend <= pairs.len(); the hints stay inside the row.
                    let base = unsafe { pairs.as_ptr().add(ppos) }.cast::<u8>();
                    for off in (0..bytes).step_by(64) {
                        prefetch(base.wrapping_add(off));
                    }
                }
                // SAFETY: i < cursor.len() == hist.len(); forwarded (`accumulate_all`'s contract).
                unsafe {
                    accumulate_row(
                        &mut self.sums,
                        &mut self.touched,
                        pairs,
                        self.cursor.get_unchecked_mut(i),
                        self.hist.get_unchecked(i).1,
                        start,
                        hi,
                        n_active,
                    )
                };
            }
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
    // Out of line, as the kernel's drain (benchmarks/README.md).
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

/// The kernel's `accumulate` over S's (column, value) entries: same body,
/// same addition order (same bits). `BOUNDED` stops at the first column
/// `>= hi` (a row past the block end, no search first); unbounded takes the
/// whole segment and keeps no count (two instructions less per entry: the
/// ML-32M warm batch, where S sits in cache, felt them). Returns how many.
///
/// # Safety
/// Every taken `col - offset` is `< sums.len()`.
#[inline(always)]
unsafe fn accumulate<const BOUNDED: bool>(
    sums: &mut [f32],
    touched: &mut Vec<u32>,
    p: &[Entry],
    v1: f32,
    offset: usize,
    hi: usize,
) -> usize {
    // At most one push per entry, so one reserve covers the segment.
    touched.reserve(p.len());
    let buf = touched.as_mut_ptr();
    let mut n = touched.len();
    let mut taken = p.len();
    for (i, &Entry { col, val }) in p.iter().enumerate() {
        if BOUNDED && col as usize >= hi {
            taken = i;
            break;
        }
        let local = col as usize - offset;
        // SAFETY: caller contract; n < capacity (reserve above).
        unsafe {
            let s = sums.get_unchecked_mut(local);
            let old = *s; // reuse: the touched store may alias
            // Branchless first touch, as the kernel's.
            buf.add(n).write(local as u32);
            n += (old == 0.0) as usize;
            *s = old + val * v1;
        }
    }
    // SAFETY: entries [len, n) were written above, n <= capacity.
    unsafe { touched.set_len(n) };
    taken
}

/// History rows ahead to prefetch, and the most lines prefetched per row (16 lines = 128 entries: a whole
/// k <= 128 row; longer rows stream on the hardware prefetcher after that).
const PREFETCH_ROWS: usize = 4;
const PREFETCH_LINES: usize = 16;

/// One history row's segment in the block `[start, hi)`, its cursor moved past it. Out of line so the accumulate
/// loops keep the registers they had before the prefetch (x86_64: no extra stack operand); one call per row.
///
/// # Safety
/// As `Scratch::accumulate_all`: the cursor is a suffix of its row's span, `hi - start <= sums.len()`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_row(
    sums: &mut [f32],
    touched: &mut Vec<u32>,
    pairs: &[Entry],
    cur: &mut (usize, usize, usize),
    w: f32,
    start: usize,
    hi: usize,
    n_active: usize,
) {
    let (pos, end, next) = *cur;
    // Cached, so a row with nothing in this block costs no load from S.
    if next >= hi {
        return;
    }
    // SAFETY: a suffix of a row span.
    let p = unsafe { pairs.get_unchecked(pos..end) };
    // Not empty (next < hi). The last block takes the rest of the row
    // (every column < n_active); a row past the block end stops inside the
    // accumulate loop, no search first.
    // SAFETY: start <= c < hi, so c - start < block <= sums.len().
    let n = unsafe {
        if hi == n_active || (p[p.len() - 1].col as usize) < hi {
            accumulate::<false>(sums, touched, p, w, start, hi)
        } else {
            accumulate::<true>(sums, touched, p, w, start, hi)
        }
    };
    *cur = (pos + n, end, next_col(p, n));
}

/// The column at `p[n]`, `usize::MAX` past the end (never in a block).
#[inline(always)]
fn next_col(p: &[Entry], n: usize) -> usize {
    p.get(n).map_or(usize::MAX, |e| e.col as usize)
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
/// `COPY`: one instantiation per caller, as `Core::lookup_into`'s.
fn canonicalize<const COPY: bool>(hist: &mut Vec<(u32, f32)>) {
    hist.sort_unstable_by_key(|&(i, _)| i);
    hist.dedup_by(|b, a| {
        let dup = a.0 == b.0;
        if dup {
            a.1 += b.1;
        }
        dup
    });
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::CsrView;

    thread_local! {
        static PANIC: Cell<bool> = const { Cell::new(false) };
    }

    /// Called by `Scratch::run` mid-request (include bits set, top-K full).
    pub(super) fn panic_if_asked() {
        if PANIC.get() {
            panic!("test: mid-request");
        }
    }

    // A panic mid-request drops the dirty scratch (G4): the next request on
    // the same worker answers as a fresh worker, its include bits not ORed
    // with the panicked request's.
    #[test]
    fn panic_mid_request_then_clean() {
        let (indptr, indices, data) = (vec![0i64, 2, 2, 2], vec![1, 2], vec![1.0f32, 2.0]);
        let s = CsrView::new("S", &indptr, &indices, &data, 3).unwrap();
        let m = Model::with_order(s, vec![0, 1, 2], None).unwrap();
        let req = |include| Request {
            ids: &[0],
            weights: None,
            k: 5,
            exclude_seen: true,
            include: Some(include),
            exclude: None,
            sorted: true,
        };
        let run = |w: &mut Worker, r: &Request| {
            let mut out = (vec![], vec![]);
            w.recommend(&m, r, &mut out.0, &mut out.1).map(|()| out)
        };
        let mut w = Worker::new(&m, BlockSize::Off, 1).unwrap();
        run(&mut w, &req(&[1])).unwrap(); // warm
        PANIC.set(true);
        let hit = catch_unwind(AssertUnwindSafe(|| run(&mut w, &req(&[2]))));
        PANIC.set(false);
        assert!(hit.is_err());
        let fresh = run(&mut Worker::new(&m, BlockSize::Off, 1).unwrap(), &req(&[1])).unwrap();
        assert_eq!(fresh, (vec![1], vec![1.0]));
        assert_eq!(run(&mut w, &req(&[1])).unwrap(), fresh);
        // The implicit worker of the free functions too.
        PANIC.set(true);
        let hit = catch_unwind(|| recommend(&m, 0, &req(&[2]), &mut vec![], &mut vec![]));
        PANIC.set(false);
        assert!(hit.is_err());
        let mut out = (vec![], vec![]);
        recommend(&m, 0, &req(&[1]), &mut out.0, &mut out.1).unwrap();
        assert_eq!(out, fresh);
    }
}
