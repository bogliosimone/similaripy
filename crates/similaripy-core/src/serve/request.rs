//! One history (`Worker::recommend`, `recommend`), a batch of them
//! (`recommend_batch`: rayon over users calling the same function) and the
//! scores of given items (`score`, `score_batch`), in the caller's ids.

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
// `drain_head` masks a candidate out with score -inf: every floor must stay above it.
const _: () = assert!(THRESHOLD > f32::NEG_INFINITY);
/// Candidates per request through the branchless drain head, in units of k (`drain_head`; tuned on the M2:
/// 8 x k, the x86 day confirms it).
const HEAD: usize = 8;

/// Extra views per request at most (`Request::views`, `ScoreRequest::views`,
/// `BatchRequest::views`, `score_batch`'s).
pub const MAX_VIEWS: usize = 8;

/// One history. Ids are the caller's (the model's external ids, or S's column
/// indices without them); ids the model doesn't have are ignored everywhere.
/// `Default`: the library's defaults (k 10, exclude_seen, sorted), no history,
/// no rules: set what differs, `..Default::default()` for the rest.
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
    /// More views of the served model (`Model::view` of the same base, at
    /// most `MAX_VIEWS`): an item must be allowed by every one, ANDed with the
    /// served model's mask and `include` (V16). Up to two masks in all cost
    /// what one view costs; three or more pay a test per mask for each
    /// candidate (+15..+50% per request vs the same AND as one view, Y500M):
    /// for a combination served often, build it once (`view` of a view).
    pub views: &'a [&'a Model],
}

impl Default for Request<'_> {
    fn default() -> Self {
        Self {
            ids: &[],
            weights: None,
            k: 10,
            exclude_seen: true,
            include: None,
            exclude: None,
            sorted: true,
            views: &[],
        }
    }
}

/// One history and the items to score (`Worker::score`): ids as `Request`'s,
/// `views` as `Request::views` (an item one of them excludes scores 0.0).
#[derive(Default)]
pub struct ScoreRequest<'a> {
    pub ids: &'a [i64],
    /// One finite weight per id; None = all 1.0.
    pub weights: Option<&'a [f32]>,
    pub items: &'a [i64],
    pub views: &'a [&'a Model],
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
    /// Calls rejected as invalid (weights, views).
    pub errors: u64,
    /// History ids as sent.
    pub ids: u64,
    /// History ids the model has, among those looked up (`ids - known_ids`
    /// = unknown, ignored).
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

    fn with_block(block: usize) -> Self {
        Self {
            scratch: None,
            block,
            stats: Stats::default(),
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
        if !req.views.is_empty()
            && let Err(e) = check_views(model, req.views)
        {
            self.stats.errors += 1;
            return Err(e);
        }
        let mut s = self.scratch.take().unwrap_or_default();
        let known = s.run(model, self.block, req, out);
        self.stats.known_ids += known as u64;
        self.scratch = Some(s);
        Ok(())
    }

    /// The scores of `req.items` for one history into `out` (replaced),
    /// aligned with the items (duplicates repeated): `sum_i w_i * S[i, j]`,
    /// the same bits as `recommend`'s score of j (same accumulation; the block
    /// size changes no column's addition order). No top-K, no threshold. 0.0
    /// for items the model doesn't have, items with no column (e.g. outside a
    /// trimmed build's candidates) and items the view's mask or `req.views`
    /// exclude; the request has no include / exclude (availability is
    /// retrieval's).
    pub fn score(
        &mut self,
        model: &Model,
        req: &ScoreRequest,
        out: &mut Vec<f32>,
    ) -> Result<(), Error> {
        self.stats.requests += 1;
        self.stats.ids += req.ids.len() as u64;
        if let Err(e) = check_weights(req.ids, req.weights).and_then(|()| match req.views {
            [] => Ok(()),
            v => check_views(model, v),
        }) {
            self.stats.errors += 1;
            return Err(e);
        }
        out.clear();
        out.resize(req.items.len(), 0.0);
        let mut s = self.scratch.take().unwrap_or_default();
        let known = s.score(model, self.block, req, &mut self.cand, out);
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

/// `views`: at most `MAX_VIEWS`, each a view of `model`'s base. Out of line
/// (a request calls it only with views; a batch once up front, for a clear
/// error before any user runs, then each user's request again): the request
/// path keeps its registers.
#[cold]
#[inline(never)]
fn check_views(model: &Model, views: &[&Model]) -> Result<(), Error> {
    if views.len() > MAX_VIEWS {
        return invalid(format!("views: at most {MAX_VIEWS} per request"));
    }
    if !views.iter().all(|v| v.same_core(model)) {
        return invalid("views: each must be a view of the served model (the same base)");
    }
    Ok(())
}

/// `Worker::score` on the calling thread's implicit worker, with a resolved
/// `block` (`BlockSize::resolve`).
pub fn score(model: &Model, block: usize, req: &ScoreRequest) -> Result<Vec<f32>, Error> {
    let mut out = Vec::new();
    with_worker(block, |w| w.score(model, req, &mut out))?;
    Ok(out)
}

/// `score` per user on the current rayon pool (user u's scores equal
/// `score` of that history and items bit for bit), flat: aligned with
/// `items.ids` (user u's at `items.offsets[u]..items.offsets[u + 1]`);
/// `views` (`ScoreRequest::views`) for every user. The block is resolved
/// once, for the pool's threads.
pub fn score_batch(
    model: &Model,
    block_size: BlockSize,
    histories: Lists,
    weights: Option<&[f32]>,
    items: Lists,
    views: &[&Model],
) -> Result<Vec<f32>, Error> {
    let users = users(histories, weights)?;
    items.check("items", users)?;
    check_views(model, views)?;
    let block = block_size.resolve(model.n_active, rayon::current_num_threads())?;
    let rows: Vec<Vec<f32>> = (0..users)
        .into_par_iter()
        .with_max_len(4)
        .map(|u| {
            let r = histories.span(u);
            let req = ScoreRequest {
                ids: &histories.ids[r.clone()],
                weights: weights.map(|w| &w[r]),
                items: items.get(u),
                views,
            };
            score(model, block, &req)
        })
        .collect::<Result<_, _>>()?;
    Ok(rows.concat())
}

/// The internal → caller's id map of a model's results.
fn caller_id(model: &Model) -> impl Fn(u32) -> i64 + '_ {
    let ext = model.ext.as_ref().map(|e| &e.by_internal[..]);
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

    /// User u's range in `ids` (and in weights aligned with them).
    fn span(&self, u: usize) -> std::ops::Range<usize> {
        self.offsets[u] as usize..self.offsets[u + 1] as usize
    }

    fn get(&self, u: usize) -> &'a [i64] {
        &self.ids[self.span(u)]
    }
}

/// A batch's user count, its histories and their weights checked.
fn users(histories: Lists, weights: Option<&[f32]>) -> Result<usize, Error> {
    let users = histories.offsets.len().saturating_sub(1);
    histories.check("histories", users)?;
    if weights.is_some_and(|w| w.len() != histories.ids.len()) {
        return invalid("weights: one weight per history id expected");
    }
    Ok(users)
}

/// Many histories, each answered as a `Request` with the same options:
/// `weights` aligned with `histories.ids`, `include` / `exclude` one list per
/// user (`no_include[u]`: user u has no include rule), `views` for every user.
/// `Default`: no users, `Request::default()`'s options.
pub struct BatchRequest<'a> {
    pub histories: Lists<'a>,
    pub weights: Option<&'a [f32]>,
    pub k: usize,
    pub exclude_seen: bool,
    pub include: Option<Lists<'a>>,
    pub no_include: Option<&'a [bool]>,
    pub exclude: Option<Lists<'a>>,
    pub sorted: bool,
    pub views: &'a [&'a Model],
}

impl Default for BatchRequest<'_> {
    fn default() -> Self {
        let r = Request::default();
        Self {
            histories: Lists {
                offsets: &[0],
                ids: &[],
            },
            weights: None,
            k: r.k,
            exclude_seen: r.exclude_seen,
            include: None,
            no_include: None,
            exclude: None,
            sorted: r.sorted,
            views: &[],
        }
    }
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
    let users = users(b.histories, b.weights)?;
    check_views(model, b.views)?;
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
            let r = b.histories.span(u);
            let req = Request {
                ids: &b.histories.ids[r.clone()],
                weights: b.weights.map(|w| &w[r]),
                k: b.k,
                exclude_seen: b.exclude_seen,
                include: b
                    .include
                    .filter(|_| !b.no_include.is_some_and(|n| n[u]))
                    .map(|l| l.get(u)),
                exclude: b.exclude.map(|l| l.get(u)),
                sorted: b.sorted,
                views: b.views,
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
        let include_bits = req.include.map(|_| &include[..]);
        // Initialized only for a request with views: no stores on the others.
        let mut buf;
        let (allowed, many) = if req.views.iter().all(|v| v.allowed.is_none()) {
            (Masks(m.allowed.as_deref(), include_bits), &[][..])
        } else {
            // The include ids the model knows, duplicates counted: an upper bound,
            // enough to order by.
            let n_include = include_bits.map_or(0, |_| self.included.len());
            buf = [(0, &[][..]); MAX_MASKS];
            masks(m, include_bits, n_include, req.views, &mut buf)
        };
        // SAFETY: hist / blocked ids < n_items (lookup); sums.len() >= block;
        // every mask holds words(n_items) words (views share the core:
        // `check_views`).
        unsafe { self.accumulate_all(m, block, allowed, many) };
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

    /// Scores of `req.items` into `out` (zeros, one per item): see
    /// `Worker::score`. Returns how many history ids the model has.
    fn score(
        &mut self,
        m: &Model,
        block: usize,
        req: &ScoreRequest,
        cand: &mut Vec<(u32, u32)>,
        out: &mut [f32],
    ) -> usize {
        // (internal id, position in `items`) of the candidates with a slot the
        // view allows, by internal id: the gather walks it block by block.
        // The worker's, not a scratch field (recommend's Scratch layout, and
        // so its asm, stays as it is).
        cand.clear();
        cand.extend(req.items.iter().enumerate().filter_map(|(pos, &id)| {
            let i = m.internal(id)?;
            let allowed = |v: &Model| {
                // SAFETY: i < n_items; every mask holds words(n_items) words
                // (the views share the core: `check_views`).
                v.allowed
                    .as_deref()
                    .is_none_or(|a| unsafe { bit(a, i as usize) })
            };
            ((i as usize) < m.n_active && allowed(m) && req.views.iter().all(|v| allowed(v)))
                .then_some((i, pos as u32))
        }));
        cand.sort_unstable();
        self.hist.clear();
        // Looked up even with no candidate: `known` stays exact (the
        // server's `history.unknown`).
        m.lookup_into::<true>(req.ids, req.weights, &mut self.hist);
        let known = self.hist.len();
        let Some(&(last, _)) = cand.last() else {
            return known;
        };
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
        // The row loop hints row i + PREFETCH_ROWS, so the first rows get theirs here.
        prefetch_head(pairs, &self.cursor);
    }

    /// # Safety
    /// See the call in `run`.
    unsafe fn accumulate_all(
        &mut self,
        m: &Model,
        block: usize,
        allowed: Masks,
        many: &[(usize, &[u64])],
    ) {
        let pairs: &[Entry] = &m.pairs;
        let n_active = m.n_active;
        // Per-row column blocks, as the kernel's `blocked_row`; one block (every
        // catalog <= the block size) is this loop once.
        // SAFETY: forwarded.
        unsafe { self.set_cursors(m) };
        let mut head = HEAD.saturating_mul(self.topk.k());
        let mut start = 0;
        while start < n_active {
            let hi = (start + block).min(n_active);
            self.preset(start, hi, f32::NEG_INFINITY);
            // SAFETY: forwarded.
            unsafe { self.accumulate_block(pairs, start, hi, n_active) };
            if !self.touched.is_empty() {
                // SAFETY: forwarded.
                unsafe {
                    if many.is_empty() {
                        self.drain(start, allowed, &mut head);
                    } else {
                        self.drain_many(start, many);
                    }
                };
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
    unsafe fn drain(&mut self, offset: usize, allowed: Masks, head: &mut usize) {
        let (sums, touched, topk) = (&mut self.sums, &self.touched, &mut self.topk);
        // SAFETY: forwarded.
        unsafe {
            match allowed {
                Masks(None, None) => {
                    let n = touched.len().min(*head);
                    *head -= n;
                    if n > 0 {
                        drain_head::<0>(sums, &touched[..n], topk, offset, &[], &[]);
                    }
                    drain_impl::<0>(sums, &touched[n..], topk, offset, &[], &[])
                }
                Masks(Some(a), None) | Masks(None, Some(a)) => {
                    drain_masked::<1>(sums, touched, topk, offset, a, &[], head)
                }
                Masks(Some(a), Some(b)) => {
                    drain_masked::<2>(sums, touched, topk, offset, a, b, head)
                }
            }
        }
        self.touched.clear();
    }

    /// `drain` with three or more masks (`masks`' list).
    ///
    /// # Safety
    /// As `drain`; every mask holds words(n_items) words.
    unsafe fn drain_many(&mut self, offset: usize, masks: &[(usize, &[u64])]) {
        // SAFETY: forwarded.
        unsafe { drain_many(&mut self.sums, &self.touched, &mut self.topk, offset, masks) };
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

/// Hints the first `PREFETCH_ROWS` history rows, once per request (`recommend` and `score`): the row loop only
/// hints rows ahead of it. Probe (M2 Pro, Y500M k 100, per request): cap 10 −4..−5%, cap 50 −1..−2%,
/// cap 200 0..+1.5%. Out of line: the request path keeps its registers.
#[inline(never)]
fn prefetch_head(pairs: &[Entry], cursor: &[(usize, usize, usize)]) {
    for &(pos, end, _) in cursor.iter().take(PREFETCH_ROWS) {
        let bytes = (end - pos).min(PREFETCH_LINES * 8) * std::mem::size_of::<Entry>();
        let base = pairs.as_ptr().wrapping_add(pos).cast::<u8>();
        for off in (0..bytes).step_by(64) {
            prefetch(base.wrapping_add(off));
        }
    }
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

/// The first `HEAD` x k candidates of a request, drained without the floor branch: early on most pass the floor and
/// the branch mispredicts (~k / m taken after m candidates: ~150 per cap-10 request, ~400 at cap 200). Probe (M2
/// Pro, Y500M k 100, per request): cap 10 −7..−8%, cap 50 −3..−5%, cap 200 −1%; k 1000 / 2000 −4..−7%. Past the
/// head, the floor rejects most candidates and the branchy loop wins (a whole-drain branchless variant lost +2.6%
/// at cap 200). `MASKS` as `drain_impl`'s: a masked-out candidate gets score -inf, below every floor (>= 0), so
/// `push_all` keeps the same pushes and cuts as the gated loop (H2b: selective masks keep the floor at 0 longest).
///
/// # Safety
/// As `drain_impl`.
#[inline(never)]
unsafe fn drain_head<const MASKS: u8>(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    offset: usize,
    a: &[u64],
    b: &[u64],
) {
    topk.push_all(
        THRESHOLD,
        touched.iter().map(|&local| {
            let col = offset + local as usize;
            // SAFETY: touched < sums.len(); col < n_items and every mask holds words(n_items) words, so the bits
            // are in bounds for every candidate, not only the ones past the floor (`drain_impl` tests those only).
            unsafe {
                let val = std::mem::take(sums.get_unchecked_mut(local as usize));
                let ok = MASKS == 0 || (bit(a, col) & (MASKS == 1 || bit(b, col)));
                (col as i32, if ok { val } else { f32::NEG_INFINITY })
            }
        }),
    );
}

/// `drain` with one or two masks: the head (`drain_head`), then the gated loop. Out of line: `Scratch::drain`'s
/// no-mask arm keeps its code (inlined here, the masked heads cost it a reload per accepted push).
///
/// # Safety
/// As `drain_impl`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn drain_masked<const MASKS: u8>(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    offset: usize,
    a: &[u64],
    b: &[u64],
    head: &mut usize,
) {
    let n = touched.len().min(*head);
    *head -= n;
    // SAFETY: forwarded.
    unsafe {
        if n > 0 {
            drain_head::<MASKS>(sums, &touched[..n], topk, offset, a, b);
        }
        drain_impl::<MASKS>(sums, &touched[n..], topk, offset, a, b)
    }
}

/// The view's mask and the request's include bits (each None = all allowed),
/// or any two masks of a request with views.
#[derive(Clone, Copy)]
struct Masks<'a>(Option<&'a [u64]>, Option<&'a [u64]>);

/// The served model's mask, the include bits and `MAX_VIEWS` views' masks.
const MAX_MASKS: usize = MAX_VIEWS + 2;

/// The masks of a request with views: the served model's, the include bits
/// (`n_include` ids set) and the views', the present ones only, most
/// selective first (fewest allowed: a rejected item stops at the first test,
/// D-A2-1). Up to two go to the two-mask drains (`Scratch::drain`: `Masks`,
/// the list empty; the same code as today, the masks in selectivity order),
/// more to `drain_many` (the list). Out of line: requests without views never
/// call it.
#[inline(never)]
fn masks<'a>(
    m: &'a Model,
    include: Option<&'a [u64]>,
    n_include: usize,
    views: &[&'a Model],
    buf: &'a mut [(usize, &'a [u64]); MAX_MASKS],
) -> (Masks<'a>, &'a [(usize, &'a [u64])]) {
    let mut n = 0;
    let served = m.allowed.as_deref().map(|a| (m.n_allowed, a));
    let rest = views
        .iter()
        .filter_map(|v| Some((v.n_allowed, v.allowed.as_deref()?)));
    for e in served
        .into_iter()
        .chain(include.map(|a| (n_include, a)))
        .chain(rest)
    {
        buf[n] = e;
        n += 1;
    }
    let found = &mut buf[..n];
    found.sort_unstable_by_key(|&(count, _)| count);
    match *found {
        [] => (Masks(None, None), &[]),
        [(_, a)] => (Masks(Some(a), None), &[]),
        [(_, a), (_, b)] => (Masks(Some(a), Some(b)), &[]),
        _ => (Masks(None, None), found),
    }
}

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

/// `drain_impl` over three or more masks (`masks`' list), each tested only
/// for a score that passes the floor, the most selective first. Out of line,
/// beside `Scratch::drain`: the two-mask drains keep their code. Under
/// selective masks the floor stays at the threshold longest (2k allowed
/// pushes before the first cut), so most positive candidates pay the first
/// test.
///
/// # Safety
/// As `Scratch::drain`; every mask holds words(n_items) words.
#[inline(never)]
unsafe fn drain_many(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    offset: usize,
    masks: &[(usize, &[u64])],
) {
    let mut floor = topk.floor(THRESHOLD);
    for &local in touched {
        // SAFETY: touched < sums.len(); col < n_items (caller).
        unsafe {
            let val = std::mem::take(sums.get_unchecked_mut(local as usize));
            let col = offset + local as usize;
            if val >= floor && masks.iter().all(|&(_, a)| bit(a, col)) {
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
            k: 5,
            include: Some(include),
            ..Default::default()
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
