//! Serving: the top-K items for one history from a prepared item-item model S.
//!
//! The model is S renumbered once: items with neither a row nor a column are
//! dropped, the rest get internal ids in popularity order (chosen by
//! `similaripy/_serve.py`), rows and columns alike, so a history item's
//! internal id is both its row and its column. Columns with entries come
//! first, so the accumulator covers `0..n_active`.
//!
//! A request: ids → internal ids (dense gather or hash, `Ids`), history sorted
//! with duplicates merged, the history's own slots pre-set to `-inf` (never
//! touched, never drained), `scores = h · S` accumulated, then drained through
//! the per-call mask (tested only after the floor reject) into a top-K, best
//! first, internal → caller's ids. Batch = rayon over users calling the same
//! function, so a user gets the same bits from both. Choices measured in step
//! 4.1 (`reports/step4-explore.md` in the planning notes, D64).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::Error;
use crate::csr::CsrView;
use crate::kernel::{accumulate, lower_bound};
use crate::prep::remap_sort_rows;
use crate::topk::TopK;

/// As the similarity functions' default threshold 0: negative scores never
/// enter the top-K, 0.0 ones are dropped by `TopK`.
const THRESHOLD: f32 = 0.0;
/// No internal id (a dropped item) / an empty hash slot.
const EMPTY: u32 = u32::MAX;

fn invalid<T>(msg: impl std::fmt::Display) -> Result<T, Error> {
    Err(Error::InvalidInput(msg.to_string()))
}

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

#[inline(always)]
fn hash(key: i64, shift: u32) -> usize {
    ((key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize
}

/// # Safety
/// `i / 64 < bits.len()`.
#[inline(always)]
unsafe fn bit(bits: &[u64], i: usize) -> bool {
    // SAFETY: caller contract.
    unsafe { (bits.get_unchecked(i / 64) >> (i % 64)) & 1 != 0 }
}

/// External int64 ids: open addressing, multiply-shift hash, linear probing,
/// load <= 0.5 (~2 ns per lookup, D64).
#[derive(Clone)]
struct ExtIds {
    /// External id per original column, dropped items too (`parts`).
    by_col: Vec<i64>,
    /// External id per internal id.
    out: Vec<i64>,
    /// (external id, internal id); internal `EMPTY` = free slot.
    slots: Vec<(i64, u32)>,
    shift: u32,
}

impl ExtIds {
    fn new(by_col: Vec<i64>, new_to_old: &[i32]) -> Result<Self, Error> {
        let mut sorted = by_col.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return invalid("item_ids must be unique");
        }
        drop(sorted);
        let out: Vec<i64> = new_to_old.iter().map(|&c| by_col[c as usize]).collect();
        let cap = (2 * out.len()).next_power_of_two().max(2);
        let shift = 64 - cap.trailing_zeros();
        let mut slots = vec![(0i64, EMPTY); cap];
        for (internal, &key) in out.iter().enumerate() {
            let mut s = hash(key, shift);
            while slots[s].1 != EMPTY {
                s = (s + 1) & (cap - 1);
            }
            slots[s] = (key, internal as u32);
        }
        Ok(Self {
            by_col,
            out,
            slots,
            shift,
        })
    }

    #[inline]
    fn get(&self, id: i64) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut s = hash(id, self.shift);
        loop {
            let (key, v) = self.slots[s];
            if v == EMPTY {
                return None;
            }
            if key == id {
                return Some(v);
            }
            s = (s + 1) & mask;
        }
    }
}

/// A prepared model: owns its arrays, validated once, immutable.
pub struct Model {
    /// Shared by the models `with_allowed` derives (same internal ids), so a
    /// per-call mask works on all of them and on no other model.
    layout: u64,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    n_active: usize,
    /// Column block size, 0 = one block (derived at build, never stored).
    block: usize,
    /// Internal id → S's original column.
    new_to_old: Vec<i32>,
    /// Original column → internal id (`EMPTY`: dropped).
    dense_in: Vec<u32>,
    /// None: ids are S's column indices.
    ext: Option<ExtIds>,
}

/// The ids of one request.
#[derive(Clone, Copy)]
pub enum Ids<'a> {
    /// The model's external ids (S's column indices if it has none).
    External(&'a [i64]),
    /// S's column indices (batch histories).
    Columns(&'a [i32]),
}

impl Ids<'_> {
    fn len(&self) -> usize {
        match self {
            Ids::External(x) => x.len(),
            Ids::Columns(x) => x.len(),
        }
    }
}

/// What to do with ids the model doesn't have (never seen, or dropped at
/// build: no row and no column).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unknown {
    Skip,
    Raise,
}

/// A per-call availability mask: bitset over one layout's internal ids.
pub struct ItemMask {
    layout: u64,
    bits: Vec<u64>,
}

/// One history.
pub struct Request<'a> {
    pub ids: Ids<'a>,
    /// One finite weight per id; None = all 1.0.
    pub weights: Option<&'a [f32]>,
    pub k: usize,
    /// Never recommend a history item.
    pub exclude_seen: bool,
    /// From `Model::mask` of this model or of one sharing its build
    /// (`with_allowed`); None = all allowed.
    pub allowed: Option<&'a ItemMask>,
    pub unknown: Unknown,
}

/// The arrays a model is rebuilt from (pickling): S in internal ids, the
/// layout, S's original item count and the external ids.
pub struct Parts<'a> {
    pub indptr: &'a [i32],
    pub indices: &'a [i32],
    pub data: &'a [f32],
    pub new_to_old: &'a [i32],
    pub n_cols: usize,
    pub ext: Option<&'a [i64]>,
}

fn next_layout() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl Model {
    /// From S (n × n over the original ids; rows sorted, no duplicates) and
    /// `new_to_old`: internal id → original id for every kept item, in the
    /// serving order. Items left out must have an empty row and an empty
    /// column. `ext`: external id per original id (unique), None = the ids
    /// are S's column indices. `block`: column block size of every request,
    /// 0 = one block (as the kernel's: under concurrent requests a
    /// cache-sized accumulator per thread wins, D65). Copies S once into the
    /// internal layout (rows reordered, columns renumbered and sorted, on the
    /// current rayon pool).
    pub fn new(
        s: CsrView,
        new_to_old: Vec<i32>,
        ext: Option<Vec<i64>>,
        block: usize,
    ) -> Result<Self, Error> {
        let n = s.n_cols();
        if s.n_rows() != n {
            return invalid("S must be square (item x item)");
        }
        if ext.as_ref().is_some_and(|e| e.len() != n) {
            return invalid(format!("item_ids must have one id per item ({n})"));
        }
        let mut dense_in = vec![EMPTY; n];
        for (new, &old) in new_to_old.iter().enumerate() {
            match usize::try_from(old).ok().and_then(|o| dense_in.get_mut(o)) {
                Some(d) if *d == EMPTY => *d = new as u32,
                _ => return invalid("new_to_old must list distinct item ids"),
            }
        }
        let (src_indptr, src_indices, src_data) = s.arrays();
        if src_indices.iter().any(|&c| dense_in[c as usize] == EMPTY)
            || (0..n).any(|r| dense_in[r] == EMPTY && src_indptr[r] != src_indptr[r + 1])
        {
            return invalid("new_to_old leaves out an item with entries");
        }
        // Rows in internal order, then columns renumbered and sorted.
        let mut indptr = Vec::with_capacity(new_to_old.len() + 1);
        indptr.push(0i32);
        let mut indices = Vec::with_capacity(src_indices.len());
        let mut data = Vec::with_capacity(src_data.len());
        for &old in &new_to_old {
            let (a, b) = (
                src_indptr[old as usize] as usize,
                src_indptr[old as usize + 1] as usize,
            );
            indices.extend_from_slice(&src_indices[a..b]);
            data.extend_from_slice(&src_data[a..b]);
            indptr.push(indices.len() as i32);
        }
        let forward: Vec<i32> = dense_in.iter().map(|&d| d as i32).collect();
        remap_sort_rows(&indptr, &mut indices, Some(&mut data), &forward)?;
        drop(forward);
        if indptr.windows(2).any(|w| {
            indices[w[0] as usize..w[1] as usize]
                .windows(2)
                .any(|p| p[0] == p[1])
        }) {
            return invalid("S rows must not have duplicate columns");
        }
        let ext = ext.map(|e| ExtIds::new(e, &new_to_old)).transpose()?;
        Ok(Self::from_internal(
            next_layout(),
            block,
            indptr,
            indices,
            data,
            new_to_old,
            dense_in,
            ext,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn from_internal(
        layout: u64,
        block: usize,
        indptr: Vec<i32>,
        indices: Vec<i32>,
        data: Vec<f32>,
        new_to_old: Vec<i32>,
        dense_in: Vec<u32>,
        ext: Option<ExtIds>,
    ) -> Self {
        let n_active = indices.iter().max().map_or(0, |&c| c as usize + 1);
        Self {
            layout,
            indptr,
            indices,
            data,
            n_active,
            block,
            new_to_old,
            dense_in,
            ext,
        }
    }

    /// A model with only the `allowed` items' columns (S · diag(m)); rows,
    /// layout and ids unchanged (Q4), so per-call masks keep working. Rows of
    /// hidden items still count: their history signal stays.
    pub fn with_allowed(&self, allowed: &ItemMask) -> Result<Self, Error> {
        self.check_mask(allowed)?;
        let mut indptr = Vec::with_capacity(self.indptr.len());
        indptr.push(0i32);
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for w in self.indptr.windows(2) {
            for p in w[0] as usize..w[1] as usize {
                let c = self.indices[p];
                // SAFETY: c < n_items, the mask's size (same layout).
                if unsafe { bit(&allowed.bits, c as usize) } {
                    indices.push(c);
                    data.push(self.data[p]);
                }
            }
            indptr.push(indices.len() as i32);
        }
        Ok(Self::from_internal(
            self.layout,
            self.block,
            indptr,
            indices,
            data,
            self.new_to_old.clone(),
            self.dense_in.clone(),
            self.ext.clone(),
        ))
    }

    /// Items with an internal id (S's items minus those dropped at build).
    pub fn n_items(&self) -> usize {
        self.new_to_old.len()
    }

    /// S's original item count (columns of batch histories).
    pub fn n_cols(&self) -> usize {
        self.dense_in.len()
    }

    /// Columns with entries: the accumulator length.
    pub fn n_active(&self) -> usize {
        self.n_active
    }

    pub fn parts(&self) -> Parts<'_> {
        Parts {
            indptr: &self.indptr,
            indices: &self.indices,
            data: &self.data,
            new_to_old: &self.new_to_old,
            n_cols: self.dense_in.len(),
            ext: self.ext.as_ref().map(|e| e.by_col.as_slice()),
        }
    }

    fn s(&self) -> CsrView<'_> {
        // SAFETY: built square over n_items, columns < n_items, never changed.
        unsafe { CsrView::new_unchecked(&self.indptr, &self.indices, &self.data, self.n_items()) }
    }

    fn check_mask(&self, m: &ItemMask) -> Result<(), Error> {
        if m.layout != self.layout {
            return invalid("allowed mask was built by another model");
        }
        Ok(())
    }

    /// A per-call mask allowing only `ids` (unknown ids: as `unknown`).
    pub fn mask(&self, ids: Ids, unknown: Unknown) -> Result<ItemMask, Error> {
        let mut found = Vec::new();
        self.lookup_into(ids, None, unknown, &mut found)?;
        let mut bits = vec![0u64; words(self.n_items())];
        for (i, _) in found {
            bits[i as usize / 64] |= 1 << (i % 64);
        }
        Ok(ItemMask {
            layout: self.layout,
            bits,
        })
    }

    /// Appends (internal id, weight) per known id, in input order.
    fn lookup_into(
        &self,
        ids: Ids,
        weights: Option<&[f32]>,
        unknown: Unknown,
        out: &mut Vec<(u32, f32)>,
    ) -> Result<(), Error> {
        let w = |i: usize| weights.map_or(1.0, |w| w[i]);
        let dense = |id: i64| match usize::try_from(id).ok().and_then(|c| self.dense_in.get(c)) {
            Some(&d) if d != EMPTY => Some(d),
            _ => None,
        };
        let mut push = |i: usize, id: i64, found: Option<u32>| match found {
            Some(d) => {
                out.push((d, w(i)));
                Ok(())
            }
            None if unknown == Unknown::Skip => Ok(()),
            None => invalid(format!("unknown item id: {id}")),
        };
        match (ids, &self.ext) {
            (Ids::External(ids), Some(ext)) => {
                for (i, &id) in ids.iter().enumerate() {
                    push(i, id, ext.get(id))?;
                }
            }
            (Ids::External(ids), None) => {
                for (i, &id) in ids.iter().enumerate() {
                    push(i, id, dense(id))?;
                }
            }
            (Ids::Columns(ids), _) => {
                for (i, &id) in ids.iter().enumerate() {
                    push(i, i64::from(id), dense(i64::from(id)))?;
                }
            }
        }
        Ok(())
    }
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
    let mut out = Vec::with_capacity(req.k.min(model.n_active));
    recommend_internal(model, req, &mut out)?;
    let ext = match (req.ids, &model.ext) {
        (Ids::External(_), Some(e)) => Some(&e.out),
        _ => None,
    };
    for (c, v) in out {
        ids_out.push(ext.map_or(i64::from(model.new_to_old[c as usize]), |e| e[c as usize]));
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
    let cap = k.min(model.n_active);
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
            let mut out = Vec::with_capacity(cap);
            recommend_internal(model, &req, &mut out).map(|()| out)
        })
        .collect::<Result<_, _>>()?;
    let total: usize = rows.iter().map(Vec::len).sum();
    let mut out_indptr = Vec::with_capacity(rows.len() + 1);
    out_indptr.push(0i64);
    let (mut out_cols, mut out_scores) = (Vec::with_capacity(total), Vec::with_capacity(total));
    for row in rows {
        for (c, v) in row {
            out_cols.push(model.new_to_old[c as usize]);
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
        let allowed = req.allowed.map(|a| a.bits.as_slice());
        // SAFETY: hist ids < n_items (lookup); sums.len() >= block; allowed
        // from this layout (checked by the caller), so sized for n_items.
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
        allowed: Option<&[u64]>,
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
    unsafe fn drain(&mut self, offset: usize, allowed: Option<&[u64]>) {
        // SAFETY: forwarded.
        unsafe {
            match allowed {
                None => {
                    drain_impl::<false>(&mut self.sums, &self.touched, &mut self.topk, offset, &[])
                }
                Some(a) => {
                    drain_impl::<true>(&mut self.sums, &self.touched, &mut self.topk, offset, a)
                }
            }
        }
        self.touched.clear();
    }
}

/// DOT drain: score = sum; the mask bit is tested only for scores that pass
/// the floor (most never do: <= 1% cost, D64).
///
/// # Safety
/// As `Scratch::drain`.
#[inline(always)]
unsafe fn drain_impl<const MASK: bool>(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    offset: usize,
    allowed: &[u64],
) {
    let mut floor = topk.floor(THRESHOLD);
    for &local in touched {
        // SAFETY: touched < sums.len(); col < n_items (caller).
        unsafe {
            let val = std::mem::take(sums.get_unchecked_mut(local as usize));
            let col = offset + local as usize;
            if val >= floor && (!MASK || bit(allowed, col)) {
                topk.push(col as i32, val);
                floor = topk.floor(THRESHOLD);
            }
        }
    }
}

/// Sorts by internal id and merges duplicates (weights summed), so the
/// addition order depends on the item set, not the history order (G3).
/// Exception: 3+ copies of one id with different weights sum in input order
/// (f32 addition isn't associative); the same input still gives the same bits.
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
