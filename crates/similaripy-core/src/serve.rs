//! Serving: the top-K items for one history from a prepared item-item model S.
//!
//! S is square with one internal id per item (row id = column id), in the
//! popularity layout built by `similaripy/_serve.py`: columns with entries come
//! first, so the accumulator covers `0..n_active`. A request: external ids →
//! internal (lookup), history sorted with duplicates merged, `scores = h · S`
//! accumulated per column block (as the kernel's `blocked_row`), drained
//! through the seen and mask tests into a top-K, best first, internal →
//! external ids. Batch = rayon over users calling the same function, so a user
//! gets the same bits from both.
//!
//! Step 4.1 exploration: `Variant` picks a candidate implementation per stage;
//! the losers go when the variants are locked (step 4.2).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::Error;
use crate::csr::CsrView;
use crate::kernel::{accumulate, lower_bound};
use crate::topk::TopK;

/// As the similarity functions' default threshold 0: negative scores never
/// enter the top-K, 0.0 ones are dropped by `TopK`.
const THRESHOLD: f32 = 0.0;
/// Empty hash slot.
const EMPTY: u32 = u32::MAX;

/// External id → internal id. Exploration only (V1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// Ids are S's original column indices: a dense gather (G11).
    Dense,
    /// `binary_search` over the sorted external ids.
    Binary,
    /// Branchless search over the sorted ids in Eytzinger (BFS) order.
    Eytzinger,
    /// Open addressing, multiply-shift hash, linear probing.
    Hash,
    /// As `Hash`, but every id's first probe is loaded before any compare, so
    /// the cache misses overlap whatever the branch predictor does.
    HashBatched,
}

/// How the history's own items are kept out of the results (V2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seen {
    /// Their accumulator slots start at `-inf`: never touched, never drained.
    NegInf,
    /// A bitset test after the floor reject.
    Bitset,
    /// Over-fetch `k + len(history)`, filter the final list.
    Later,
}

/// Where the per-call `allowed` mask is tested (V3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    AfterFloor,
    BeforeFloor,
    /// Over-fetch `2k`, filter the final list (may return fewer than k).
    Later,
}

/// One candidate per stage. Exploration only.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub lookup: Lookup,
    pub seen: Seen,
    pub mask: Mask,
    /// Column block size; 0 or `>= n_active` = one block (V4).
    pub block: usize,
    /// Sort the history by internal id and merge duplicates (V8). Off: history
    /// order, a duplicate adds twice (same scores up to f32 rounding).
    pub canonicalize: bool,
    /// Reuse the calling thread's scratch (V5); off: allocate per call.
    pub reuse_scratch: bool,
}

impl Default for Variant {
    fn default() -> Self {
        Self {
            lookup: Lookup::Dense,
            seen: Seen::NegInf,
            mask: Mask::AfterFloor,
            block: 0,
            canonicalize: true,
            reuse_scratch: true,
        }
    }
}

/// A prepared model: owns its arrays, validated once, immutable.
pub struct Model {
    /// Unique per process: binds the masks built by this model (G5).
    id: u64,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    n: usize,
    n_active: usize,
    /// Original column → internal id.
    dense: Vec<u32>,
    /// Internal id → external id.
    out_ids: Vec<i64>,
    sorted_keys: Vec<i64>,
    sorted_vals: Vec<u32>,
    /// 1-based Eytzinger order, slot 0 unused.
    eyt_keys: Vec<i64>,
    eyt_vals: Vec<u32>,
    slots: Vec<(i64, u32)>,
    hash_shift: u32,
}

/// A per-call availability mask: bitset over one model's internal ids. Only
/// that model accepts it (another model of the same size has another order).
pub struct ItemMask {
    model: u64,
    bits: Vec<u64>,
}

/// One history.
pub struct Request<'a> {
    pub ids: &'a [i64],
    /// None = all 1.0.
    pub weights: Option<&'a [f32]>,
    pub k: usize,
    pub exclude_seen: bool,
    /// From `Model::mask` of the same model; None = all allowed.
    pub allowed: Option<&'a ItemMask>,
}

fn invalid<T>(msg: &str) -> Result<T, Error> {
    Err(Error::InvalidInput(format!("serve: {msg}")))
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

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

impl Model {
    /// `S` (n × n, CSR, rows sorted, no duplicates) in internal ids; `new_to_old`
    /// = internal id → S's original column (a permutation); `ext[c]` = external
    /// id of original column `c` (unique).
    pub fn new(
        indptr: Vec<i32>,
        indices: Vec<i32>,
        data: Vec<f32>,
        new_to_old: &[i32],
        ext: &[i64],
    ) -> Result<Self, Error> {
        let n = new_to_old.len();
        if indptr.len() != n + 1 {
            return invalid("S must be square, len(indptr) == len(new_to_old) + 1");
        }
        if ext.len() != n {
            return invalid("len(item_ids) != number of items");
        }
        CsrView::new("S", &indptr, &indices, &data, n)?;
        if indptr
            .windows(2)
            .any(|w| !indices[w[0] as usize..w[1] as usize].is_sorted_by(|a, b| a < b))
        {
            return invalid("S rows must be strictly sorted");
        }
        let mut dense = vec![EMPTY; n];
        for (new, &old) in new_to_old.iter().enumerate() {
            match dense.get_mut(old as usize) {
                Some(d) if *d == EMPTY => *d = new as u32,
                _ => return invalid("new_to_old is not a permutation"),
            }
        }
        let n_active = indices.iter().max().map_or(0, |&c| c as usize + 1);
        let out_ids: Vec<i64> = new_to_old.iter().map(|&old| ext[old as usize]).collect();

        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by_key(|&i| out_ids[i as usize]);
        let sorted_keys: Vec<i64> = order.iter().map(|&i| out_ids[i as usize]).collect();
        if sorted_keys.windows(2).any(|w| w[0] == w[1]) {
            return invalid("item ids must be unique");
        }

        let (mut eyt_keys, mut eyt_vals) = (vec![0; n + 1], vec![0; n + 1]);
        fn fill(src: (&[i64], &[u32]), dst: (&mut [i64], &mut [u32]), next: &mut usize, k: usize) {
            if k < dst.0.len() {
                fill(src, (&mut *dst.0, &mut *dst.1), next, 2 * k);
                (dst.0[k], dst.1[k]) = (src.0[*next], src.1[*next]);
                *next += 1;
                fill(src, dst, next, 2 * k + 1);
            }
        }
        fill(
            (&sorted_keys, &order),
            (&mut eyt_keys, &mut eyt_vals),
            &mut 0,
            1,
        );

        let cap = (2 * n).next_power_of_two().max(2);
        let hash_shift = 64 - cap.trailing_zeros();
        let mut slots = vec![(0i64, EMPTY); cap];
        for (i, &key) in out_ids.iter().enumerate() {
            let mut s = hash(key, hash_shift);
            while slots[s].1 != EMPTY {
                s = (s + 1) & (cap - 1);
            }
            slots[s] = (key, i as u32);
        }

        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            indptr,
            indices,
            data,
            n,
            n_active,
            dense,
            out_ids,
            sorted_keys,
            sorted_vals: order,
            eyt_keys,
            eyt_vals,
            slots,
            hash_shift,
        })
    }

    pub fn n_items(&self) -> usize {
        self.n
    }

    /// Columns with entries: the accumulator length.
    pub fn n_active(&self) -> usize {
        self.n_active
    }

    fn s(&self) -> CsrView<'_> {
        // SAFETY: validated in `new`, never changed.
        unsafe { CsrView::new_unchecked(&self.indptr, &self.indices, &self.data, self.n) }
    }

    /// The per-call mask allowing only the known `ids`.
    pub fn mask(&self, ids: &[i64], how: Lookup) -> ItemMask {
        let mut bits = vec![0u64; words(self.n)];
        let (mut found, mut probe) = (Vec::new(), Vec::new());
        self.lookup_into(ids, None, how, &mut found, &mut probe);
        for (i, _) in found {
            bits[i as usize / 64] |= 1 << (i % 64);
        }
        ItemMask {
            model: self.id,
            bits,
        }
    }

    /// Appends (internal id, weight) for each known id, in input order; unknown
    /// ids are skipped.
    pub fn lookup_into(
        &self,
        ids: &[i64],
        weights: Option<&[f32]>,
        how: Lookup,
        out: &mut Vec<(u32, f32)>,
        probe: &mut Vec<(u32, i64)>,
    ) {
        let w = |i: usize| weights.map_or(1.0, |w| w[i]);
        match how {
            Lookup::Dense => {
                for (i, &id) in ids.iter().enumerate() {
                    if let Some(&d) = usize::try_from(id).ok().and_then(|id| self.dense.get(id)) {
                        out.push((d, w(i)));
                    }
                }
            }
            Lookup::Binary => {
                for (i, id) in ids.iter().enumerate() {
                    if let Ok(p) = self.sorted_keys.binary_search(id) {
                        out.push((self.sorted_vals[p], w(i)));
                    }
                }
            }
            Lookup::Eytzinger => {
                let keys = &self.eyt_keys;
                for (i, &id) in ids.iter().enumerate() {
                    let mut k = 1;
                    while k < keys.len() {
                        // SAFETY: k < keys.len().
                        k = 2 * k + (unsafe { *keys.get_unchecked(k) } < id) as usize;
                    }
                    // Undo the right turns past the answer (Khuong & Morin).
                    k >>= k.trailing_ones() + 1;
                    if k != 0 && keys[k] == id {
                        out.push((self.eyt_vals[k], w(i)));
                    }
                }
            }
            Lookup::Hash => {
                for (i, &id) in ids.iter().enumerate() {
                    if let Some(v) = self.probe_from(hash(id, self.hash_shift), id) {
                        out.push((v, w(i)));
                    }
                }
            }
            Lookup::HashBatched => {
                probe.clear();
                probe.extend(ids.iter().map(|&id| {
                    let s = hash(id, self.hash_shift);
                    (s as u32, self.slots[s].0)
                }));
                for (i, (&id, &(s, first))) in ids.iter().zip(probe.iter()).enumerate() {
                    let s = s as usize;
                    let v = if first == id && self.slots[s].1 != EMPTY {
                        Some(self.slots[s].1)
                    } else {
                        self.probe_from(s, id)
                    };
                    if let Some(v) = v {
                        out.push((v, w(i)));
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn probe_from(&self, mut s: usize, id: i64) -> Option<u32> {
        let mask = self.slots.len() - 1;
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

/// Per-thread state reused across requests; every buffer is empty (and `sums`
/// all 0.0, `seen` all clear) between requests. Grows to the largest model.
#[derive(Default)]
struct Scratch {
    sums: Vec<f32>,
    touched: Vec<u32>,
    cursor: Vec<(u32, u32)>,
    topk: TopK,
    seen: Vec<u64>,
    hist: Vec<(u32, f32)>,
    probe: Vec<(u32, i64)>,
    cols: Vec<i32>,
    vals: Vec<f32>,
}

thread_local! {
    /// Taken out for a request, put back only on a normal return: a panic
    /// mid-request drops the dirty scratch instead of reusing it (G4).
    static SCRATCH: Cell<Option<Scratch>> = const { Cell::new(None) };
}

/// The top `req.k` items for one history, best first (score desc, then the
/// more popular item), appended to `ids_out` (external ids) and `scores_out`.
/// Unknown ids are skipped. Fewer than k results when fewer items score > 0.
pub fn recommend(
    model: &Model,
    req: &Request,
    v: &Variant,
    ids_out: &mut Vec<i64>,
    scores_out: &mut Vec<f32>,
) -> Result<(), Error> {
    if let Some(w) = req.weights {
        if w.len() != req.ids.len() {
            return invalid("len(weights) != len(ids)");
        }
        if !w.iter().all(|x| x.is_finite()) {
            return invalid("weights must be finite");
        }
    }
    if req.allowed.is_some_and(|a| a.model != model.id) {
        return invalid("allowed mask was built for another model");
    }
    if !v.reuse_scratch {
        Scratch::default().run(model, req, v, ids_out, scores_out);
        return Ok(());
    }
    let mut s = SCRATCH.take().unwrap_or_default();
    s.run(model, req, v, ids_out, scores_out);
    SCRATCH.set(Some(s));
    Ok(())
}

/// Batch results as CSR: (indptr, ids, scores).
pub type Batch = (Vec<i64>, Vec<i64>, Vec<f32>);

/// `recommend` for each row of H (`indptr` into `ids` / `weights`) on the
/// current rayon pool. Returns the results as CSR: (indptr, ids, scores).
pub fn recommend_batch(
    model: &Model,
    indptr: &[i64],
    ids: &[i64],
    weights: Option<&[f32]>,
    k: usize,
    exclude_seen: bool,
    v: &Variant,
) -> Result<Batch, Error> {
    if indptr.first() != Some(&0)
        || indptr.windows(2).any(|w| w[0] > w[1])
        || indptr.last().map(|&e| e as usize) != Some(ids.len())
        || weights.is_some_and(|w| w.len() != ids.len())
    {
        return invalid("bad history CSR");
    }
    let rows: Vec<(Vec<i64>, Vec<f32>)> = (0..indptr.len() - 1)
        .into_par_iter()
        // Small tasks: request cost varies 100x across users.
        .with_max_len(4)
        .map(|u| {
            let (a, b) = (indptr[u] as usize, indptr[u + 1] as usize);
            let req = Request {
                ids: &ids[a..b],
                weights: weights.map(|w| &w[a..b]),
                k,
                exclude_seen,
                allowed: None,
            };
            let cap = k.min(model.n_active);
            let mut out = (Vec::with_capacity(cap), Vec::with_capacity(cap));
            recommend(model, &req, v, &mut out.0, &mut out.1).map(|()| out)
        })
        .collect::<Result<_, _>>()?;
    let mut out_indptr = Vec::with_capacity(rows.len() + 1);
    out_indptr.push(0i64);
    let total: usize = rows.iter().map(|r| r.0.len()).sum();
    let (mut out_ids, mut out_scores) = (Vec::with_capacity(total), Vec::with_capacity(total));
    for (i, s) in rows {
        out_ids.extend_from_slice(&i);
        out_scores.extend_from_slice(&s);
        out_indptr.push(out_ids.len() as i64);
    }
    Ok((out_indptr, out_ids, out_scores))
}

impl Scratch {
    fn run(
        &mut self,
        m: &Model,
        req: &Request,
        v: &Variant,
        ids_out: &mut Vec<i64>,
        scores_out: &mut Vec<f32>,
    ) {
        self.hist.clear();
        m.lookup_into(
            req.ids,
            req.weights,
            v.lookup,
            &mut self.hist,
            &mut self.probe,
        );
        if v.canonicalize {
            canonicalize(&mut self.hist);
        }
        let k = req.k.min(m.n_active);
        let seen_later = req.exclude_seen && v.seen == Seen::Later;
        let mask_later = req.allowed.is_some() && v.mask == Mask::Later;
        let extra = if seen_later { self.hist.len() } else { 0 } + if mask_later { k } else { 0 };
        self.topk.set_k((k + extra).min(m.n_active));

        let seen_bits = req.exclude_seen && v.seen != Seen::NegInf;
        if seen_bits {
            if self.seen.len() < words(m.n) {
                self.seen.resize(words(m.n), 0);
            }
            for &(i, _) in &self.hist {
                self.seen[i as usize / 64] |= 1 << (i % 64);
            }
        }
        let block = if v.block == 0 || v.block >= m.n_active {
            m.n_active
        } else {
            v.block
        };
        if self.sums.len() < block {
            self.sums.resize(block, 0.0);
        }
        let neg_inf = req.exclude_seen && v.seen == Seen::NegInf;
        let in_drain = (
            req.exclude_seen && v.seen == Seen::Bitset,
            if req.allowed.is_none() {
                0
            } else {
                match v.mask {
                    Mask::AfterFloor => 1,
                    Mask::BeforeFloor => 2,
                    Mask::Later => 0,
                }
            },
        );
        let allowed = req.allowed.map_or(&[][..], |a| &a.bits);
        // SAFETY: hist ids < n (lookup); sums.len() >= block; seen sized for n
        // when tested; allowed built by this model (checked in `recommend`), so
        // sized for n.
        unsafe { self.accumulate_all(m, block, neg_inf, in_drain, allowed) };

        self.cols.clear();
        self.vals.clear();
        self.topk.drain_by_score(&mut self.cols, &mut self.vals);
        let mut kept = 0;
        for (&c, &val) in self.cols.iter().zip(&self.vals) {
            if kept == k {
                break;
            }
            let c = c as usize;
            // SAFETY: c < n_active <= n; bitsets sized for n when used.
            let drop =
                unsafe { (seen_later && bit(&self.seen, c)) || (mask_later && !bit(allowed, c)) };
            if !drop {
                ids_out.push(m.out_ids[c]);
                scores_out.push(val);
                kept += 1;
            }
        }
        if seen_bits {
            for &(i, _) in &self.hist {
                self.seen[i as usize / 64] = 0;
            }
        }
    }

    /// # Safety
    /// See the call in `run`.
    unsafe fn accumulate_all(
        &mut self,
        m: &Model,
        block: usize,
        neg_inf: bool,
        in_drain: (bool, u8),
        allowed: &[u64],
    ) {
        let s = m.s();
        let n_active = m.n_active;
        let mut start = 0;
        if block == n_active {
            self.preset(0, n_active, neg_inf, f32::NEG_INFINITY);
            for &(id, w) in &self.hist {
                // SAFETY: id < n; S indices < n_active == sums.len() bound.
                unsafe {
                    let (c, vals) = s.row_unchecked(id as usize);
                    accumulate(&mut self.sums, &mut self.touched, c, vals, w, 0);
                }
            }
            // SAFETY: forwarded.
            unsafe { self.drain(0, in_drain, allowed) };
            self.preset(0, n_active, neg_inf, 0.0);
            return;
        }
        // Per-row column blocks, as the kernel's `blocked_row`.
        self.cursor.clear();
        self.cursor.extend(self.hist.iter().map(|&(id, _)| {
            // SAFETY: id < n.
            let (a, b) = unsafe { s.row_span_unchecked(id as usize) };
            (a as u32, b as u32)
        }));
        while start < n_active {
            let hi = (start + block).min(n_active);
            self.preset(start, hi, neg_inf, f32::NEG_INFINITY);
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
                unsafe { self.drain(start, in_drain, allowed) };
            }
            // Even when the drain was skipped: the slots were set.
            self.preset(start, hi, neg_inf, 0.0);
            start = hi;
        }
    }

    /// Sets the history's slots inside `[lo, hi)` to `val` (`-inf` before
    /// accumulating, 0.0 after the drain). Ids `>= n_active` (empty column)
    /// never get a slot (G2).
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
    /// Touched + `offset` < n; bitsets sized for n where tested.
    #[inline(never)]
    unsafe fn drain(&mut self, offset: usize, in_drain: (bool, u8), allowed: &[u64]) {
        macro_rules! with {
            ($seen:literal, $mask:literal) => {
                drain_impl::<$seen, $mask>(
                    &mut self.sums,
                    &self.touched,
                    &mut self.topk,
                    offset,
                    &self.seen,
                    allowed,
                )
            };
        }
        // SAFETY: forwarded.
        unsafe {
            match in_drain {
                (false, 0) => with!(false, 0),
                (false, 1) => with!(false, 1),
                (false, _) => with!(false, 2),
                (true, 0) => with!(true, 0),
                (true, 1) => with!(true, 1),
                (true, _) => with!(true, 2),
            }
        }
        self.touched.clear();
    }
}

/// DOT drain: score = sum. `MASK`: 0 none, 1 after the floor reject, 2 before.
///
/// # Safety
/// As `Scratch::drain`.
#[inline(always)]
unsafe fn drain_impl<const SEEN: bool, const MASK: u8>(
    sums: &mut [f32],
    touched: &[u32],
    topk: &mut TopK,
    offset: usize,
    seen: &[u64],
    allowed: &[u64],
) {
    let mut floor = topk.floor(THRESHOLD);
    for &local in touched {
        // SAFETY: touched < sums.len(); col < n (caller).
        unsafe {
            let val = std::mem::take(sums.get_unchecked_mut(local as usize));
            let col = offset + local as usize;
            if MASK == 2 && !bit(allowed, col) {
                continue;
            }
            if val >= floor {
                if (SEEN && bit(seen, col)) || (MASK == 1 && !bit(allowed, col)) {
                    continue;
                }
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
