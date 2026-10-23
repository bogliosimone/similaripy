//! Serving: the top-K items for one history from a prepared item-item model S.
//!
//! The model is S made canonical and renumbered in one copy: items with
//! neither a row nor a column are dropped, the rest get internal ids in
//! popularity order (descending column count, ties to the lower column), rows
//! and columns alike, so a history item's internal id is both its row and its
//! column. Columns with entries come first, so the accumulator covers
//! `0..n_active`.
//!
//! A request: ids → internal ids (dense gather or hash, `Ids`), history sorted
//! with duplicates merged, the history's own slots pre-set to `-inf` (never
//! touched, never drained), `scores = h · S` accumulated, then drained through
//! the per-call mask (tested only after the floor reject) into a top-K, best
//! first, internal → caller's ids. Batch = rayon over users calling the same
//! function, so a user gets the same bits from both. Choices measured in step
//! 4.1 (`reports/step4-explore.md` in the planning notes, D64).

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::csr::CsrView;
use crate::kernel::{accumulate, lower_bound};
use crate::prep::{LEAF_NNZ, canonical_row, par_leaves, sort_rows};
use crate::topk::TopK;
use crate::{DEFAULT_BLOCK_SIZE, Error};

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

/// The immutable part of a model: S in the internal layout and the id maps,
/// validated once, shared by every `view` of it (no copy of S).
pub struct Core {
    /// Shared by the models of one build (views, `trimmed`: same internal
    /// ids), so a mask works on all of them and on no other model.
    layout: u64,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    n_active: usize,
    /// As given (saved, pickled).
    block_size: BlockSize,
    /// Column block size of a request, 0 = one block (`block_size` resolved).
    block: usize,
    /// Internal id → S's original column.
    new_to_old: Vec<i32>,
    /// Original column → internal id (`EMPTY`: dropped).
    dense_in: Vec<u32>,
    /// None: ids are S's column indices.
    ext: Option<ExtIds>,
}

/// A prepared model: a shared `Core` plus an optional default availability
/// mask (`view`), ANDed with a per-call mask. Immutable: to change the mask,
/// make a new view and swap the reference (in-flight requests keep theirs).
pub struct Model {
    core: Arc<Core>,
    /// Bitset over internal ids; None = all allowed.
    allowed: Option<Vec<u64>>,
}

impl std::ops::Deref for Model {
    type Target = Core;

    fn deref(&self) -> &Core {
        &self.core
    }
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
    /// (views, `trimmed`); None = all allowed.
    pub allowed: Option<&'a ItemMask>,
    pub unknown: Unknown,
}

/// Column block size of every request, as the similarity functions'
/// `block_size`. Kept as given (saved, pickled): `Auto` follows the library's
/// choice of the version that loads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockSize {
    /// `DEFAULT_BLOCK_SIZE`.
    Auto,
    /// One block (no blocking): for a lone request on an idle machine.
    Off,
    /// Entries per block, > 0.
    Size(usize),
}

impl BlockSize {
    fn resolve(self) -> Result<usize, Error> {
        match self {
            BlockSize::Auto => Ok(DEFAULT_BLOCK_SIZE),
            BlockSize::Off => Ok(0),
            BlockSize::Size(0) => invalid("block_size must be > 0"),
            BlockSize::Size(n) => Ok(n),
        }
    }
}

/// What a model is made of (pickling, saving): S in internal ids, the
/// layout, S's original item count, the external ids, the default mask and
/// the block size as given; `Model::from_parts` takes the same back.
pub struct Parts<'a> {
    pub indptr: &'a [i32],
    pub indices: &'a [i32],
    pub data: &'a [f32],
    pub new_to_old: &'a [i32],
    pub n_cols: usize,
    pub ext: Option<&'a [i64]>,
    /// Bitset over internal ids, None = all allowed.
    pub allowed: Option<&'a [u64]>,
    pub block_size: BlockSize,
}

fn next_layout() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl Model {
    /// From S (n × n CSR over the original ids, as given: unsorted rows,
    /// duplicates and zeros are fine; read, never changed), in popularity
    /// order. `ext`: external id per original id (unique), None = the ids are
    /// S's column indices. No default mask. Peak memory: the model's own
    /// arrays plus O(n) (no canonical copy of S).
    pub fn new(s: CsrView, ext: Option<Vec<i64>>, block_size: BlockSize) -> Result<Self, Error> {
        let (lens, counts) = canonical_counts(&s)?;
        let mut order: Vec<i32> = (0..s.n_cols() as i32).collect();
        order.sort_by_key(|&c| std::cmp::Reverse(counts[c as usize])); // stable: ties to the lower column
        order.retain(|&c| lens[c as usize] > 0 || counts[c as usize] > 0);
        Self::build(s, &lens, &counts, order, ext, block_size)
    }

    /// As `new`, in a given serving order: `new_to_old` = internal id →
    /// original id for every kept item; items left out must have no entries
    /// (once S is canonical).
    pub fn with_order(
        s: CsrView,
        new_to_old: Vec<i32>,
        ext: Option<Vec<i64>>,
        block_size: BlockSize,
    ) -> Result<Self, Error> {
        let (lens, counts) = canonical_counts(&s)?;
        Self::build(s, &lens, &counts, new_to_old, ext, block_size)
    }

    /// The one copy of S: each row made canonical in a small per-thread
    /// buffer (as in `canonical_counts`), columns renumbered, sorted.
    fn build(
        s: CsrView,
        lens: &[u32],
        counts: &[u32],
        new_to_old: Vec<i32>,
        ext: Option<Vec<i64>>,
        block_size: BlockSize,
    ) -> Result<Self, Error> {
        let n = s.n_cols();
        let dense_in = dense_in(&new_to_old, n)?;
        if (0..n).any(|i| dense_in[i] == EMPTY && (lens[i] > 0 || counts[i] > 0)) {
            return invalid("new_to_old leaves out an item with entries");
        }
        let mut indptr = Vec::with_capacity(new_to_old.len() + 1);
        indptr.push(0i32);
        let mut total = 0usize;
        for &old in &new_to_old {
            total += lens[old as usize] as usize;
            indptr.push(total as i32); // <= S's nnz, an i32
        }
        let (mut indices, mut data) = (vec![0i32; total], vec![0f32; total]);
        let (src_ptr, src_cols, src_vals) = s.arrays();
        par_leaves(
            &indptr,
            &mut indices,
            &mut data,
            LEAF_NNZ,
            &|row0, ip, out_c, out_v| {
                let base = ip[0] as usize;
                let (mut cols, mut vals, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
                for (i, w) in ip.windows(2).enumerate() {
                    let old = new_to_old[row0 + i] as usize;
                    let k = canonical_copy(
                        (src_ptr, src_cols, src_vals),
                        old,
                        &mut cols,
                        &mut vals,
                        &mut pairs,
                    );
                    let (a, z) = (w[0] as usize - base, w[1] as usize - base);
                    // Same row, same steps as in `canonical_counts`.
                    assert_eq!(k, z - a, "S changed during the build");
                    for (j, (&c, &v)) in cols[..k].iter().zip(&vals).enumerate() {
                        out_c[a + j] = dense_in[c as usize] as i32;
                        out_v[a + j] = v;
                    }
                }
                sort_rows(ip, out_c, out_v);
            },
        );
        drop(dense_in);
        let core = Core::from_arrays(
            next_layout(),
            indptr,
            indices,
            data,
            new_to_old,
            n,
            ext,
            block_size,
        )?;
        Ok(Self {
            core: Arc::new(core),
            allowed: None,
        })
    }

    /// A model from `parts` (pickling, loading), checked as a build's output.
    /// A new build for masks: masks of the original are rejected.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        indptr: Vec<i32>,
        indices: Vec<i32>,
        data: Vec<f32>,
        new_to_old: Vec<i32>,
        n_cols: usize,
        ext: Option<Vec<i64>>,
        allowed: Option<Vec<u64>>,
        block_size: BlockSize,
    ) -> Result<Self, Error> {
        let core = Core::from_arrays(
            next_layout(),
            indptr,
            indices,
            data,
            new_to_old,
            n_cols,
            ext,
            block_size,
        )?;
        if allowed
            .as_ref()
            .is_some_and(|a| a.len() != words(core.n_items()))
        {
            return invalid("allowed: one bit per item expected");
        }
        Ok(Self {
            core: Arc::new(core),
            allowed,
        })
    }

    /// The same model (S shared, no copy) with `allowed` as its default
    /// mask (None: all allowed). A view of a view replaces the mask, it
    /// doesn't narrow it.
    pub fn view(&self, allowed: Option<&ItemMask>) -> Result<Self, Error> {
        if let Some(m) = allowed {
            self.check_mask(m)?;
        }
        Ok(Self {
            core: Arc::clone(&self.core),
            allowed: allowed.map(|m| m.bits.clone()),
        })
    }

    /// A model whose S keeps only the `allowed` items' columns (S · diag(m),
    /// a copy): they can never be recommended, and their entries cost no
    /// memory or work. Rows, layout and ids unchanged, so masks keep working;
    /// rows of hidden items still count (their history signal stays). The
    /// default mask is kept.
    pub fn trimmed(&self, allowed: &ItemMask) -> Result<Self, Error> {
        Ok(Self {
            core: Arc::new(self.core.trimmed(allowed)?),
            allowed: self.allowed.clone(),
        })
    }

    pub fn parts(&self) -> Parts<'_> {
        Parts {
            indptr: &self.indptr,
            indices: &self.indices,
            data: &self.data,
            new_to_old: &self.new_to_old,
            n_cols: self.dense_in.len(),
            ext: self.ext.as_ref().map(|e| e.by_col.as_slice()),
            allowed: self.allowed.as_deref(),
            block_size: self.block_size,
        }
    }
}

/// Original id → internal id (`EMPTY`: dropped); `new_to_old` must list
/// distinct ids < `n_cols`.
fn dense_in(new_to_old: &[i32], n_cols: usize) -> Result<Vec<u32>, Error> {
    if n_cols > i32::MAX as usize {
        return invalid("more than 2^31 - 1 items");
    }
    let mut dense_in = vec![EMPTY; n_cols];
    for (new, &old) in new_to_old.iter().enumerate() {
        match usize::try_from(old).ok().and_then(|o| dense_in.get_mut(o)) {
            Some(d) if *d == EMPTY => *d = new as u32,
            _ => return invalid("new_to_old must list distinct item ids"),
        }
    }
    Ok(dense_in)
}

/// Pass 1 of the build: per original row its canonical length, per column
/// its canonical entry count (sequential: one O(nnz) pass, rows reused in
/// one small buffer).
fn canonical_counts(s: &CsrView) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let n = s.n_cols();
    if s.n_rows() != n {
        return invalid("S must be square (item x item)");
    }
    let (mut lens, mut counts) = (vec![0u32; n], vec![0u32; n]);
    let (mut cols, mut vals, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
    for (r, len) in lens.iter_mut().enumerate() {
        let k = canonical_copy(s.arrays(), r, &mut cols, &mut vals, &mut pairs);
        *len = k as u32;
        for &c in &cols[..k] {
            counts[c as usize] += 1;
        }
    }
    Ok((lens, counts))
}

/// Row `r` of S made canonical (`canonical_row`) in `cols`/`vals`; returns
/// its length.
fn canonical_copy(
    (indptr, indices, data): (&[i32], &[i32], &[f32]),
    r: usize,
    cols: &mut Vec<i32>,
    vals: &mut Vec<f32>,
    pairs: &mut Vec<(i32, f32)>,
) -> usize {
    let (a, b) = (indptr[r] as usize, indptr[r + 1] as usize);
    cols.clear();
    cols.extend_from_slice(&indices[a..b]);
    vals.clear();
    vals.extend_from_slice(&data[a..b]);
    canonical_row(cols, vals, pairs)
}

impl Core {
    /// The internal layout, checked once whatever made it (a build, a pickle,
    /// a saved model): every invariant the unchecked reads rely on, plus
    /// finite values. O(nnz).
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        layout: u64,
        indptr: Vec<i32>,
        indices: Vec<i32>,
        data: Vec<f32>,
        new_to_old: Vec<i32>,
        n_cols: usize,
        ext: Option<Vec<i64>>,
        block_size: BlockSize,
    ) -> Result<Self, Error> {
        let block = block_size.resolve()?;
        let dense_in = dense_in(&new_to_old, n_cols)?;
        let n = new_to_old.len();
        let s = CsrView::new("S", &indptr, &indices, &data, n)?;
        if s.n_rows() != n {
            return invalid("S must have one row per item");
        }
        if indptr.windows(2).any(|w| {
            !indices[w[0] as usize..w[1] as usize]
                .windows(2)
                .all(|p| p[0] < p[1])
        }) {
            return invalid("S rows must be sorted, without duplicate columns");
        }
        if !data.iter().all(|v| v.is_finite()) {
            return invalid("S must be finite (in float32)");
        }
        if ext.as_ref().is_some_and(|e| e.len() != n_cols) {
            return invalid(format!("item_ids must have one id per item ({n_cols})"));
        }
        let ext = ext.map(|e| ExtIds::new(e, &new_to_old)).transpose()?;
        let n_active = indices.iter().max().map_or(0, |&c| c as usize + 1);
        Ok(Self {
            layout,
            indptr,
            indices,
            data,
            n_active,
            block_size,
            block,
            new_to_old,
            dense_in,
            ext,
        })
    }

    fn trimmed(&self, allowed: &ItemMask) -> Result<Self, Error> {
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
        Self::from_arrays(
            self.layout,
            indptr,
            indices,
            data,
            self.new_to_old.clone(),
            self.n_cols(),
            self.ext.as_ref().map(|e| e.by_col.clone()),
            self.block_size,
        )
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
