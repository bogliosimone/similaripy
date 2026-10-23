//! The model: S in the internal layout and the id maps, built from S (one
//! canonical copy, popularity order), from parts or from a saved directory,
//! always through `Core::from_arrays` (the one check); views and trimmed
//! copies of it.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ids::{ExtIds, ItemMask, dense_in};
use super::{EMPTY, bit, invalid, words};
use crate::csr::CsrView;
use crate::prep::{LEAF_NNZ, canonical_copy, canonical_counts, par_leaves, sort_rows};
use crate::store::{self, Buf};
use crate::{DEFAULT_BLOCK_SIZE, Error};

/// Saved model format (`meta.json`): a version reads every format up to its
/// own; bump it when the files change meaning (Q1).
const FORMAT: u64 = 1;
const KIND: &str = "ItemKNN";
/// Every file a saved model may have.
const FILES: [&str; 7] = [
    "meta.json",
    "indptr.npy",
    "indices.npy",
    "data.npy",
    "new_to_old.npy",
    "item_ids.npy",
    "allowed.npy",
];

fn check_square(s: &CsrView) -> Result<(), Error> {
    if s.n_rows() != s.n_cols() {
        return invalid("S must be square (item x item)");
    }
    if s.n_cols() > i32::MAX as usize {
        return invalid("more than 2^31 - 1 items");
    }
    Ok(())
}

/// A saved model and nothing else (`overwrite` replaces only that): a
/// `meta.json` of this kind, model files only.
fn is_saved_model(dir: &Path) -> bool {
    let kind = std::fs::read_to_string(dir.join("meta.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .is_some_and(|m| m.get("kind").is_some_and(|k| k == KIND));
    kind && std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.into_iter().all(|e| {
            e.is_ok_and(|e| {
                e.file_type().is_ok_and(|t| t.is_file())
                    && FILES.contains(&e.file_name().to_string_lossy().as_ref())
            })
        })
    })
}

/// The immutable part of a model: S in the internal layout and the id maps,
/// validated once, shared by every `view` of it (no copy of S). Built only by
/// `Core::from_arrays` (the one check): the unchecked reads in `request`
/// rely on it, so the fields are visible inside `serve` only.
pub struct Core {
    /// Shared by the models of one build (views, `trimmed`: same internal
    /// ids), so a mask works on all of them and on no other model.
    pub(super) layout: u64,
    pub(super) indptr: Buf<i32>,
    pub(super) indices: Buf<i32>,
    pub(super) data: Buf<f32>,
    pub(super) n_active: usize,
    /// As given (saved, pickled).
    pub(super) block_size: BlockSize,
    /// Column block size of a request, 0 = one block (`block_size` resolved).
    pub(super) block: usize,
    /// Internal id → S's original column.
    pub(super) new_to_old: Buf<i32>,
    /// Original column → internal id (`EMPTY`: dropped).
    pub(super) dense_in: Vec<u32>,
    /// None: ids are S's column indices.
    pub(super) ext: Option<ExtIds>,
}

/// A prepared model: a shared `Core` plus an optional default availability
/// mask (`view`), ANDed with a per-call mask. Immutable: to change the mask,
/// make a new view and swap the reference (in-flight requests keep theirs).
pub struct Model {
    pub(super) core: Arc<Core>,
    /// Bitset over internal ids; None = all allowed.
    pub(super) allowed: Option<Vec<u64>>,
}

impl std::ops::Deref for Model {
    type Target = Core;

    fn deref(&self) -> &Core {
        &self.core
    }
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
        check_square(&s)?;
        let (lens, counts) = canonical_counts(&s);
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
        check_square(&s)?;
        let (lens, counts) = canonical_counts(&s);
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
            indptr.into(),
            indices.into(),
            data.into(),
            new_to_old.into(),
            n,
            ext.map(Buf::from),
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
            indptr.into(),
            indices.into(),
            data.into(),
            new_to_old.into(),
            n_cols,
            ext.map(Buf::from),
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

    /// Saves the model in `dir`: one `.npy` per array (S in internal ids,
    /// `new_to_old`, `item_ids` = the external id per original column, and
    /// `allowed` = the default mask's bits) plus `meta.json`, written last.
    /// A missing or empty `dir` is filled; a non-empty one is an error unless
    /// `overwrite` and it holds a saved model and nothing else (not a
    /// symlink), which is then replaced by a new directory (renamed into
    /// place: processes that mapped the old files keep them; on Windows the
    /// rename is refused while any process maps them, and the old model
    /// stays; `dir` is briefly missing, one saver at a time). `written_by`
    /// goes into `meta.json`.
    pub fn save(&self, dir: &Path, written_by: &str, overwrite: bool) -> Result<(), Error> {
        let p = self.parts();
        store::save_dir(dir, overwrite, is_saved_model, |d| {
            store::write(d, "indptr", p.indptr)?;
            store::write(d, "indices", p.indices)?;
            store::write(d, "data", p.data)?;
            store::write(d, "new_to_old", p.new_to_old)?;
            if let Some(e) = p.ext {
                store::write(d, "item_ids", e)?;
            }
            if let Some(a) = p.allowed {
                store::write(d, "allowed", a)?;
            }
            let block_size = match p.block_size {
                BlockSize::Auto => serde_json::json!("auto"),
                BlockSize::Off => serde_json::Value::Null,
                BlockSize::Size(n) => serde_json::json!(n),
            };
            let meta = serde_json::json!({
                "format": FORMAT,
                "kind": KIND,
                "written_by": written_by,
                "value_dtype": "float32",
                "n_items": p.new_to_old.len(),
                "n_cols": p.n_cols,
                "nnz": p.data.len(),
                "item_ids": p.ext.is_some(),
                "allowed": p.allowed.is_some(),
                "block_size": block_size,
            });
            let text = serde_json::to_string_pretty(&meta).expect("plain JSON values") + "\n";
            store::write_text(d, "meta.json", &text)
        })
    }

    /// A model saved by `save` (this format version or an older one),
    /// mapped (`mmap`: the arrays stay in the page cache, shared by every
    /// process that maps them; the files must not change meanwhile) or read
    /// into memory, and checked as a build's output (O(nnz), which also pages
    /// in a mapped S). A new build for masks. A corrupted file is an error,
    /// never an out-of-bounds read; resource exhaustion (a `meta.json` asking
    /// for billions of items) is not guarded: load files you trust.
    pub fn load(dir: &Path, mmap: bool) -> Result<Self, Error> {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path).map_err(store::io(&path))?;
        let bad = |m: String| Error::InvalidInput(format!("{}: {m}", path.display()));
        let meta: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
        let field = |k: &str| meta.get(k).ok_or_else(|| bad(format!("no {k:?}")));
        let uint = |k: &str| {
            field(k)?
                .as_u64()
                .ok_or_else(|| bad(format!("{k:?} must be an integer >= 0")))
        };
        let flag = |k: &str| {
            field(k)?
                .as_bool()
                .ok_or_else(|| bad(format!("{k:?} must be true or false")))
        };
        let format = uint("format")?;
        if format > FORMAT {
            return Err(bad(format!(
                "saved by a newer similaripy (model format {format}); this version reads formats up to {FORMAT}: upgrade similaripy"
            )));
        }
        if format == 0 || field("kind")? != KIND {
            return Err(bad(format!("not a saved {KIND} model")));
        }
        match field("value_dtype")?.as_str() {
            Some("float32") => {}
            v => {
                return Err(bad(format!(
                    "value dtype {v:?} not supported by this version: upgrade similaripy"
                )));
            }
        }
        let block_size = match field("block_size")? {
            serde_json::Value::Null => BlockSize::Off,
            v if v == "auto" => BlockSize::Auto,
            v => match v.as_u64() {
                Some(n) if n > 0 => BlockSize::Size(n as usize),
                _ => {
                    return Err(bad(format!(
                        "block_size must be \"auto\", null or an integer > 0, got {v}"
                    )));
                }
            },
        };
        let n_cols =
            usize::try_from(uint("n_cols")?).map_err(|_| bad("n_cols too large".into()))?;
        let ext = flag("item_ids")?
            .then(|| store::read::<i64>(dir, "item_ids", mmap))
            .transpose()?;
        let allowed = flag("allowed")?
            .then(|| store::read::<u64>(dir, "allowed", false).map(Buf::into_vec))
            .transpose()?;
        let core = Core::from_arrays(
            next_layout(),
            store::read(dir, "indptr", mmap)?,
            store::read(dir, "indices", mmap)?,
            store::read(dir, "data", mmap)?,
            store::read(dir, "new_to_old", mmap)?,
            n_cols,
            ext,
            block_size,
        )?;
        if (core.n_items() as u64, core.data.len() as u64) != (uint("n_items")?, uint("nnz")?) {
            return Err(bad("n_items / nnz don't match the arrays".into()));
        }
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
            ext: self.ext.as_ref().map(|e| &e.by_col[..]),
            allowed: self.allowed.as_deref(),
            block_size: self.block_size,
        }
    }
}

impl Core {
    /// The internal layout, checked once whatever made it (a build, a pickle,
    /// a saved model): every invariant the unchecked reads rely on, plus
    /// finite values. O(nnz).
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        layout: u64,
        indptr: Buf<i32>,
        indices: Buf<i32>,
        data: Buf<f32>,
        new_to_old: Buf<i32>,
        n_cols: usize,
        ext: Option<Buf<i64>>,
        block_size: BlockSize,
    ) -> Result<Self, Error> {
        let block = block_size.resolve()?;
        // Before `dense_in` allocates n_cols slots (n_cols may come from a file).
        if ext.as_ref().is_some_and(|e| e.len() != n_cols) {
            return invalid(format!("item_ids must have one id per item ({n_cols})"));
        }
        let dense_in = dense_in(&new_to_old, n_cols)?;
        let n = new_to_old.len();
        let (ip, ix, d): (&[i32], &[i32], &[f32]) = (&indptr, &indices, &data);
        let s = CsrView::new("S", ip, ix, d, n)?;
        if s.n_rows() != n {
            return invalid("S must have one row per item");
        }
        if ip.windows(2).any(|w| {
            !ix[w[0] as usize..w[1] as usize]
                .windows(2)
                .all(|p| p[0] < p[1])
        }) {
            return invalid("S rows must be sorted, without duplicate columns");
        }
        if !d.iter().all(|v| v.is_finite()) {
            return invalid("S must be finite (in float32)");
        }
        let n_active = ix.iter().max().map_or(0, |&c| c as usize + 1);
        let ext = ext.map(|e| ExtIds::new(e, &new_to_old)).transpose()?;
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
        let (ip, ix, d): (&[i32], &[i32], &[f32]) = (&self.indptr, &self.indices, &self.data);
        let mut indptr = Vec::with_capacity(ip.len());
        indptr.push(0i32);
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for w in ip.windows(2) {
            for p in w[0] as usize..w[1] as usize {
                let c = ix[p];
                // SAFETY: c < n_items, the mask's size (same layout).
                if unsafe { bit(&allowed.bits, c as usize) } {
                    indices.push(c);
                    data.push(d[p]);
                }
            }
            indptr.push(indices.len() as i32);
        }
        Self::from_arrays(
            self.layout,
            indptr.into(),
            indices.into(),
            data.into(),
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

    pub(super) fn s(&self) -> CsrView<'_> {
        // SAFETY: built square over n_items, columns < n_items, never changed.
        unsafe { CsrView::new_unchecked(&self.indptr, &self.indices, &self.data, self.n_items()) }
    }
}
