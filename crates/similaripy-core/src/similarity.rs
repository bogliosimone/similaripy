//! Top-K similarities from canonical inputs: the S-Plus pipeline the Python
//! wrappers call (D130). The driver only fixes the step order; each step is a
//! file in `similarity/`:
//!
//! canonical inputs (`prep::canonical`, the caller) -> checks (options, finite)
//! -> binary -> random walk -> score rules -> norms -> depop-only as cosine
//! -> product form + prescale -> list selectors dropped from m2 -> popularity
//! layout -> block size -> `kernel::compute` -> CSR indptr from the counts.
//!
//! Sums (norms, popularity) accumulate in f64 in data order, cast once to f32
//! (RC-3). Results: same version + machine = same bits, whatever the threads.

mod checks;
mod layout;
mod norms;
mod scale;
mod select;

use std::time::Instant;

use crate::Error;
use crate::block::{BlockSize, auto_block_size};
use crate::csr::{CsrPattern, CsrView};
use crate::kernel::{Inputs, Params, compute};
use crate::prep::Csr;

/// Depopularization weights of one side: none (ones), 'sum' (sums of |x|,
/// row sums for m1, column sums for m2), or one weight per row / column
/// (finite, >= 0; > 0 where the row / column has entries). Raised to p1 / p2.
#[derive(Clone, Copy, Debug)]
pub enum Pop<'a> {
    None,
    Sum,
    Weights(&'a [f32]),
}

/// p3alpha / rp3beta: both inputs' rows l1-normalized (transition
/// probabilities), then raised to `alpha` keeping signs. `popularity`
/// (rp3beta): m2's depop weights = its column sums of |x| before the walk
/// (replaces `pop2`).
#[derive(Clone, Copy, Debug)]
pub struct Walk {
    pub alpha: f32,
    pub popularity: bool,
}

/// Columns to exclude (`filter_cols`) or keep (`target_cols`): one id list
/// for every row (in [0, n_cols)), or a canonical matrix (one row of columns
/// per m1 row; values ignored, an entry is selected iff stored).
pub enum Selector<'a> {
    Ids(&'a [i64]),
    Matrix(Csr<f32>),
}

/// Kernel column order: `Popularity` (default) renumbers m2's columns by
/// descending count (popular columns share the cache-hot blocks; ties go to
/// the more popular column, then the lower id); `Original` keeps the ids
/// (ties to the lower id). Results are the same up to tie order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnOrder {
    Popularity,
    Original,
}

/// What the driver is doing, for a progress display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prep,
    /// Scoring `rows` target rows (`on_row` runs once per row).
    Compute {
        rows: usize,
    },
}

/// The `s_plus` arguments. Defaults: dot product, k = 100, every row.
pub struct Options<'a> {
    pub k: usize,
    /// Scoring weights (a1 l1 l2 l3 t1 t2 shrinks threshold).
    pub params: Params,
    /// Cosine exponents (x per m1 row, y per m2 column).
    pub c1: f32,
    pub c2: f32,
    /// Depop exponents.
    pub p1: f32,
    pub p2: f32,
    /// Added to both cosine sums of squares before the power.
    pub additive_shrink: f32,
    pub pop1: Pop<'a>,
    pub pop2: Pop<'a>,
    /// Every stored value becomes 1 (after the finite check, before the walk).
    pub binary: bool,
    pub walk: Option<Walk>,
    /// m1 rows to compute (any order, duplicates ignored); None = all.
    pub target_rows: Option<&'a [i64]>,
    pub filter_cols: Option<Selector<'a>>,
    pub target_cols: Option<Selector<'a>>,
    pub block_size: BlockSize,
    pub column_order: ColumnOrder,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            k: 100,
            params: Params::default(),
            c1: 0.5,
            c2: 0.5,
            p1: 0.0,
            p2: 0.0,
            additive_shrink: 0.0,
            pop1: Pop::None,
            pop2: Pop::None,
            binary: false,
            walk: None,
            target_rows: None,
            filter_cols: None,
            target_cols: None,
            block_size: BlockSize::Auto,
            column_order: ColumnOrder::Popularity,
        }
    }
}

/// The result: an `n_rows x n_cols` canonical CSR (at most k entries per
/// row, rows sorted by column, no zeros; rows not targeted are empty) and
/// the kernel's wall time.
pub struct Similarity {
    pub csr: Csr<f32>,
    pub kernel_seconds: f64,
}

/// Top-`k` scores of `m1 · m2` rows (`n_cols` output columns; m1's columns
/// index m2's rows), normalized as `o` says, on the current rayon pool. Both
/// inputs are the call's own canonical copies (`prep::canonical`), changed in
/// place. `on_row` runs after each finished row, `on_phase` at each phase.
///
/// Errors: invalid options, values not finite, the score rules on signed /
/// weighted input, values out of float32 range for a pre-scaled score.
pub fn similarity<F: Fn() + Sync>(
    mut m1: Csr<f32>,
    mut m2: Csr<f32>,
    n_cols: usize,
    o: Options,
    on_row: &F,
    on_phase: impl Fn(Phase),
) -> Result<Similarity, Error> {
    on_phase(Phase::Prep);
    // Structure first: every later step indexes by these ids.
    CsrView::new("m2", &m2.indptr, &m2.indices, &m2.data, n_cols)?;
    CsrView::new("m1", &m1.indptr, &m1.indices, &m1.data, m2.indptr.len() - 1)?;
    let n_rows = m1.indptr.len() - 1;
    checks::options(&o, n_rows, n_cols)?;
    let targets = select::targets(o.target_rows, n_rows)?;
    let (filter, target) = (
        select::Sel::new("filter_cols", o.filter_cols, n_rows, n_cols)?,
        select::Sel::new("target_cols", o.target_cols, n_rows, n_cols)?,
    );
    checks::finite("matrix1", &m1.data)?;
    checks::finite("matrix2", &m2.data)?;
    if o.binary {
        m1.data.fill(1.0);
        m2.data.fill(1.0);
    }
    let mut pop2 = o.pop2;
    let mut walk_pop = None;
    if let Some(w) = o.walk {
        walk_pop = norms::walk(&mut m1, &mut m2, n_cols, w)?;
        if walk_pop.is_some() {
            pop2 = Pop::Sum; // the pre-walk sums below
        }
    }
    let mut params = o.params;
    // after a walk the values are probabilities, no longer ones
    let binary = o.binary && o.walk.is_none();
    checks::rules(&m1, &m2, n_cols, &params, binary, o.pop1, pop2)?;

    let o_norms = norms::Terms {
        c1: o.c1,
        c2: o.c2,
        p1: o.p1,
        p2: o.p2,
        additive_shrink: o.additive_shrink,
        pop1: o.pop1,
        pop2,
    };
    let mut nm = norms::norms(&m1, &m2, n_cols, &params, &o_norms, walk_pop);
    norms::depop_as_cosine(&mut params, &mut nm);
    if scale::product_form(&params) {
        scale::prescale(&mut m1, &mut m2, params.l2, &nm)?;
        params.l2 = 0.0; // plain dot of the scaled inputs: DOT mode
        nm = norms::Owned::default();
    }
    if let Some(keep) = select::keep(&filter, &target, n_cols) {
        select::drop_columns(&mut m2, &keep);
    }
    let (filter, target) = (filter.into_pattern(), target.into_pattern());
    let mut patterns = [filter, target];
    let col_map = layout::layout(&mut m2, &mut nm, &mut patterns, n_cols, o.column_order)?;
    let [filter, target] = patterns;

    let block = match o.block_size {
        // hot bytes per column: the f32 accumulator + each y-norm array the drain reads
        BlockSize::Auto => auto_block_size(
            n_cols,
            rayon::current_num_threads(),
            4 * (1 + nm.y_arrays()),
        ),
        b => b.resolve(n_cols, rayon::current_num_threads())?,
    };
    let inp = Inputs {
        m1: CsrView::new("m1", &m1.indptr, &m1.indices, &m1.data, m2.indptr.len() - 1)?,
        m2: CsrView::new("m2", &m2.indptr, &m2.indices, &m2.data, n_cols)?,
        targets: &targets,
        norms: nm.view(),
        params,
        filter: pattern("filter_cols", &filter, n_cols)?,
        target_cols: pattern("target_cols", &target, n_cols)?,
        col_map: col_map.as_deref(),
    };
    on_phase(Phase::Compute {
        rows: targets.len(),
    });
    let start = Instant::now();
    let out = compute(&inp, o.k, block, on_row)?;
    let kernel_seconds = start.elapsed().as_secs_f64();

    // Targets are sorted and unique: the rows come out in row order.
    let mut indptr = vec![0i64; n_rows + 1];
    for (&t, &c) in targets.iter().zip(&out.counts) {
        indptr[t as usize + 1] = i64::from(c);
    }
    for i in 0..n_rows {
        indptr[i + 1] += indptr[i];
    }
    Ok(Similarity {
        csr: Csr {
            indptr,
            indices: out.cols,
            data: out.values,
        },
        kernel_seconds,
    })
}

fn pattern<'a>(
    name: &'static str,
    p: &'a Option<select::Pattern>,
    n_cols: usize,
) -> Result<Option<CsrPattern<'a>>, Error> {
    p.as_ref()
        .map(|p| CsrPattern::new(name, &p.indptr, &p.indices, n_cols))
        .transpose()
}
