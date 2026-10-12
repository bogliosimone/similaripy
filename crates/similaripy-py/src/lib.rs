//! Python bindings of the Rust core (`similaripy._core`). `similaripy/_s_plus.py`
//! does the prep and calls `compute`.

mod normalize;
mod progress;

use std::cell::RefCell;
use std::sync::Arc;

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use progress::ProgressBar;
use similaripy_core::{CsrPattern, CsrView, Error, Inputs, Norms, Output, Params, compute};

const SELECTION_MATRIX: i32 = 2;

pub(crate) fn to_py(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

thread_local! {
    /// `num_threads = n > 0`: one pool per calling thread, rebuilt when n changes (D20).
    /// Concurrent callers each get their own team, as with OpenMP.
    static POOL: RefCell<Option<(usize, Arc<ThreadPool>)>> = const { RefCell::new(None) };
}

fn cached_pool(n: usize) -> Result<Arc<ThreadPool>, Error> {
    POOL.with_borrow_mut(|cached| {
        if let Some((m, pool)) = cached.as_ref()
            && *m == n
        {
            return Ok(pool.clone());
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(cached.insert((n, Arc::new(pool))).1.clone())
    })
}

fn selection<'a>(
    mode: i32,
    name: &'static str,
    indptr: &'a [i32],
    indices: &'a [i32],
    n_cols: usize,
) -> PyResult<Option<CsrPattern<'a>>> {
    // NONE and ARRAY (pre-filtered in Python) need no per-row check.
    if mode != SELECTION_MATRIX {
        return Ok(None);
    }
    CsrPattern::new(name, indptr, indices, n_cols)
        .map(Some)
        .map_err(to_py)
}

/// Same arguments as the C++ kernel. Writes into the caller's initialised
/// `rows`/`cols`/`values`; slots past a row's results stay untouched. Releases the GIL. `num_threads = 0` runs on the
/// global rayon pool (concurrent callers queue on it).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_similarities(
    py: Python<'_>,
    targets: PyReadonlyArray1<i32>,
    m1_data: PyReadonlyArray1<f32>,
    m1_indices: PyReadonlyArray1<i32>,
    m1_indptr: PyReadonlyArray1<i32>,
    m1_n_cols: usize,
    m2_data: PyReadonlyArray1<f32>,
    m2_indices: PyReadonlyArray1<i32>,
    m2_indptr: PyReadonlyArray1<i32>,
    x_tversky: PyReadonlyArray1<f32>,
    y_tversky: PyReadonlyArray1<f32>,
    x_cosine: PyReadonlyArray1<f32>,
    y_cosine: PyReadonlyArray1<f32>,
    x_depop: PyReadonlyArray1<f32>,
    y_depop: PyReadonlyArray1<f32>,
    a1: f32,
    l1: f32,
    l2: f32,
    l3: f32,
    t1: f32,
    t2: f32,
    stabilized_shrink: f32,
    bayesian_shrink: f32,
    threshold: f32,
    k: usize,
    n_output_cols: usize,
    filter_mode: i32,
    filter_indptr: PyReadonlyArray1<i32>,
    filter_indices: PyReadonlyArray1<i32>,
    target_mode: i32,
    target_indptr: PyReadonlyArray1<i32>,
    target_indices: PyReadonlyArray1<i32>,
    mut rows: PyReadwriteArray1<i32>,
    mut cols: PyReadwriteArray1<i32>,
    mut values: PyReadwriteArray1<f32>,
    num_threads: usize,
    block_size: usize,
    progress: Option<Bound<'_, ProgressBar>>,
) -> PyResult<()> {
    let m1 = CsrView::new(
        "m1",
        m1_indptr.as_slice()?,
        m1_indices.as_slice()?,
        m1_data.as_slice()?,
        m1_n_cols,
    );
    let m2 = CsrView::new(
        "m2",
        m2_indptr.as_slice()?,
        m2_indices.as_slice()?,
        m2_data.as_slice()?,
        n_output_cols,
    );
    let (m1, m2) = (m1.map_err(to_py)?, m2.map_err(to_py)?);
    let inp = Inputs {
        m1,
        m2,
        targets: targets.as_slice()?,
        norms: Norms {
            x_tversky: x_tversky.as_slice()?,
            y_tversky: y_tversky.as_slice()?,
            x_cosine: x_cosine.as_slice()?,
            y_cosine: y_cosine.as_slice()?,
            x_depop: x_depop.as_slice()?,
            y_depop: y_depop.as_slice()?,
        },
        params: Params {
            a1,
            l1,
            l2,
            l3,
            t1,
            t2,
            stabilized_shrink,
            bayesian_shrink,
            threshold,
        },
        filter: selection(
            filter_mode,
            "filter_cols",
            filter_indptr.as_slice()?,
            filter_indices.as_slice()?,
            n_output_cols,
        )?,
        target_cols: selection(
            target_mode,
            "target_cols",
            target_indptr.as_slice()?,
            target_indices.as_slice()?,
            n_output_cols,
        )?,
    };
    let out = Output {
        rows: rows.as_slice_mut()?,
        cols: cols.as_slice_mut()?,
        values: values.as_slice_mut()?,
    };

    let bar = progress.as_ref().map(|b| b.get());
    py.detach(|| {
        // Separate instantiations: without a bar the kernel has no per-row branch.
        let run = || match bar {
            Some(b) => compute(&inp, k, block_size, out, &|| b.tick()),
            None => compute(&inp, k, block_size, out, &|| {}),
        };
        with_pool(num_threads, run)
    })
    .map_err(to_py)
}

/// Runs `f` on the global rayon pool (`num_threads = 0`) or on the calling
/// thread's cached pool of `num_threads` workers.
fn with_pool<R>(num_threads: usize, f: impl FnOnce() -> Result<R, Error> + Send) -> Result<R, Error>
where
    R: Send,
{
    if num_threads == 0 {
        f()
    } else {
        cached_pool(num_threads)?.install(f)
    }
}

/// In place: renumbers columns by `forward` (empty = keep) and sorts every
/// row, carrying `data` (prep for the blocked kernel).
#[pyfunction]
fn remap_sort_rows(
    py: Python<'_>,
    indptr: PyReadonlyArray1<'_, i32>,
    mut indices: PyReadwriteArray1<'_, i32>,
    mut data: PyReadwriteArray1<'_, f32>,
    forward: PyReadonlyArray1<'_, i32>,
    num_threads: usize,
) -> PyResult<()> {
    let (indptr, forward) = (indptr.as_slice()?, forward.as_slice()?);
    let (indices, data) = (indices.as_slice_mut()?, data.as_slice_mut()?);
    py.detach(|| {
        with_pool(num_threads, || {
            similaripy_core::prep::remap_sort_rows(indptr, indices, data, forward)
        })
    })
    .map_err(to_py)
}

/// Threads used with `num_threads = 0`: the global rayon pool (`RAYON_NUM_THREADS`).
#[pyfunction]
fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_similarities, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(remap_sort_rows, m)?)?;
    m.add_class::<ProgressBar>()?;
    m.add_function(wrap_pyfunction!(normalize::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::tfidf, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::bm25plus, m)?)
}
