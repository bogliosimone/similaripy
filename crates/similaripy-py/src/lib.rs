//! Python bindings of the Rust core (`similaripy._core`). `similaripy/_s_plus.py`
//! does the prep and calls `compute`.

mod normalize;
mod progress;

use std::cell::RefCell;
use std::sync::Arc;

use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use progress::ProgressBar;
use similaripy_core::normalize::Value;
use similaripy_core::{CsrPattern, CsrView, Error, Inputs, Norms, Params, compute};

pub(crate) fn to_py(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

thread_local! {
    /// `num_threads = n > 0`: one pool per calling thread, rebuilt when n changes.
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

/// A matrix column selector as `(indptr, indices)`; None = no selector (list
/// selectors are applied in Python). Empty `Some` selects nothing.
type Selection<'py> = Option<(PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, i32>)>;

fn selection<'a>(
    name: &'static str,
    sel: &'a Selection<'_>,
    n_cols: usize,
) -> PyResult<Option<CsrPattern<'a>>> {
    let Some((indptr, indices)) = sel else {
        return Ok(None);
    };
    CsrPattern::new(name, indptr.as_slice()?, indices.as_slice()?, n_cols)
        .map(Some)
        .map_err(to_py)
}

/// The kernel inputs as flat arrays (positional for now), plus `col_map`
/// (output id of each m2 column) and `block_size`
/// (None = no column blocking). Returns `(cols, values, counts)`: row `i` of
/// `targets` has `counts[i]` results, sorted by column, packed in
/// `cols`/`values` (numpy arrays over the Rust buffers, no copy). Releases the
/// GIL. `num_threads = 0` runs on the global rayon pool (concurrent callers
/// queue on it).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn compute_similarities<'py>(
    py: Python<'py>,
    targets: PyReadonlyArray1<i32>,
    m1_data: PyReadonlyArray1<f32>,
    m1_indices: PyReadonlyArray1<i32>,
    m1_indptr: PyReadonlyArray1<i32>,
    m1_n_cols: usize,
    m2_data: PyReadonlyArray1<f32>,
    m2_indices: PyReadonlyArray1<i32>,
    m2_indptr: PyReadonlyArray1<i32>,
    x_tversky: Option<PyReadonlyArray1<f32>>,
    y_tversky: Option<PyReadonlyArray1<f32>>,
    x_cosine: Option<PyReadonlyArray1<f32>>,
    y_cosine: Option<PyReadonlyArray1<f32>>,
    x_depop: Option<PyReadonlyArray1<f32>>,
    y_depop: Option<PyReadonlyArray1<f32>>,
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
    filter: Selection<'py>,
    target: Selection<'py>,
    col_map: PyReadonlyArray1<i32>,
    num_threads: usize,
    block_size: Option<usize>,
    progress: Option<Bound<'_, ProgressBar>>,
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u32>>,
)> {
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
            x_tversky: x_tversky.as_ref().map(|a| a.as_slice()).transpose()?,
            y_tversky: y_tversky.as_ref().map(|a| a.as_slice()).transpose()?,
            x_cosine: x_cosine.as_ref().map(|a| a.as_slice()).transpose()?,
            y_cosine: y_cosine.as_ref().map(|a| a.as_slice()).transpose()?,
            x_depop: x_depop.as_ref().map(|a| a.as_slice()).transpose()?,
            y_depop: y_depop.as_ref().map(|a| a.as_slice()).transpose()?,
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
        filter: selection("filter_cols", &filter, n_output_cols)?,
        target_cols: selection("target_cols", &target, n_output_cols)?,
        col_map: Some(col_map.as_slice()?),
    };

    let block_size = block_size.unwrap_or(0); // the core's 0 = no blocking
    let bar = progress.as_ref().map(|b| b.get());
    let out = py
        .detach(|| {
            // Separate instantiations: without a bar the kernel has no per-row branch.
            let run = || match bar {
                Some(b) => compute(&inp, k, block_size, &|| b.tick()),
                None => compute(&inp, k, block_size, &|| {}),
            };
            with_pool(num_threads, run)
        })
        .map_err(to_py)?;
    Ok((
        out.cols.into_pyarray(py),
        out.values.into_pyarray(py),
        out.counts.into_pyarray(py),
    ))
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

/// In place: renumbers columns by `forward` and sorts every row, carrying
/// `data` (None: the pattern only; prep for the blocked kernel).
#[pyfunction]
fn remap_sort_rows(
    py: Python<'_>,
    indptr: PyReadonlyArray1<'_, i32>,
    mut indices: PyReadwriteArray1<'_, i32>,
    mut data: Option<PyReadwriteArray1<'_, f32>>,
    forward: PyReadonlyArray1<'_, i32>,
    num_threads: usize,
) -> PyResult<()> {
    let (indptr, forward) = (indptr.as_slice()?, forward.as_slice()?);
    let indices = indices.as_slice_mut()?;
    let data = data.as_mut().map(|d| d.as_slice_mut()).transpose()?;
    py.detach(|| {
        with_pool(num_threads, || {
            similaripy_core::prep::remap_sort_rows(indptr, indices, data, forward)
        })
    })
    .map_err(to_py)
}

/// In place: canonical rows (sorted, duplicates summed, zeros dropped),
/// compacted to the front; returns the new indptr. Data float32 or float64.
#[pyfunction]
fn canonicalize_rows<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadwriteArray1<'py, i32>,
    data: &Bound<'py, PyAny>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    match data.extract::<PyReadwriteArray1<'py, f32>>() {
        Ok(d) => canonicalize(py, indptr, indices, d, num_threads),
        Err(_) => canonicalize::<f64>(py, indptr, indices, data.extract()?, num_threads),
    }
}

fn canonicalize<'py, V: Value + Element>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    mut indices: PyReadwriteArray1<'py, i32>,
    mut data: PyReadwriteArray1<'py, V>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let indptr = indptr.as_slice()?;
    let (indices, data) = (indices.as_slice_mut()?, data.as_slice_mut()?);
    let new_indptr = py
        .detach(|| {
            with_pool(num_threads, || {
                similaripy_core::prep::canonicalize_rows(indptr, indices, data)
            })
        })
        .map_err(to_py)?;
    Ok(new_indptr.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_similarities, m)?)?;
    m.add_function(wrap_pyfunction!(remap_sort_rows, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_rows, m)?)?;
    m.add_class::<ProgressBar>()?;
    m.add_function(wrap_pyfunction!(normalize::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::tfidf, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::bm25plus, m)?)
}
