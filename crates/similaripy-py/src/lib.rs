//! Python bindings of the Rust core (`similaripy._core`). `similaripy/_s_plus.py`
//! does the prep and calls `compute`; `similaripy/serving.py` builds the serving
//! model (`serve.rs`).

mod normalize;
mod progress;
mod serve;

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
    match e {
        // FileNotFoundError, FileExistsError, ... as Python's own
        Error::Io(kind, msg) => std::io::Error::new(kind, msg).into(),
        e => PyValueError::new_err(e.to_string()),
    }
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

/// A CSR matrix, read by attribute from `similaripy._s_plus.Csr`.
#[derive(FromPyObject)]
struct CsrArg<'py> {
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    n_cols: usize,
}

impl CsrArg<'_> {
    fn view(&self, name: &'static str) -> PyResult<CsrView<'_>> {
        CsrView::new(
            name,
            self.indptr.as_slice()?,
            self.indices.as_slice()?,
            self.data.as_slice()?,
            self.n_cols,
        )
        .map_err(to_py)
    }
}

/// The normalization terms, read by attribute from `similaripy._s_plus.Norms`
/// (None = unused).
#[derive(FromPyObject)]
struct NormsArg<'py> {
    x_tversky: Option<PyReadonlyArray1<'py, f32>>,
    y_tversky: Option<PyReadonlyArray1<'py, f32>>,
    x_cosine: Option<PyReadonlyArray1<'py, f32>>,
    y_cosine: Option<PyReadonlyArray1<'py, f32>>,
    x_depop: Option<PyReadonlyArray1<'py, f32>>,
    y_depop: Option<PyReadonlyArray1<'py, f32>>,
}

impl NormsArg<'_> {
    fn norms(&self) -> PyResult<Norms<'_>> {
        fn s<'a>(a: &'a Option<PyReadonlyArray1<'_, f32>>) -> PyResult<Option<&'a [f32]>> {
            Ok(a.as_ref().map(|a| a.as_slice()).transpose()?)
        }
        Ok(Norms {
            x_tversky: s(&self.x_tversky)?,
            y_tversky: s(&self.y_tversky)?,
            x_cosine: s(&self.x_cosine)?,
            y_cosine: s(&self.y_cosine)?,
            x_depop: s(&self.x_depop)?,
            y_depop: s(&self.y_depop)?,
        })
    }
}

/// The scoring weights, read by attribute from `similaripy._s_plus.Params`
/// (the core's `Params` stays free of Python).
#[derive(FromPyObject)]
struct ParamsArg {
    a1: f32,
    l1: f32,
    l2: f32,
    l3: f32,
    t1: f32,
    t2: f32,
    stabilized_shrink: f32,
    bayesian_shrink: f32,
    threshold: f32,
}

impl From<ParamsArg> for Params {
    fn from(p: ParamsArg) -> Self {
        Params {
            a1: p.a1,
            l1: p.l1,
            l2: p.l2,
            l3: p.l3,
            t1: p.t1,
            t2: p.t2,
            stabilized_shrink: p.stabilized_shrink,
            bayesian_shrink: p.bayesian_shrink,
            threshold: p.threshold,
        }
    }
}

/// The kernel inputs (`m2.n_cols` = output columns), plus `col_map` (output id
/// of each m2 column) and `block_size` (None = no column blocking). Returns
/// `(cols, values, counts)`: row `i` of `targets` has `counts[i]` results,
/// sorted by column, packed in `cols`/`values` (numpy arrays over the Rust
/// buffers, no copy). Releases the GIL. `num_threads = 0` runs on the global
/// rayon pool (concurrent callers queue on it).
#[pyfunction]
#[pyo3(signature = (*, targets, m1, m2, norms, params, k, filter, target, col_map, num_threads, block_size, progress))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn compute_similarities<'py>(
    py: Python<'py>,
    targets: PyReadonlyArray1<i32>,
    m1: CsrArg<'py>,
    m2: CsrArg<'py>,
    norms: NormsArg<'py>,
    params: ParamsArg,
    k: usize,
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
    let n_cols = m2.n_cols;
    let inp = Inputs {
        m1: m1.view("m1")?,
        m2: m2.view("m2")?,
        targets: targets.as_slice()?,
        norms: norms.norms()?,
        params: params.into(),
        filter: selection("filter_cols", &filter, n_cols)?,
        target_cols: selection("target_cols", &target, n_cols)?,
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
pub(crate) fn with_pool<R>(
    num_threads: usize,
    f: impl FnOnce() -> Result<R, Error> + Send,
) -> Result<R, Error>
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
#[pyo3(signature = (*, indptr, indices, data, forward, num_threads))]
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

/// Interaction logs grouped into a canonical CSR (`prep::group_interactions`):
/// (indptr, indices, data float32, latest timestamps float64 or None).
#[pyfunction]
#[pyo3(signature = (*, rows, cols, values, t, n_rows, n_cols, duplicates))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn group_interactions<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
    values: Option<PyReadonlyArray1<'py, f64>>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    n_rows: usize,
    n_cols: usize,
    duplicates: &str,
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    Option<Bound<'py, PyArray1<f64>>>,
)> {
    use similaripy_core::prep::Duplicates;
    let duplicates = match duplicates {
        "sum" => Duplicates::Sum,
        "first" => Duplicates::First,
        "last" => Duplicates::Last,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown duplicates '{duplicates}'"
            )));
        }
    };
    let (rows, cols) = (rows.as_slice()?, cols.as_slice()?);
    let values = values.as_ref().map(|v| v.as_slice()).transpose()?;
    let t = t.as_ref().map(|t| t.as_slice()).transpose()?;
    let g = py
        .detach(|| {
            similaripy_core::prep::group_interactions(
                rows, cols, values, t, n_rows, n_cols, duplicates,
            )
        })
        .map_err(to_py)?;
    Ok((
        g.indptr.into_pyarray(py),
        g.indices.into_pyarray(py),
        g.data.into_pyarray(py),
        g.latest.map(|l| l.into_pyarray(py)),
    ))
}

/// In place: canonical rows (sorted, duplicates summed, zeros dropped),
/// compacted to the front; returns the new indptr. Data float32 or float64.
#[pyfunction]
#[pyo3(signature = (*, indptr, indices, data, num_threads))]
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
    m.add("DEFAULT_BLOCK_SIZE", similaripy_core::DEFAULT_BLOCK_SIZE)?;
    m.add_function(wrap_pyfunction!(compute_similarities, m)?)?;
    m.add_class::<serve::PyModel>()?;
    m.add_function(wrap_pyfunction!(remap_sort_rows, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_rows, m)?)?;
    m.add_class::<ProgressBar>()?;
    m.add_function(wrap_pyfunction!(group_interactions, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::tfidf, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::recency_rank, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::bm25plus, m)?)
}
