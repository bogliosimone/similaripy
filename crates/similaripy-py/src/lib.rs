//! Python bindings of the Rust core (`similaripy._core`). `similaripy/_s_plus.py`
//! checks the arguments and calls `similarity` (`similarity.rs`: canonical
//! copies + the S-Plus driver); `similaripy/serving.py` builds the serving
//! model (`serve.rs`).

mod normalize;
mod progress;
mod serve;
mod similarity;

use std::cell::RefCell;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use progress::ProgressBar;
use similaripy_core::{Error, Params};

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

/// The scoring weights, read by attribute from `similaripy._s_plus.Params`
/// (the core's `Params` stays free of Python).
#[derive(FromPyObject)]
pub(crate) struct ParamsArg {
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

/// Interaction logs grouped into a canonical CSR (`prep::group_interactions`):
/// (indptr, indices, data float32).
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
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
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
    ))
}

/// `block_size='auto'` of the similarity functions over `n_cols` output
/// columns on `num_threads` threads (0 = the rayon global pool), `col_bytes`
/// hot bytes per column; 0 = no blocking. See `similaripy_core::block`.
#[pyfunction]
#[pyo3(signature = (n_cols, num_threads, col_bytes = 4))]
fn auto_block_size(n_cols: usize, num_threads: usize, col_bytes: usize) -> usize {
    let threads = if num_threads == 0 {
        rayon::current_num_threads()
    } else {
        num_threads
    };
    similaripy_core::block::auto_block_size(n_cols, threads, col_bytes)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DEFAULT_BLOCK_SIZE", similaripy_core::DEFAULT_BLOCK_SIZE)?;
    // The serving defaults (k, exclude_history, sorted): the Python signatures'
    // literals are checked against them (tests/test_serve.py).
    let d = similaripy_core::serve::Request::default();
    m.add("REQUEST_DEFAULTS", (d.k, d.exclude_history, d.sorted))?;
    m.add_function(wrap_pyfunction!(similarity::similarity, m)?)?;
    m.add_function(wrap_pyfunction!(auto_block_size, m)?)?;
    m.add_class::<serve::PyModel>()?;
    m.add_function(wrap_pyfunction!(similarity::canonical, m)?)?;
    m.add_class::<ProgressBar>()?;
    m.add_function(wrap_pyfunction!(group_interactions, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::tfidf, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::decay_weights, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::user_ages, m)?)?;
    m.add_function(wrap_pyfunction!(normalize::bm25plus, m)?)
}
