//! Bindings of the in-place normalizations on the canonical copy
//! (`_canonical_csr`): data float32, indices int32, indptr int64.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use similaripy_core::Error;
use similaripy_core::normalize::{self as core, Axis, IdfMode, Norm, TfMode};

use crate::{to_py, with_pool};

/// The canonical arrays, borrowed; `axis` 1 = rows, 0 = columns (numpy's).
struct Args<'py> {
    data: PyReadwriteArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    axis: Axis,
}

impl<'py> Args<'py> {
    fn new(
        data: PyReadwriteArray1<'py, f32>,
        indices: PyReadonlyArray1<'py, i32>,
        indptr: PyReadonlyArray1<'py, i64>,
        shape: (usize, usize),
        axis: usize,
    ) -> PyResult<Self> {
        let axis = match axis {
            1 => Axis::Rows,
            0 => Axis::Cols,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "axis must be 0 or 1, got {axis}"
                )));
            }
        };
        if indptr.len()? != shape.0 + 1 {
            return Err(PyValueError::new_err("len(indptr) != n_rows + 1"));
        }
        Ok(Args {
            data,
            indices,
            indptr,
            shape,
            axis,
        })
    }

    /// `op(axis, data, indices, indptr, n_cols)` with the GIL released.
    fn run(
        mut self,
        py: Python<'_>,
        op: impl FnOnce(Axis, &mut [f32], &[i32], &[i64], usize) -> Result<(), Error> + Send,
    ) -> PyResult<()> {
        let (axis, n_cols) = (self.axis, self.shape.1);
        let (data, indices, indptr) = (
            self.data.as_slice_mut()?,
            self.indices.as_slice()?,
            self.indptr.as_slice()?,
        );
        py.detach(move || with_pool(0, || op(axis, data, indices, indptr, n_cols)))
            .map_err(to_py)
    }
}

/// In-place l1 / l2 / max normalization of each row (axis 1) or column (axis 0).
#[pyfunction]
#[pyo3(signature = (*, data, indices, indptr, shape, axis, norm))]
pub fn normalize<'py>(
    py: Python<'py>,
    data: PyReadwriteArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    axis: usize,
    norm: &str,
) -> PyResult<()> {
    let norm = match norm {
        "l1" => Norm::L1,
        "l2" => Norm::L2,
        "max" => Norm::Max,
        _ => return Err(PyValueError::new_err(format!("unknown norm '{norm}'"))),
    };
    Args::new(data, indices, indptr, shape, axis)?
        .run(py, |a, d, i, p, n| core::normalize(norm, a, d, i, p, n))
}

/// In-place TF-IDF (documents = rows for axis 1, columns for axis 0).
#[pyfunction]
#[pyo3(signature = (*, data, indices, indptr, shape, axis, tf_mode, idf_mode, logbase))]
#[allow(clippy::too_many_arguments)]
pub fn tfidf<'py>(
    py: Python<'py>,
    data: PyReadwriteArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    axis: usize,
    tf_mode: &str,
    idf_mode: &str,
    logbase: f64,
) -> PyResult<()> {
    let tf = TfMode::parse(tf_mode).map_err(to_py)?;
    let idf = IdfMode::parse(idf_mode).map_err(to_py)?;
    Args::new(data, indices, indptr, shape, axis)?.run(py, |a, d, i, p, n| {
        core::tfidf(a, d, i, p, n, tf, idf, logbase)
    })
}

/// In-place BM25+ (documents = rows for axis 1, columns for axis 0); `delta = 0` is BM25.
#[pyfunction]
#[pyo3(signature = (*, data, indices, indptr, shape, axis, k1, b, delta, tf_mode, idf_mode, logbase))]
#[allow(clippy::too_many_arguments)]
pub fn bm25plus<'py>(
    py: Python<'py>,
    data: PyReadwriteArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    axis: usize,
    k1: f64,
    b: f64,
    delta: f64,
    tf_mode: &str,
    idf_mode: &str,
    logbase: f64,
) -> PyResult<()> {
    let tf = TfMode::parse(tf_mode).map_err(to_py)?;
    let idf = IdfMode::parse(idf_mode).map_err(to_py)?;
    Args::new(data, indices, indptr, shape, axis)?.run(py, |a, d, i, p, n| {
        core::bm25plus(a, d, i, p, n, k1, b, delta, tf, idf, logbase)
    })
}

fn curve(shape: &str, scale: f64, exponent: Option<f64>) -> PyResult<core::Decay> {
    let shape = match shape {
        "exponential" => core::Shape::Exponential,
        "power" => core::Shape::Power(exponent.unwrap_or(1.0)),
        "linear" => core::Shape::Linear,
        "step" => core::Shape::Step,
        _ => return Err(PyValueError::new_err(format!("unknown shape '{shape}'"))),
    };
    core::Decay::new(shape, scale).map_err(to_py)
}

/// The f32 decay weight of each age (`Decay::weights`).
#[pyfunction]
#[pyo3(signature = (*, age, shape, scale, exponent))]
pub fn decay_weights<'py>(
    py: Python<'py>,
    age: PyReadonlyArray1<'py, f64>,
    shape: &str,
    scale: f64,
    exponent: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let curve = curve(shape, scale, exponent)?;
    let age = age.as_slice()?;
    let w = py.detach(|| curve.weights(age)).map_err(to_py)?;
    Ok(w.into_pyarray(py))
}

/// The age of each event within its user's events (`normalize::user_ages`):
/// `kind` "latest" (needs t) | "rank".
#[pyfunction]
#[pyo3(signature = (*, users, n_users, t, kind))]
pub fn user_ages<'py>(
    py: Python<'py>,
    users: PyReadonlyArray1<'py, i32>,
    n_users: usize,
    t: Option<PyReadonlyArray1<'py, f64>>,
    kind: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = match kind {
        "latest" => core::UserAge::SinceLatest,
        "rank" => core::UserAge::Rank,
        _ => return Err(PyValueError::new_err(format!("unknown kind '{kind}'"))),
    };
    let users = users.as_slice()?;
    let t = t.as_ref().map(|t| t.as_slice()).transpose()?;
    let ages = py
        .detach(|| with_pool(0, || core::user_ages(users, n_users, t, kind)))
        .map_err(to_py)?;
    Ok(ages.into_pyarray(py))
}
