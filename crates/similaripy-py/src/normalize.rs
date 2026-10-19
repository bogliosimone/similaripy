//! Bindings of the in-place normalizations on the canonical copy
//! (`_canonical_csr`): data float32 | float64, indices/indptr int32.

use numpy::{Element, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use similaripy_core::Error;
use similaripy_core::normalize::{self as core, IdfMode, Norm, TfMode, Value};

use crate::to_py;

#[derive(Clone, Copy)]
enum Op {
    Norm(Norm),
    Tfidf(TfMode, IdfMode, f64),
    Bm25plus {
        k1: f64,
        b: f64,
        delta: f64,
        tf: TfMode,
        idf: IdfMode,
        logbase: f64,
    },
}

impl Op {
    fn apply<T: Value, I: Copy + Into<i64> + Sync>(
        self,
        data: &mut [T],
        indices: &[I],
        indptr: &[I],
        n_cols: usize,
    ) -> Result<(), Error> {
        match self {
            Op::Norm(n) => core::normalize_rows(n, data, indptr),
            Op::Tfidf(tf, idf, lb) => core::tfidf(data, indices, indptr, n_cols, tf, idf, lb),
            Op::Bm25plus {
                k1,
                b,
                delta,
                tf,
                idf,
                logbase,
            } => core::bm25plus(
                data, indices, indptr, n_cols, k1, b, delta, tf, idf, logbase,
            ),
        }
    }
}

fn run<T: Value + Element>(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    shape: (usize, usize),
    op: Op,
) -> PyResult<()> {
    let mut data: PyReadwriteArray1<T> = data.extract()?;
    let indices: PyReadonlyArray1<i32> = indices.extract()?;
    let indptr: PyReadonlyArray1<i32> = indptr.extract()?;
    let (data, indices, indptr) = (
        data.as_slice_mut()?,
        indices.as_slice()?,
        indptr.as_slice()?,
    );
    if indptr.len() != shape.0 + 1 {
        return Err(PyValueError::new_err("len(indptr) != n_rows + 1"));
    }
    py.detach(|| op.apply(data, indices, indptr, shape.1))
        .map_err(to_py)
}

/// Picks the data dtype instantiation.
fn dispatch(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    shape: (usize, usize),
    op: Op,
) -> PyResult<()> {
    if data.extract::<PyReadonlyArray1<f32>>().is_ok() {
        run::<f32>(py, data, indices, indptr, shape, op)
    } else if data.extract::<PyReadonlyArray1<f64>>().is_ok() {
        run::<f64>(py, data, indices, indptr, shape, op)
    } else {
        Err(PyTypeError::new_err("expected 1-d float32/float64 data"))
    }
}

/// In-place l1 / l2 / max row normalization.
#[pyfunction]
#[pyo3(signature = (*, data, indices, indptr, shape, norm))]
pub fn normalize(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    shape: (usize, usize),
    norm: &str,
) -> PyResult<()> {
    let norm = match norm {
        "l1" => Norm::L1,
        "l2" => Norm::L2,
        "max" => Norm::Max,
        _ => return Err(PyValueError::new_err(format!("unknown norm '{norm}'"))),
    };
    dispatch(py, data, indices, indptr, shape, Op::Norm(norm))
}

/// In-place TF-IDF (rows = documents).
#[pyfunction]
#[pyo3(signature = (*, data, indices, indptr, shape, tf_mode, idf_mode, logbase))]
#[allow(clippy::too_many_arguments)]
pub fn tfidf(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    shape: (usize, usize),
    tf_mode: &str,
    idf_mode: &str,
    logbase: f64,
) -> PyResult<()> {
    let op = Op::Tfidf(
        TfMode::parse(tf_mode).map_err(to_py)?,
        IdfMode::parse(idf_mode).map_err(to_py)?,
        logbase,
    );
    dispatch(py, data, indices, indptr, shape, op)
}

/// In-place BM25+ (rows = documents); `delta = 0` is BM25.
#[pyfunction]
#[pyo3(signature = (*, data, indices, indptr, shape, k1, b, delta, tf_mode, idf_mode, logbase))]
#[allow(clippy::too_many_arguments)]
pub fn bm25plus(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    shape: (usize, usize),
    k1: f64,
    b: f64,
    delta: f64,
    tf_mode: &str,
    idf_mode: &str,
    logbase: f64,
) -> PyResult<()> {
    let op = Op::Bm25plus {
        k1,
        b,
        delta,
        tf: TfMode::parse(tf_mode).map_err(to_py)?,
        idf: IdfMode::parse(idf_mode).map_err(to_py)?,
        logbase,
    };
    dispatch(py, data, indices, indptr, shape, op)
}
