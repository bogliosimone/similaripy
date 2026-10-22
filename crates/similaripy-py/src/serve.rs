//! Serving bindings: `ServeModel` owns the core `Model` (Rust buffers, one
//! copy at build, no numpy inside, D63). Each call copies the ids in, runs
//! pure Rust without the GIL and moves the results out. `similaripy/_serve.py`
//! does the prep and wraps it as `Recommender`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use similaripy_core::CsrView;
use similaripy_core::serve::{Ids, ItemMask, Model, Request, Unknown, recommend, recommend_batch};

use crate::{to_py, with_pool};

type Arr<'py, T> = Bound<'py, PyArray1<T>>;

fn unknown(s: &str) -> PyResult<Unknown> {
    match s {
        "skip" => Ok(Unknown::Skip),
        "raise" => Ok(Unknown::Raise),
        _ => Err(PyValueError::new_err(format!(
            "unknown must be 'skip' or 'raise', got {s:?}"
        ))),
    }
}

/// A per-call availability mask, valid for the model that built it and the
/// models `with_allowed` derives from the same build.
#[pyclass(frozen, name = "ServeMask", module = "similaripy._core")]
pub struct PyMask(ItemMask);

#[pyclass(frozen, name = "ServeModel", module = "similaripy._core")]
pub struct PyModel(Model);

#[pymethods]
impl PyModel {
    /// S (n × n CSR over the original ids, canonical rows), `new_to_old`
    /// (internal id → original id per kept item, serving order) and `ext`
    /// (external id per original id, None = column indices).
    #[new]
    #[pyo3(signature = (*, indptr, indices, data, n, new_to_old, ext, block, num_threads))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i32>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        n: usize,
        new_to_old: PyReadonlyArray1<'_, i32>,
        ext: Option<PyReadonlyArray1<'_, i64>>,
        block: usize,
        num_threads: usize,
    ) -> PyResult<Self> {
        let (indptr, indices, data) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
        let new_to_old = new_to_old.as_slice()?.to_vec();
        let ext = ext.map(|e| e.as_slice().map(<[i64]>::to_vec)).transpose()?;
        py.detach(|| {
            let s = CsrView::new("S", indptr, indices, data, n)?;
            with_pool(num_threads, || Model::new(s, new_to_old, ext, block))
        })
        .map(Self)
        .map_err(to_py)
    }

    #[getter]
    fn n_items(&self) -> usize {
        self.0.n_items()
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.0.n_cols()
    }

    #[getter]
    fn n_active(&self) -> usize {
        self.0.n_active()
    }

    fn mask(
        &self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, i64>,
        unknown: &str,
    ) -> PyResult<PyMask> {
        let (ids, u) = (ids.as_slice()?, self::unknown(unknown)?);
        py.detach(|| self.0.mask(Ids::External(ids), u))
            .map(PyMask)
            .map_err(to_py)
    }

    fn with_allowed(&self, py: Python<'_>, allowed: &PyMask) -> PyResult<Self> {
        py.detach(|| self.0.with_allowed(&allowed.0))
            .map(Self)
            .map_err(to_py)
    }

    #[pyo3(signature = (ids, weights, *, k, exclude_seen, allowed, unknown))]
    #[allow(clippy::too_many_arguments)]
    fn recommend<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'_, i64>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        k: usize,
        exclude_seen: bool,
        allowed: Option<&PyMask>,
        unknown: &str,
    ) -> PyResult<(Arr<'py, i64>, Arr<'py, f32>)> {
        let ids = ids.as_slice()?.to_vec();
        let weights = weights
            .map(|w| w.as_slice().map(<[f32]>::to_vec))
            .transpose()?;
        let u = self::unknown(unknown)?;
        let model = &self.0;
        let allowed = allowed.map(|a| &a.0);
        let (out_ids, out_scores) = py
            .detach(|| {
                let req = Request {
                    ids: Ids::External(&ids),
                    weights: weights.as_deref(),
                    k,
                    exclude_seen,
                    allowed,
                    unknown: u,
                };
                let mut out = (Vec::new(), Vec::new());
                recommend(model, &req, &mut out.0, &mut out.1).map(|()| out)
            })
            .map_err(to_py)?;
        Ok((out_ids.into_pyarray(py), out_scores.into_pyarray(py)))
    }

    /// One history per row of H (CSR over S's original columns). Returns
    /// (indptr, columns, scores) over S's original columns.
    #[pyo3(signature = (indptr, cols, weights, *, k, exclude_seen, allowed, num_threads))]
    #[allow(clippy::too_many_arguments)]
    fn batch<'py>(
        &self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'_, i64>,
        cols: PyReadonlyArray1<'_, i32>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        k: usize,
        exclude_seen: bool,
        allowed: Option<&PyMask>,
        num_threads: usize,
    ) -> PyResult<(Arr<'py, i64>, Arr<'py, i32>, Arr<'py, f32>)> {
        let (indptr, cols) = (indptr.as_slice()?, cols.as_slice()?);
        let weights = weights.as_ref().map(|w| w.as_slice()).transpose()?;
        let model = &self.0;
        let allowed = allowed.map(|a| &a.0);
        let (p, c, s) = py
            .detach(|| {
                with_pool(num_threads, || {
                    recommend_batch(model, indptr, cols, weights, k, exclude_seen, allowed)
                })
            })
            .map_err(to_py)?;
        Ok((p.into_pyarray(py), c.into_pyarray(py), s.into_pyarray(py)))
    }

    /// Copies of the arrays the model is rebuilt from (pickling): (indptr,
    /// indices, data, new_to_old, n_cols, ext or None), S in internal ids.
    #[allow(clippy::type_complexity)]
    fn parts<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Arr<'py, i32>,
        Arr<'py, i32>,
        Arr<'py, f32>,
        Arr<'py, i32>,
        usize,
        Option<Arr<'py, i64>>,
    ) {
        let p = self.0.parts();
        (
            PyArray1::from_slice(py, p.indptr),
            PyArray1::from_slice(py, p.indices),
            PyArray1::from_slice(py, p.data),
            PyArray1::from_slice(py, p.new_to_old),
            p.n_cols,
            p.ext.map(|e| PyArray1::from_slice(py, e)),
        )
    }
}
