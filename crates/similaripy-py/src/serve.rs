//! Serving bindings: `ServeModel` owns the core `Model` (Rust buffers, one
//! copy at build, no numpy inside, D63). The build reads the caller's S
//! arrays in place (the core makes them canonical during its one copy). Each
//! call copies the ids in, runs pure Rust without the GIL and moves the
//! results out. `similaripy/_serve.py` wraps it as `ItemKNN`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use similaripy_core::CsrView;
use similaripy_core::serve::{
    BlockSize, Ids, ItemMask, Model, Request, Unknown, recommend, recommend_batch,
};

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

/// 'auto', None (no blocking) or an int > 0, as the similarity functions'.
fn block_size(v: &Bound<'_, PyAny>) -> PyResult<BlockSize> {
    if v.is_none() {
        return Ok(BlockSize::Off);
    }
    if v.extract::<&str>().is_ok_and(|s| s == "auto") {
        return Ok(BlockSize::Auto);
    }
    match v.extract::<usize>() {
        Ok(n) if n > 0 && !v.is_instance_of::<pyo3::types::PyBool>() => Ok(BlockSize::Size(n)),
        _ => Err(PyValueError::new_err(format!(
            "block_size must be 'auto', None (no blocking) or an integer > 0, got {}",
            v.repr()?
        ))),
    }
}

fn block_size_py(py: Python<'_>, b: BlockSize) -> PyResult<Py<PyAny>> {
    Ok(match b {
        BlockSize::Auto => "auto".into_pyobject(py)?.into_any().unbind(),
        BlockSize::Off => py.None(),
        BlockSize::Size(n) => n.into_pyobject(py)?.into_any().unbind(),
    })
}

/// An availability mask, valid for every model of the build that made it
/// (views, trimmed models) and rejected elsewhere.
#[pyclass(frozen, name = "ServeMask", module = "similaripy._core")]
pub struct PyMask(ItemMask);

#[pyclass(frozen, name = "ServeModel", module = "similaripy._core")]
pub struct PyModel(Model);

#[pymethods]
impl PyModel {
    /// S (n × n CSR over the original ids, any rows: read in place, never
    /// changed) and `ext` (external id per original id, None = column
    /// indices); popularity order.
    #[new]
    #[pyo3(signature = (*, indptr, indices, data, n, ext, block_size, num_threads))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i32>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        n: usize,
        ext: Option<PyReadonlyArray1<'_, i64>>,
        block_size: &Bound<'_, PyAny>,
        num_threads: usize,
    ) -> PyResult<Self> {
        let block = self::block_size(block_size)?;
        let (indptr, indices, data) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
        let ext = ext.map(|e| e.as_slice().map(<[i64]>::to_vec)).transpose()?;
        py.detach(|| {
            let s = CsrView::new("S", indptr, indices, data, n)?;
            with_pool(num_threads, || Model::new(s, ext, block))
        })
        .map(Self)
        .map_err(to_py)
    }

    /// The inverse of `parts` (unpickling), checked as a build's output.
    #[staticmethod]
    #[pyo3(signature = (indptr, indices, data, new_to_old, n_cols, ext, allowed, block_size))]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i32>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        new_to_old: PyReadonlyArray1<'_, i32>,
        n_cols: usize,
        ext: Option<PyReadonlyArray1<'_, i64>>,
        allowed: Option<PyReadonlyArray1<'_, u64>>,
        block_size: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let block = self::block_size(block_size)?;
        let v = |a: &PyReadonlyArray1<'_, i32>| a.as_slice().map(<[i32]>::to_vec);
        let (indptr, indices, new_to_old) = (v(&indptr)?, v(&indices)?, v(&new_to_old)?);
        let data = data.as_slice()?.to_vec();
        let ext = ext.map(|e| e.as_slice().map(<[i64]>::to_vec)).transpose()?;
        let allowed = allowed
            .map(|a| a.as_slice().map(<[u64]>::to_vec))
            .transpose()?;
        py.detach(|| {
            Model::from_parts(
                indptr, indices, data, new_to_old, n_cols, ext, allowed, block,
            )
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

    /// A copy whose S keeps only the `allowed` items' columns.
    fn trimmed(&self, py: Python<'_>, allowed: &PyMask) -> PyResult<Self> {
        py.detach(|| self.0.trimmed(&allowed.0))
            .map(Self)
            .map_err(to_py)
    }

    /// The same model (S shared) with `allowed` as its default mask.
    #[pyo3(signature = (allowed))]
    fn view(&self, allowed: Option<&PyMask>) -> PyResult<Self> {
        self.0.view(allowed.map(|a| &a.0)).map(Self).map_err(to_py)
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

    /// Copies of what the model is made of (pickling), the arguments of
    /// `from_parts`: (indptr, indices, data, new_to_old, n_cols, ext or None,
    /// default-mask bits or None, block_size as given), S in internal ids.
    #[allow(clippy::type_complexity)]
    fn parts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Arr<'py, i32>,
        Arr<'py, i32>,
        Arr<'py, f32>,
        Arr<'py, i32>,
        usize,
        Option<Arr<'py, i64>>,
        Option<Arr<'py, u64>>,
        Py<PyAny>,
    )> {
        let p = self.0.parts();
        Ok((
            PyArray1::from_slice(py, p.indptr),
            PyArray1::from_slice(py, p.indices),
            PyArray1::from_slice(py, p.data),
            PyArray1::from_slice(py, p.new_to_old),
            p.n_cols,
            p.ext.map(|e| PyArray1::from_slice(py, e)),
            p.allowed.map(|a| PyArray1::from_slice(py, a)),
            block_size_py(py, p.block_size)?,
        ))
    }
}
