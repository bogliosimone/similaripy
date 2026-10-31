//! Serving bindings: `ServeModel` owns the core `Model` (Rust buffers, one
//! copy at build, no numpy inside, D63). The build reads the caller's S
//! arrays in place (the core makes them canonical during its one copy). Each
//! call copies the ids in, runs pure Rust without the GIL and moves the
//! results out. `similaripy/serving.py` wraps it as `ItemKNN`.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use similaripy_core::CsrView;
use similaripy_core::serve::{
    BatchRequest, BlockSize, Lists, Model, Request, recommend, recommend_batch,
};

use crate::{Indptr, to_py, with_pool};

type Arr<'py, T> = Bound<'py, PyArray1<T>>;

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

type Ids<'py> = Option<PyReadonlyArray1<'py, i64>>;

fn lists<'a>(offsets: Option<&'a [i64]>, ids: Option<&'a [i64]>) -> Option<Lists<'a>> {
    offsets
        .zip(ids)
        .map(|(offsets, ids)| Lists { offsets, ids })
}

fn slice<'a, T: numpy::Element>(
    a: &'a Option<PyReadonlyArray1<'_, T>>,
) -> PyResult<Option<&'a [T]>> {
    Ok(a.as_ref().map(|a| a.as_slice()).transpose()?)
}

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
        indptr: Indptr<'_>,
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
        indptr: Indptr<'_>,
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
        let (indptr, indices, new_to_old) =
            (indptr.as_slice()?.to_vec(), v(&indices)?, v(&new_to_old)?);
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

    /// See `Model::save`.
    fn save(
        &self,
        py: Python<'_>,
        path: PathBuf,
        written_by: &str,
        overwrite: bool,
    ) -> PyResult<()> {
        py.detach(|| self.0.save(&path, written_by, overwrite))
            .map_err(to_py)
    }

    /// See `Model::load`.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf, mmap: bool) -> PyResult<Self> {
        py.detach(|| Model::load(&path, mmap))
            .map(Self)
            .map_err(to_py)
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.0.n_cols()
    }

    /// Items with similarities (internal; the benchmark harness reports it).
    #[getter]
    fn n_active(&self) -> usize {
        self.0.n_active()
    }

    /// A copy of the external id per item; None: the ids are column indices.
    fn item_ids<'py>(&self, py: Python<'py>) -> Option<Arr<'py, i64>> {
        self.0.item_ids().map(|e| PyArray1::from_slice(py, e))
    }

    /// A copy whose S keeps only the `candidates`' columns.
    fn trimmed(&self, py: Python<'_>, candidates: PyReadonlyArray1<'_, i64>) -> PyResult<Self> {
        let ids = candidates.as_slice()?;
        py.detach(|| self.0.trimmed(ids)).map(Self).map_err(to_py)
    }

    /// The same model (S shared) with `include` minus `exclude` as its
    /// default mask.
    #[pyo3(signature = (include, exclude))]
    fn view(&self, py: Python<'_>, include: Ids<'_>, exclude: Ids<'_>) -> PyResult<Self> {
        let (include, exclude) = (slice(&include)?, slice(&exclude)?);
        Ok(Self(py.detach(|| self.0.view(include, exclude))))
    }

    #[pyo3(signature = (ids, weights, *, k, exclude_seen, include, exclude, sorted))]
    #[allow(clippy::too_many_arguments)]
    fn recommend<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'_, i64>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        k: usize,
        exclude_seen: bool,
        include: Ids<'_>,
        exclude: Ids<'_>,
        sorted: bool,
    ) -> PyResult<(Arr<'py, i64>, Arr<'py, f32>)> {
        let req = Request {
            ids: ids.as_slice()?,
            weights: slice(&weights)?,
            k,
            exclude_seen,
            include: slice(&include)?,
            exclude: slice(&exclude)?,
            sorted,
        };
        let model = &self.0;
        let (out_ids, out_scores) = py
            .detach(|| {
                let mut out = (Vec::new(), Vec::new());
                recommend(model, &req, &mut out.0, &mut out.1).map(|()| out)
            })
            .map_err(to_py)?;
        Ok((out_ids.into_pyarray(py), out_scores.into_pyarray(py)))
    }

    /// One history per user, flat (`offsets` into `ids` / `weights`), and
    /// optional per-user include / exclude lists the same way. Returns
    /// (offsets, ids, scores), one list per user.
    #[pyo3(signature = (
        offsets, ids, weights, include_offsets, include_ids, no_include, exclude_offsets, exclude_ids,
        *, k, exclude_seen, sorted, num_threads
    ))]
    #[allow(clippy::too_many_arguments)]
    fn batch<'py>(
        &self,
        py: Python<'py>,
        offsets: PyReadonlyArray1<'_, i64>,
        ids: PyReadonlyArray1<'_, i64>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        include_offsets: Ids<'_>,
        include_ids: Ids<'_>,
        no_include: Option<PyReadonlyArray1<'_, bool>>,
        exclude_offsets: Ids<'_>,
        exclude_ids: Ids<'_>,
        k: usize,
        exclude_seen: bool,
        sorted: bool,
        num_threads: usize,
    ) -> PyResult<(Arr<'py, i64>, Arr<'py, i64>, Arr<'py, f32>)> {
        let b = BatchRequest {
            histories: Lists {
                offsets: offsets.as_slice()?,
                ids: ids.as_slice()?,
            },
            weights: slice(&weights)?,
            k,
            exclude_seen,
            include: lists(slice(&include_offsets)?, slice(&include_ids)?),
            no_include: slice(&no_include)?,
            exclude: lists(slice(&exclude_offsets)?, slice(&exclude_ids)?),
            sorted,
        };
        let model = &self.0;
        let (o, i, s) = py
            .detach(|| with_pool(num_threads, || recommend_batch(model, &b)))
            .map_err(to_py)?;
        Ok((o.into_pyarray(py), i.into_pyarray(py), s.into_pyarray(py)))
    }

    /// Copies of what the model is made of (pickling), the arguments of
    /// `from_parts`: (indptr, indices, data, new_to_old, n_cols, ext or None,
    /// default-mask bits or None, block_size as given), S in internal ids.
    #[allow(clippy::type_complexity)]
    fn parts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Arr<'py, i64>,
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
