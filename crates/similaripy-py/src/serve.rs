//! Serving bindings: `ServeModel` owns the core `Model` (Rust buffers, one
//! copy at build, no numpy inside, D63) and its block size, a runtime setting
//! (never saved, W1): resolved once for single calls, once per batch call. The build reads the caller's S
//! arrays in place (the core makes them canonical during its one copy). Each
//! call copies the ids in, runs pure Rust without the GIL and moves the
//! results out. `similaripy/serving.py` wraps it as `ItemKNN`.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use similaripy_core::CsrView;
use similaripy_core::serve::{
    BatchRequest, BlockSize, Entry, ItemOrder, Lists, Model, Request, recommend, recommend_batch,
    score, score_batch,
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

/// 'auto', 'clustered', 'popularity' or 'original'.
fn item_order(v: &str) -> PyResult<ItemOrder> {
    Ok(match v {
        "auto" => ItemOrder::Auto,
        "clustered" => ItemOrder::Clustered,
        "popularity" => ItemOrder::Popularity,
        "original" => ItemOrder::Original,
        _ => {
            return Err(PyValueError::new_err(format!(
                "item_order must be 'auto', 'clustered', 'popularity' or 'original', got {v:?}"
            )));
        }
    })
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
pub struct PyModel {
    model: Model,
    /// As given (pickled).
    block_size: BlockSize,
    /// For single calls: they can't see the server's other callers, so
    /// 'auto' assumes every core on the L2 busy (usize::MAX clamps to its
    /// cores; D65).
    block: usize,
}

impl PyModel {
    fn wrap(model: Model, block_size: BlockSize) -> Result<Self, similaripy_core::Error> {
        let block = block_size.resolve(model.n_active(), usize::MAX)?;
        Ok(Self {
            model,
            block_size,
            block,
        })
    }
}

#[pymethods]
impl PyModel {
    /// S (n × n CSR over the original ids, any rows: read in place, never
    /// changed), `ext` (external id per original id, None = column
    /// indices) and `keep` (one flag per column, None = all: S trimmed to
    /// them first, then ordered), in `item_order`.
    #[new]
    #[pyo3(signature = (*, indptr, indices, data, n, ext, keep, item_order, block_size, num_threads))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        indptr: Indptr<'_>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        n: usize,
        ext: Option<PyReadonlyArray1<'_, i64>>,
        keep: Option<PyReadonlyArray1<'_, bool>>,
        item_order: &str,
        block_size: &Bound<'_, PyAny>,
        num_threads: usize,
    ) -> PyResult<Self> {
        let order = self::item_order(item_order)?;
        let block = self::block_size(block_size)?;
        let (indptr, indices, data) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
        let ext = ext.map(|e| e.as_slice().map(<[i64]>::to_vec)).transpose()?;
        let keep = slice(&keep)?;
        py.detach(|| {
            let s = CsrView::new("S", indptr, indices, data, n)?;
            with_pool(num_threads, || Model::new(s, ext, keep, order))
                .and_then(|m| Self::wrap(m, block))
        })
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
        let (indices, data) = (indices.as_slice()?, data.as_slice()?);
        if indices.len() != data.len() {
            return Err(PyValueError::new_err("indices and data differ in length"));
        }
        // Python keeps S as two arrays (pickles stay plain numpy); the model
        // holds them interleaved.
        let pairs = indices
            .iter()
            .zip(data)
            .map(|(&col, &val)| Entry { col, val })
            .collect();
        let (indptr, new_to_old) = (indptr.as_slice()?.to_vec(), new_to_old.as_slice()?.to_vec());
        let ext = ext.map(|e| e.as_slice().map(<[i64]>::to_vec)).transpose()?;
        let allowed = allowed
            .map(|a| a.as_slice().map(<[u64]>::to_vec))
            .transpose()?;
        py.detach(|| {
            Model::from_parts(indptr, pairs, new_to_old, n_cols, ext, allowed)
                .and_then(|m| Self::wrap(m, block))
        })
        .map_err(to_py)
    }

    /// See `Model::save`.
    fn save(&self, py: Python<'_>, path: PathBuf, written_by: &str) -> PyResult<()> {
        py.detach(|| self.model.save(&path, written_by))
            .map_err(to_py)
    }

    /// See `Model::load`; `block_size` as `new`'s.
    #[staticmethod]
    fn load(
        py: Python<'_>,
        path: PathBuf,
        mmap: bool,
        block_size: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let block = self::block_size(block_size)?;
        py.detach(|| Model::load(&path, mmap).and_then(|m| Self::wrap(m, block)))
            .map_err(to_py)
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.model.n_cols()
    }

    /// Items with similarities (internal; the benchmark harness reports it).
    #[getter]
    fn n_active(&self) -> usize {
        self.model.n_active()
    }

    /// A copy of the external id per item; None: the ids are column indices.
    fn item_ids<'py>(&self, py: Python<'py>) -> Option<Arr<'py, i64>> {
        self.model.item_ids().map(|e| PyArray1::from_slice(py, e))
    }

    /// The same model (S shared) with `include` minus `exclude` as its
    /// default mask.
    #[pyo3(signature = (include, exclude))]
    fn view(&self, py: Python<'_>, include: Ids<'_>, exclude: Ids<'_>) -> PyResult<Self> {
        let (include, exclude) = (slice(&include)?, slice(&exclude)?);
        Ok(Self {
            model: py.detach(|| self.model.view(include, exclude)),
            block_size: self.block_size,
            block: self.block,
        })
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
        let (model, block) = (&self.model, self.block);
        let (out_ids, out_scores) = py
            .detach(|| {
                let mut out = (Vec::new(), Vec::new());
                recommend(model, block, &req, &mut out.0, &mut out.1).map(|()| out)
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
        let (model, block_size) = (&self.model, self.block_size);
        let (o, i, s) = py
            .detach(|| with_pool(num_threads, || recommend_batch(model, block_size, &b)))
            .map_err(to_py)?;
        Ok((o.into_pyarray(py), i.into_pyarray(py), s.into_pyarray(py)))
    }

    /// The scores of `items` for one history, aligned with `items`.
    #[pyo3(signature = (ids, weights, items))]
    fn score<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'_, i64>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        items: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Arr<'py, f32>> {
        let (ids, weights, items) = (ids.as_slice()?, slice(&weights)?, items.as_slice()?);
        let (model, block) = (&self.model, self.block);
        let out = py
            .detach(|| score(model, block, ids, weights, items))
            .map_err(to_py)?;
        Ok(out.into_pyarray(py))
    }

    /// One history and one item list per user, flat as `batch`'s; the scores
    /// flat, aligned with `item_ids`.
    #[pyo3(signature = (offsets, ids, weights, item_offsets, item_ids, *, num_threads))]
    #[allow(clippy::too_many_arguments)]
    fn score_batch<'py>(
        &self,
        py: Python<'py>,
        offsets: PyReadonlyArray1<'_, i64>,
        ids: PyReadonlyArray1<'_, i64>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        item_offsets: PyReadonlyArray1<'_, i64>,
        item_ids: PyReadonlyArray1<'_, i64>,
        num_threads: usize,
    ) -> PyResult<Arr<'py, f32>> {
        let histories = Lists {
            offsets: offsets.as_slice()?,
            ids: ids.as_slice()?,
        };
        let items = Lists {
            offsets: item_offsets.as_slice()?,
            ids: item_ids.as_slice()?,
        };
        let weights = slice(&weights)?;
        let (model, block_size) = (&self.model, self.block_size);
        let out = py
            .detach(|| {
                with_pool(num_threads, || {
                    score_batch(model, block_size, histories, weights, items)
                })
            })
            .map_err(to_py)?;
        Ok(out.into_pyarray(py))
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
        let p = self.model.parts();
        Ok((
            PyArray1::from_slice(py, p.indptr),
            PyArray1::from_iter(py, p.pairs.iter().map(|e| e.col)),
            PyArray1::from_iter(py, p.pairs.iter().map(|e| e.val)),
            PyArray1::from_slice(py, p.new_to_old),
            p.n_cols,
            p.ext.map(|e| PyArray1::from_slice(py, e)),
            p.allowed.map(|a| PyArray1::from_slice(py, a)),
            block_size_py(py, self.block_size)?,
        ))
    }
}
