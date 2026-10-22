//! Serving bindings (step 4.1 exploration): `ServeModel` owns the core `Model`
//! (Rust buffers, one copy at build, no numpy inside, D63); each call copies
//! the ids in, runs pure Rust without the GIL and moves the results out.
//! `ServeVariant` and the `bench*` methods exist only until the variants are
//! locked (step 4.2).

use std::hint::black_box;
use std::time::Instant;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use similaripy_core::serve::{
    ItemMask, Lookup, Mask, Model, Request, Seen, Variant, recommend, recommend_batch,
};

use crate::{to_py, with_pool};

type Recs<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>);

/// One candidate implementation per stage (exploration only).
#[pyclass(frozen, name = "ServeVariant", module = "similaripy._core")]
pub struct PyVariant(Variant);

#[pymethods]
impl PyVariant {
    #[new]
    #[pyo3(signature = (*, lookup="dense", seen="neginf", mask="after", block=0, canonicalize=true, reuse_scratch=true))]
    fn new(
        lookup: &str,
        seen: &str,
        mask: &str,
        block: usize,
        canonicalize: bool,
        reuse_scratch: bool,
    ) -> PyResult<Self> {
        let bad = |what: &str, v: &str| PyValueError::new_err(format!("unknown {what}: {v:?}"));
        Ok(Self(Variant {
            lookup: match lookup {
                "dense" => Lookup::Dense,
                "binary" => Lookup::Binary,
                "eytzinger" => Lookup::Eytzinger,
                "hash" => Lookup::Hash,
                "hash_batched" => Lookup::HashBatched,
                _ => return Err(bad("lookup", lookup)),
            },
            seen: match seen {
                "neginf" => Seen::NegInf,
                "bitset" => Seen::Bitset,
                "later" => Seen::Later,
                _ => return Err(bad("seen", seen)),
            },
            mask: match mask {
                "after" => Mask::AfterFloor,
                "before" => Mask::BeforeFloor,
                "later" => Mask::Later,
                _ => return Err(bad("mask", mask)),
            },
            block,
            canonicalize,
            reuse_scratch,
        }))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A per-call availability mask, bound to the model that built it.
#[pyclass(frozen, name = "ServeMask", module = "similaripy._core")]
pub struct PyMask(ItemMask);

#[pyclass(frozen, name = "ServeModel", module = "similaripy._core")]
pub struct PyModel(Model);

/// A batch of histories as CSR, read from a dict.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct History<'py> {
    indptr: PyReadonlyArray1<'py, i64>,
    ids: PyReadonlyArray1<'py, i64>,
    weights: Option<PyReadonlyArray1<'py, f32>>,
}

#[pymethods]
impl PyModel {
    /// S (n × n CSR, internal ids, rows sorted), `new_to_old` (internal id →
    /// S's original column) and `ext` (external id per original column).
    #[new]
    #[pyo3(signature = (*, indptr, indices, data, new_to_old, ext))]
    fn new(
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i32>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        new_to_old: PyReadonlyArray1<'_, i32>,
        ext: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Self> {
        let (indptr, indices, data) = (
            indptr.as_slice()?.to_vec(),
            indices.as_slice()?.to_vec(),
            data.as_slice()?.to_vec(),
        );
        let (new_to_old, ext) = (new_to_old.as_slice()?, ext.as_slice()?);
        py.detach(|| Model::new(indptr, indices, data, new_to_old, ext))
            .map(Self)
            .map_err(to_py)
    }

    #[getter]
    fn n_items(&self) -> usize {
        self.0.n_items()
    }

    #[getter]
    fn n_active(&self) -> usize {
        self.0.n_active()
    }

    fn mask(&self, ids: PyReadonlyArray1<'_, i64>, variant: &PyVariant) -> PyResult<PyMask> {
        Ok(PyMask(self.0.mask(ids.as_slice()?, variant.0.lookup)))
    }

    #[pyo3(signature = (ids, weights, *, k, exclude_seen, allowed, variant))]
    #[allow(clippy::too_many_arguments)]
    fn recommend<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'_, i64>,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        k: usize,
        exclude_seen: bool,
        allowed: Option<&PyMask>,
        variant: &PyVariant,
    ) -> PyResult<Recs<'py>> {
        let ids = ids.as_slice()?.to_vec();
        let weights = weights
            .map(|w| w.as_slice().map(<[f32]>::to_vec))
            .transpose()?;
        let (model, v) = (&self.0, &variant.0);
        let allowed = allowed.map(|a| &a.0);
        let (out_ids, out_scores) = py
            .detach(|| {
                let req = Request {
                    ids: &ids,
                    weights: weights.as_deref(),
                    k,
                    exclude_seen,
                    allowed,
                };
                let cap = k.min(model.n_active());
                let mut out = (Vec::with_capacity(cap), Vec::with_capacity(cap));
                recommend(model, &req, v, &mut out.0, &mut out.1).map(|()| out)
            })
            .map_err(to_py)?;
        Ok((out_ids.into_pyarray(py), out_scores.into_pyarray(py)))
    }

    /// Batch (design C): `recommend` per history row on the rayon pool.
    /// Returns (indptr, ids, scores, Rust seconds).
    #[pyo3(signature = (history, *, k, exclude_seen, variant, num_threads))]
    #[allow(clippy::type_complexity)]
    fn batch<'py>(
        &self,
        py: Python<'py>,
        history: History<'_>,
        k: usize,
        exclude_seen: bool,
        variant: &PyVariant,
        num_threads: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f32>>,
        f64,
    )> {
        let (indptr, ids) = (history.indptr.as_slice()?, history.ids.as_slice()?);
        let weights = history.weights.as_ref().map(|w| w.as_slice()).transpose()?;
        let (model, v) = (&self.0, &variant.0);
        let ((p, i, s), secs) = py
            .detach(|| {
                let t = Instant::now();
                let r = with_pool(num_threads, || {
                    recommend_batch(model, indptr, ids, weights, k, exclude_seen, v)
                });
                r.map(|r| (r, t.elapsed().as_secs_f64()))
            })
            .map_err(to_py)?;
        Ok((
            p.into_pyarray(py),
            i.into_pyarray(py),
            s.into_pyarray(py),
            secs,
        ))
    }

    /// Rust-side ns per request, each history row in turn on the calling
    /// thread. `cold`: reads `evict_mb` MB between requests (caches cold-ish).
    #[pyo3(signature = (history, *, k, exclude_seen, allowed, variant, cold, evict_mb=48))]
    #[allow(clippy::too_many_arguments)]
    fn bench<'py>(
        &self,
        py: Python<'py>,
        history: History<'_>,
        k: usize,
        exclude_seen: bool,
        allowed: Option<&PyMask>,
        variant: &PyVariant,
        cold: bool,
        evict_mb: usize,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let (indptr, ids) = (history.indptr.as_slice()?, history.ids.as_slice()?);
        let weights = history.weights.as_ref().map(|w| w.as_slice()).transpose()?;
        let (model, v) = (&self.0, &variant.0);
        let allowed = allowed.map(|a| &a.0);
        let ns = py
            .detach(|| {
                let evict = if cold {
                    vec![1u8; evict_mb << 20]
                } else {
                    vec![]
                };
                let mut out = (Vec::new(), Vec::new());
                let mut ns = Vec::with_capacity(indptr.len().saturating_sub(1));
                for w in indptr.windows(2) {
                    let (a, b) = (w[0] as usize, w[1] as usize);
                    if cold {
                        black_box(evict.iter().step_by(64).fold(0u8, |x, &y| x ^ y));
                    }
                    let req = Request {
                        ids: &ids[a..b],
                        weights: weights.map(|w| &w[a..b]),
                        k,
                        exclude_seen,
                        allowed,
                    };
                    out.0.clear();
                    out.1.clear();
                    let t = Instant::now();
                    recommend(model, &req, v, &mut out.0, &mut out.1)?;
                    ns.push(t.elapsed().as_nanos() as u64);
                    black_box(&out);
                }
                Ok(ns)
            })
            .map_err(to_py)?;
        Ok(ns.into_pyarray(py))
    }

    /// Rust-side ns for looking up `ids` `repeat` times (V1 in isolation).
    fn bench_lookup(
        &self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, i64>,
        variant: &PyVariant,
        repeat: usize,
    ) -> PyResult<u64> {
        let ids = ids.as_slice()?;
        let (model, how) = (&self.0, variant.0.lookup);
        Ok(py.detach(|| {
            let (mut out, mut probe) = (Vec::with_capacity(ids.len()), Vec::new());
            let t = Instant::now();
            for _ in 0..repeat {
                out.clear();
                model.lookup_into(black_box(ids), None, how, &mut out, &mut probe);
                black_box(&out);
            }
            t.elapsed().as_nanos() as u64
        }))
    }
}
