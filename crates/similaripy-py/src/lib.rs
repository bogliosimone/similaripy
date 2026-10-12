//! Python bindings of the Rust core. During Step 2a this mirrors the C++
//! `compute_similarities_parallel` call, so `s_plus.pyx` can swap kernels
//! after the identical prep (`_backend='rust'`).

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use similaripy_core::{CsrPattern, CsrView, Error, Inputs, Norms, Output, Params, compute};

const SELECTION_MATRIX: i32 = 2;

fn to_py(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn selection<'a>(
    mode: i32,
    name: &'static str,
    indptr: &'a [i32],
    indices: &'a [i32],
    n_cols: usize,
) -> PyResult<Option<CsrPattern<'a>>> {
    // NONE and ARRAY (pre-filtered in Python) need no per-row check.
    if mode != SELECTION_MATRIX {
        return Ok(None);
    }
    CsrPattern::new(name, indptr, indices, n_cols)
        .map(Some)
        .map_err(to_py)
}

/// Same arguments as the C++ kernel, minus the progress bar. Writes into
/// the caller's zeroed `rows`/`cols`/`values`. Releases the GIL.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_similarities(
    py: Python<'_>,
    targets: PyReadonlyArray1<i32>,
    m1_data: PyReadonlyArray1<f32>,
    m1_indices: PyReadonlyArray1<i32>,
    m1_indptr: PyReadonlyArray1<i32>,
    m1_n_cols: usize,
    m2_data: PyReadonlyArray1<f32>,
    m2_indices: PyReadonlyArray1<i32>,
    m2_indptr: PyReadonlyArray1<i32>,
    x_tversky: PyReadonlyArray1<f32>,
    y_tversky: PyReadonlyArray1<f32>,
    x_cosine: PyReadonlyArray1<f32>,
    y_cosine: PyReadonlyArray1<f32>,
    x_depop: PyReadonlyArray1<f32>,
    y_depop: PyReadonlyArray1<f32>,
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
    filter_mode: i32,
    filter_indptr: PyReadonlyArray1<i32>,
    filter_indices: PyReadonlyArray1<i32>,
    target_mode: i32,
    target_indptr: PyReadonlyArray1<i32>,
    target_indices: PyReadonlyArray1<i32>,
    mut rows: PyReadwriteArray1<i32>,
    mut cols: PyReadwriteArray1<i32>,
    mut values: PyReadwriteArray1<f32>,
    num_threads: usize,
    block_size: usize,
) -> PyResult<()> {
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
            x_tversky: x_tversky.as_slice()?,
            y_tversky: y_tversky.as_slice()?,
            x_cosine: x_cosine.as_slice()?,
            y_cosine: y_cosine.as_slice()?,
            x_depop: x_depop.as_slice()?,
            y_depop: y_depop.as_slice()?,
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
        filter: selection(
            filter_mode,
            "filter_cols",
            filter_indptr.as_slice()?,
            filter_indices.as_slice()?,
            n_output_cols,
        )?,
        target_cols: selection(
            target_mode,
            "target_cols",
            target_indptr.as_slice()?,
            target_indices.as_slice()?,
            n_output_cols,
        )?,
    };
    let out = Output {
        rows: rows.as_slice_mut()?,
        cols: cols.as_slice_mut()?,
        values: values.as_slice_mut()?,
    };

    py.detach(|| {
        let run = || compute(&inp, k, block_size, out);
        if num_threads == 0 {
            run()
        } else {
            // Per-call pool for now; cached per calling thread in slice 1b (D20).
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| Error::InvalidInput(e.to_string()))?
                .install(run)
        }
    })
    .map_err(to_py)
}

#[pymodule]
fn _similaripy_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_similarities, m)?)
}
