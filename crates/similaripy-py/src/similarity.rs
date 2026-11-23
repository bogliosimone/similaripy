//! `_core.canonical` and `_core.similarity`: a caller's scipy arrays copied
//! once (`prep::canonical`), and the S-Plus driver on those copies
//! (`similaripy_core::similarity`), each in one GIL-free call.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use similaripy_core::Error;
use similaripy_core::prep::{self, Csr, Sparse};
use similaripy_core::similarity::{
    self as core, ColumnOrder, Options, Phase, Pop, Selector, Similarity, Walk,
};

use crate::progress::ProgressBar;
use crate::serve;
use crate::{ParamsArg, to_py, with_pool};

type Ro<'py, T> = PyReadonlyArray1<'py, T>;

#[derive(Clone, Copy)]
enum Format {
    Csr,
    Csc,
    Coo,
}

/// The (index, data) dtypes `prep::canonical` reads: (major, minor, data).
enum Arrays<'py> {
    I32F32(Ro<'py, i32>, Ro<'py, i32>, Ro<'py, f32>),
    I32F64(Ro<'py, i32>, Ro<'py, i32>, Ro<'py, f64>),
    I64F32(Ro<'py, i64>, Ro<'py, i64>, Ro<'py, f32>),
    I64F64(Ro<'py, i64>, Ro<'py, i64>, Ro<'py, f64>),
}

/// The same arrays as slices, for the GIL-free copy.
enum Data<'a> {
    I32F32(&'a [i32], &'a [i32], &'a [f32]),
    I32F64(&'a [i32], &'a [i32], &'a [f64]),
    I64F32(&'a [i64], &'a [i64], &'a [f32]),
    I64F64(&'a [i64], &'a [i64], &'a [f64]),
}

/// A caller's matrix as scipy stores it: `format` 'csr' | 'csc' | 'coo';
/// `major` = indptr (COO: rows), `minor` = indices (COO: cols), both int32 or
/// both int64; data float32 | float64; `canonical` = scipy's
/// `has_canonical_format` (a hint).
struct Input<'py> {
    format: Format,
    shape: (usize, usize),
    canonical: bool,
    arrays: Arrays<'py>,
}

/// `Input`'s fields read by attribute from `similaripy._s_plus.Sparse`.
#[derive(FromPyObject)]
pub(crate) struct SparseArg<'py> {
    format: String,
    shape: (usize, usize),
    major: Bound<'py, PyAny>,
    minor: Bound<'py, PyAny>,
    data: Bound<'py, PyAny>,
    canonical: bool,
}

impl<'py> Input<'py> {
    fn new(
        format: &str,
        shape: (usize, usize),
        major: &Bound<'py, PyAny>,
        minor: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
        canonical: bool,
    ) -> PyResult<Self> {
        let format = match format {
            "csr" => Format::Csr,
            "csc" => Format::Csc,
            "coo" => Format::Coo,
            f => {
                return Err(PyValueError::new_err(format!(
                    "canonical: format {f:?} is not 'csr', 'csc' or 'coo'"
                )));
            }
        };
        let i32_index = major.extract::<Ro<i32>>().is_ok();
        let f32_data = data.extract::<Ro<f32>>().is_ok();
        let (a, b, c) = (major, minor, data);
        let arrays = match (i32_index, f32_data) {
            (true, true) => Arrays::I32F32(a.extract()?, b.extract()?, c.extract()?),
            (true, false) => Arrays::I32F64(a.extract()?, b.extract()?, c.extract()?),
            (false, true) => Arrays::I64F32(a.extract()?, b.extract()?, c.extract()?),
            (false, false) => Arrays::I64F64(a.extract()?, b.extract()?, c.extract()?),
        };
        Ok(Self {
            format,
            shape,
            canonical,
            arrays,
        })
    }

    fn from_arg(m: &SparseArg<'py>) -> PyResult<Self> {
        Self::new(&m.format, m.shape, &m.major, &m.minor, &m.data, m.canonical)
    }

    fn slices(&self) -> PyResult<Slices<'_>> {
        let data = match &self.arrays {
            Arrays::I32F32(a, b, c) => Data::I32F32(a.as_slice()?, b.as_slice()?, c.as_slice()?),
            Arrays::I32F64(a, b, c) => Data::I32F64(a.as_slice()?, b.as_slice()?, c.as_slice()?),
            Arrays::I64F32(a, b, c) => Data::I64F32(a.as_slice()?, b.as_slice()?, c.as_slice()?),
            Arrays::I64F64(a, b, c) => Data::I64F64(a.as_slice()?, b.as_slice()?, c.as_slice()?),
        };
        Ok(Slices {
            format: self.format,
            shape: self.shape,
            canonical: self.canonical,
            data,
        })
    }
}

struct Slices<'a> {
    format: Format,
    shape: (usize, usize),
    canonical: bool,
    data: Data<'a>,
}

impl Slices<'_> {
    /// `prep::canonical`: the owned canonical CSR (f32 values).
    fn copy(&self) -> Result<Csr<f32>, Error> {
        match self.data {
            Data::I32F32(a, b, c) => self.copy_as(a, b, c),
            Data::I32F64(a, b, c) => self.copy_as(a, b, c),
            Data::I64F32(a, b, c) => self.copy_as(a, b, c),
            Data::I64F64(a, b, c) => self.copy_as(a, b, c),
        }
    }

    fn copy_as<I: Copy + Into<i64>, V: Copy + Into<f64>>(
        &self,
        major: &[I],
        minor: &[I],
        data: &[V],
    ) -> Result<Csr<f32>, Error> {
        let m = match self.format {
            Format::Csr => Sparse::Csr {
                indptr: major,
                indices: minor,
                data,
            },
            Format::Csc => Sparse::Csc {
                indptr: major,
                indices: minor,
                data,
            },
            Format::Coo => Sparse::Coo {
                rows: major,
                cols: minor,
                data,
            },
        };
        prep::canonical(m, self.shape, self.canonical)
    }
}

/// `prep::canonical`: a caller's matrix (see `Input`) copied once into an
/// owned canonical CSR. Returns (indptr int64, indices int32, data float32).
#[pyfunction]
#[pyo3(signature = (*, format, shape, major, minor, data, canonical, num_threads))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn canonical<'py>(
    py: Python<'py>,
    format: &str,
    shape: (usize, usize),
    major: &Bound<'py, PyAny>,
    minor: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    canonical: bool,
    num_threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let input = Input::new(format, shape, major, minor, data, canonical)?;
    let s = input.slices()?;
    let m = py
        .detach(|| with_pool(num_threads, || s.copy()))
        .map_err(to_py)?;
    Ok((
        m.indptr.into_pyarray(py),
        m.indices.into_pyarray(py),
        m.data.into_pyarray(py),
    ))
}

/// 'none', 'sum' or one float32 weight per row / column.
enum PopArg<'py> {
    None,
    Sum,
    Weights(Ro<'py, f32>),
}

impl<'py> PopArg<'py> {
    fn new(v: &Bound<'py, PyAny>) -> PyResult<Self> {
        match v.extract::<&str>() {
            Ok("none") => Ok(Self::None),
            Ok("sum") => Ok(Self::Sum),
            Ok(s) => Err(PyValueError::new_err(format!(
                "depop weights must be \"none\", \"sum\" or an array, got {s:?}"
            ))),
            Err(_) => Ok(Self::Weights(v.extract()?)),
        }
    }

    fn pop(&self) -> PyResult<Pop<'_>> {
        Ok(match self {
            Self::None => Pop::None,
            Self::Sum => Pop::Sum,
            Self::Weights(w) => Pop::Weights(w.as_slice()?),
        })
    }
}

/// A column selector: one int64 id list for every row, or a matrix (see
/// `Input`; copied canonical in the GIL-free block, its values then ignored).
enum SelArg<'py> {
    Ids(Ro<'py, i64>),
    Matrix(Input<'py>),
}

enum SelSlices<'a> {
    Ids(&'a [i64]),
    Matrix(Slices<'a>),
}

impl<'py> SelArg<'py> {
    fn new(v: Option<Bound<'py, PyAny>>) -> PyResult<Option<Self>> {
        let Some(v) = v else { return Ok(None) };
        if let Ok(ids) = v.extract::<Ro<i64>>() {
            return Ok(Some(Self::Ids(ids)));
        }
        Ok(Some(Self::Matrix(Input::from_arg(&v.extract()?)?)))
    }

    fn slices(&self) -> PyResult<SelSlices<'_>> {
        Ok(match self {
            Self::Ids(ids) => SelSlices::Ids(ids.as_slice()?),
            Self::Matrix(m) => SelSlices::Matrix(m.slices()?),
        })
    }
}

impl<'a> SelSlices<'a> {
    fn selector(&self) -> Result<Selector<'a>, Error> {
        Ok(match self {
            Self::Ids(ids) => Selector::Ids(ids),
            Self::Matrix(m) => Selector::Matrix(m.copy()?),
        })
    }
}

/// `similaripy_core::similarity` on the canonical copies of `m1` and `m2`
/// (each a `Sparse`, see `Input`; output columns = m2's): every copy and the
/// driver run with the GIL released, on the global rayon pool
/// (`num_threads = 0`) or the calling thread's pool. `pop1` / `pop2`: 'none',
/// 'sum' or float32 weights; `walk`: (alpha, popularity) or None; selectors:
/// int64 ids, a `Sparse` or None; `block_size`: 'auto', None or an int > 0;
/// `column_order`: 'popularity' | 'original'. `progress` ticks per row and
/// shows the core's phases. Returns (indptr int64, indices int32, data
/// float32, kernel seconds).
#[pyfunction]
#[pyo3(signature = (
    *, m1, m2, k, params, c1, c2, p1, p2, additive_shrink, pop1, pop2, binary, walk,
    target_rows, filter_cols, target_cols, block_size, column_order, num_threads, progress
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn similarity<'py>(
    py: Python<'py>,
    m1: SparseArg<'py>,
    m2: SparseArg<'py>,
    k: usize,
    params: ParamsArg,
    c1: f32,
    c2: f32,
    p1: f32,
    p2: f32,
    additive_shrink: f32,
    pop1: Bound<'py, PyAny>,
    pop2: Bound<'py, PyAny>,
    binary: bool,
    walk: Option<(f32, bool)>,
    target_rows: Option<Ro<'py, i64>>,
    filter_cols: Option<Bound<'py, PyAny>>,
    target_cols: Option<Bound<'py, PyAny>>,
    block_size: Bound<'py, PyAny>,
    column_order: &str,
    num_threads: usize,
    progress: Option<Bound<'py, ProgressBar>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    f64,
)> {
    let column_order = match column_order {
        "popularity" => ColumnOrder::Popularity,
        "original" => ColumnOrder::Original,
        o => {
            return Err(PyValueError::new_err(format!(
                "column_order must be 'popularity' or 'original', got {o:?}"
            )));
        }
    };
    let (m1, m2) = (Input::from_arg(&m1)?, Input::from_arg(&m2)?);
    let (s1, s2) = (m1.slices()?, m2.slices()?);
    let n_cols = s2.shape.1;
    let (pop1, pop2) = (PopArg::new(&pop1)?, PopArg::new(&pop2)?);
    let (filter, target) = (SelArg::new(filter_cols)?, SelArg::new(target_cols)?);
    let (filter, target) = (
        filter.as_ref().map(SelArg::slices).transpose()?,
        target.as_ref().map(SelArg::slices).transpose()?,
    );
    let target_rows = target_rows.as_ref().map(|t| t.as_slice()).transpose()?;
    let o = Options {
        k,
        params: params.into(),
        c1,
        c2,
        p1,
        p2,
        additive_shrink,
        pop1: pop1.pop()?,
        pop2: pop2.pop()?,
        binary,
        walk: walk.map(|(alpha, popularity)| Walk { alpha, popularity }),
        target_rows,
        filter_cols: None,
        target_cols: None,
        block_size: serve::block_size(&block_size)?,
        column_order,
    };
    let bar = progress.as_ref().map(|b| b.get());
    let out: Similarity = py
        .detach(|| {
            with_pool(num_threads, || {
                let (a, b) = (s1.copy()?, s2.copy()?);
                let o = Options {
                    filter_cols: filter.as_ref().map(SelSlices::selector).transpose()?,
                    target_cols: target.as_ref().map(SelSlices::selector).transpose()?,
                    ..o
                };
                // Separate instantiations: without a bar the kernel has no per-row branch.
                match bar {
                    Some(bar) => core::similarity(a, b, n_cols, o, &|| bar.tick(), |p| match p {
                        Phase::Prep => bar.phase("Preprocessing", "prep"),
                        Phase::Compute { rows } => bar.rows_phase("Computing", "compute", rows),
                    }),
                    None => core::similarity(a, b, n_cols, o, &|| {}, |_| {}),
                }
            })
        })
        .map_err(to_py)?;
    Ok((
        out.csr.indptr.into_pyarray(py),
        out.csr.indices.into_pyarray(py),
        out.csr.data.into_pyarray(py),
        out.kernel_seconds,
    ))
}
