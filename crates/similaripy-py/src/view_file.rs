//! View file bindings: `similaripy.serving.write_view` / `read_view` over
//! `similaripy_core::serve::view_file`, the format the server's views folder
//! reads (stable in 1.0). Python only checks the ids' dtype; the name rule,
//! the atomic write and the parse are the core's.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use similaripy_core::Error;
use similaripy_core::serve::view_file::{META_SUFFIX, Rule, read_ids, valid_name, write_ids};

use crate::to_py;

type Ids<'py> = Option<PyReadonlyArray1<'py, i64>>;
type Out<'py> = Option<Bound<'py, PyArray1<i64>>>;

fn slice<'a>(a: &'a Ids<'_>) -> PyResult<Option<&'a [i64]>> {
    Ok(a.as_ref().map(|a| a.as_slice()).transpose()?)
}

fn check_name(name: &str) -> PyResult<()> {
    if valid_name(name) {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "view name {name:?}: 1-64 of a-z 0-9 _ -"
    )))
}

/// Writes the given rules' files (each replaced atomically), in order
/// include, exclude; returns their paths.
#[pyfunction]
#[pyo3(signature = (dir, name, include, exclude))]
pub(crate) fn write_view(
    py: Python<'_>,
    dir: PathBuf,
    name: &str,
    include: Ids<'_>,
    exclude: Ids<'_>,
) -> PyResult<Vec<PathBuf>> {
    check_name(name)?;
    if include.is_none() && exclude.is_none() {
        return Err(PyValueError::new_err(
            "write_view needs include, exclude or both",
        ));
    }
    let pairs = [
        (Rule::Include, slice(&include)?),
        (Rule::Exclude, slice(&exclude)?),
    ];
    py.detach(|| {
        pairs
            .iter()
            .filter_map(|&(r, ids)| Some(write_ids(&dir, name, r, ids?)))
            .collect::<Result<Vec<_>, Error>>()
    })
    .map_err(to_py)
}

/// One rule's ids, None when its file doesn't exist.
fn read_rule(dir: &Path, name: &str, rule: Rule) -> Result<Option<Vec<i64>>, Error> {
    match read_ids(&dir.join(format!("{name}{}", rule.suffix()))) {
        Err(Error::Io(ErrorKind::NotFound, _)) => Ok(None),
        // A bad line: ValueError, as a corrupted saved model's
        Err(Error::Io(ErrorKind::InvalidData, e)) => Err(Error::InvalidInput(e)),
        r => r.map(Some),
    }
}

/// A view's (include, exclude) ids as the server reads them; None = no such
/// file. No file at all: FileNotFoundError; a `NAME.meta.json`: ValueError
/// (a later server's settings, which a 1.0 server doesn't serve either).
#[pyfunction]
pub(crate) fn read_view<'py>(
    py: Python<'py>,
    dir: PathBuf,
    name: &str,
) -> PyResult<(Out<'py>, Out<'py>)> {
    check_name(name)?;
    let meta = dir.join(format!("{name}{META_SUFFIX}"));
    if meta.exists() {
        return Err(PyValueError::new_err(format!(
            "{}: view settings of a later version (not served by a 1.0 server)",
            meta.display()
        )));
    }
    let [inc, exc] = py
        .detach(|| {
            let [i, e] = [Rule::Include, Rule::Exclude].map(|r| read_rule(&dir, name, r));
            Ok::<_, Error>([i?, e?])
        })
        .map_err(to_py)?;
    if inc.is_none() && exc.is_none() {
        let e = format!("no view {name:?} in {}", dir.display());
        return Err(to_py(Error::Io(ErrorKind::NotFound, e)));
    }
    Ok((
        inc.map(|v| v.into_pyarray(py)),
        exc.map(|v| v.into_pyarray(py)),
    ))
}
