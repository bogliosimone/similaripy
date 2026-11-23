//! Rows to compute (`target_rows`) and column selectors (`filter_cols`,
//! `target_cols`): id lists become one keep mask, applied by dropping m2's
//! other columns; matrices stay per-row patterns for the kernel.

use super::Selector;
use crate::Error;
use crate::csr::check_structure;
use crate::prep::Csr;

fn bad<T>(msg: String) -> Result<T, Error> {
    Err(Error::InvalidInput(msg))
}

/// Every id in [0, n).
fn ids_in_range(name: &str, ids: &[i64], n: usize) -> Result<(), Error> {
    let (Some(&min), Some(&max)) = (ids.iter().min(), ids.iter().max()) else {
        return Ok(());
    };
    if min < 0 || max >= n as i64 {
        return bad(format!(
            "{name} ids must be in [0, {n}), got min {min} and max {max}"
        ));
    }
    Ok(())
}

/// The m1 rows to compute, sorted and unique (output rows come out in row
/// order; a duplicate would duplicate its row's entries); None = every row.
pub(crate) fn targets(ids: Option<&[i64]>, n_rows: usize) -> Result<Vec<i32>, Error> {
    if n_rows > i32::MAX as usize + 1 {
        return bad("more than 2^31 rows in matrix1 (row ids are int32)".into());
    }
    let Some(ids) = ids else {
        return Ok((0..n_rows).map(|r| r as i32).collect());
    };
    ids_in_range("target_rows", ids, n_rows)?;
    let mut t: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    t.sort_unstable();
    t.dedup();
    Ok(t)
}

/// A selector checked against the output shape. An empty filter is none; a
/// `target_cols` that selects nothing (no ids, or a matrix with no entries)
/// is an error: None scores every column.
pub(crate) enum Sel<'a> {
    None,
    Ids(&'a [i64]),
    Matrix(Pattern),
}

/// A selector matrix's pattern (the values are dropped once canonical).
pub(crate) struct Pattern {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
}

impl<'a> Sel<'a> {
    pub fn new(
        name: &'static str,
        sel: Option<Selector<'a>>,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Self, Error> {
        let empty = || -> Result<Self, Error> {
            if name == "target_cols" {
                return bad("target_cols is empty: pass None to score all columns".into());
            }
            Ok(Sel::None)
        };
        match sel {
            None => Ok(Sel::None),
            Some(Selector::Ids(ids)) => {
                ids_in_range(name, ids, n_cols)?;
                if ids.is_empty() {
                    empty()
                } else {
                    Ok(Sel::Ids(ids))
                }
            }
            Some(Selector::Matrix(m)) => {
                check_structure(name, &m.indptr, &m.indices, n_cols)?;
                let rows = m.indptr.len() - 1;
                if rows != n_rows {
                    return bad(format!("{name} must have {n_rows} rows, got {rows}"));
                }
                if m.indices.is_empty() {
                    return empty();
                }
                let Csr {
                    indptr, indices, ..
                } = m; // values dropped
                Ok(Sel::Matrix(Pattern { indptr, indices }))
            }
        }
    }

    pub fn into_pattern(self) -> Option<Pattern> {
        match self {
            Sel::Matrix(p) => Some(p),
            _ => None,
        }
    }
}

/// The columns id-list selectors keep: the target list (or every column)
/// minus the filter list; None when neither is a list.
pub(crate) fn keep(filter: &Sel, target: &Sel, n_cols: usize) -> Option<Vec<bool>> {
    let (f, t) = (
        if let Sel::Ids(f) = filter {
            Some(f)
        } else {
            None
        },
        if let Sel::Ids(t) = target {
            Some(t)
        } else {
            None
        },
    );
    if f.is_none() && t.is_none() {
        return None;
    }
    let mut keep = vec![t.is_none(); n_cols];
    for &c in t.into_iter().flat_map(|t| t.iter()) {
        keep[c as usize] = true;
    }
    for &c in f.into_iter().flat_map(|f| f.iter()) {
        keep[c as usize] = false;
    }
    Some(keep)
}

/// Drops the entries of columns not kept, in place (column ids and row order
/// unchanged; the arrays shrink to the new nnz).
// ponytail: one serial pass over nnz (cold); fuse with the layout's remap if a profile asks
pub(crate) fn drop_columns(m: &mut Csr<f32>, keep: &[bool]) {
    let mut out = 0;
    let mut start = 0;
    for r in 0..m.indptr.len() - 1 {
        let end = m.indptr[r + 1] as usize;
        for j in start..end {
            if keep[m.indices[j] as usize] {
                m.indices[out] = m.indices[j];
                m.data[out] = m.data[j];
                out += 1;
            }
        }
        start = end;
        m.indptr[r + 1] = out as i64;
    }
    m.indices.truncate(out);
    m.data.truncate(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(indptr: Vec<i64>, indices: Vec<i32>) -> Csr<f32> {
        let data = vec![1.0; indices.len()];
        Csr {
            indptr,
            indices,
            data,
        }
    }

    fn err<T>(r: Result<T, Error>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn targets_sorted_unique_in_range() {
        assert_eq!(targets(None, 3).unwrap(), [0, 1, 2]);
        assert_eq!(targets(Some(&[2, 0, 2, 1, 0]), 3).unwrap(), [0, 1, 2]);
        assert_eq!(targets(Some(&[]), 3).unwrap(), Vec::<i32>::new());
        assert!(
            err(targets(Some(&[0, 3]), 3))
                .contains("target_rows ids must be in [0, 3), got min 0 and max 3")
        );
        assert!(err(targets(Some(&[-1]), 3)).contains("got min -1"));
        assert!(
            err(targets(Some(&[1 << 32]), 3)).contains("must be in"),
            "no i32 wrap"
        );
    }

    #[test]
    fn selectors_checked_and_empty_rules() {
        let new = |name, s| Sel::new(name, Some(s), 2, 4);
        assert!(matches!(
            new("filter_cols", Selector::Ids(&[])).unwrap(),
            Sel::None
        ));
        assert!(err(new("target_cols", Selector::Ids(&[]))).contains("target_cols is empty"));
        assert!(
            err(new("filter_cols", Selector::Ids(&[4])))
                .contains("filter_cols ids must be in [0, 4)")
        );
        let empty = || Selector::Matrix(csr(vec![0, 0, 0], vec![]));
        assert!(matches!(new("filter_cols", empty()).unwrap(), Sel::None));
        assert!(err(new("target_cols", empty())).contains("target_cols is empty"));
        let short = Selector::Matrix(csr(vec![0, 1], vec![0]));
        assert!(err(new("target_cols", short)).contains("must have 2 rows, got 1"));
        let wide = Selector::Matrix(csr(vec![0, 1, 1], vec![4]));
        assert!(err(new("filter_cols", wide)).contains("column index out of range"));
        let ok = new(
            "target_cols",
            Selector::Matrix(csr(vec![0, 1, 2], vec![3, 0])),
        )
        .unwrap();
        let p = ok.into_pattern().unwrap();
        assert_eq!((p.indptr, p.indices), (vec![0, 1, 2], vec![3, 0]));
    }

    #[test]
    fn keep_mask_and_drop() {
        let none = Sel::None;
        assert!(keep(&none, &none, 4).is_none());
        let pattern = Sel::Matrix(Pattern {
            indptr: vec![0],
            indices: vec![],
        });
        assert!(keep(&pattern, &none, 4).is_none(), "matrices stay patterns");
        let k = keep(&Sel::Ids(&[1, 1]), &none, 4).unwrap();
        assert_eq!(k, [true, false, true, true]);
        let k = keep(&Sel::Ids(&[1]), &Sel::Ids(&[0, 1, 3]), 4).unwrap();
        assert_eq!(k, [true, false, false, true]);

        // rows [0 1 3], [], [1 2]
        let mut m = Csr {
            indptr: vec![0, 3, 3, 5],
            indices: vec![0, 1, 3, 1, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        drop_columns(&mut m, &k);
        assert_eq!(m.indptr, [0, 2, 2, 2]);
        assert_eq!((m.indices, m.data), (vec![0, 3], vec![1.0, 3.0]));
    }
}
