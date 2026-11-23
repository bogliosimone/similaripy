//! Kernel column order. `Popularity`: m2's columns renumbered by descending
//! count (after list selectors dropped theirs), on every call, blocked or
//! not: popular columns share the first, cache-hot blocks, and the kernel's
//! tie rule (lower column wins) means the same on every path: ties go to the
//! more popular column, then to the lower original id (stable sort). The
//! kernel writes original ids back (`col_map`). Canonical rows are sorted, as
//! blocking needs; a renumbering re-sorts them.

use super::ColumnOrder;
use super::norms::Owned;
use super::select::Pattern;
use crate::Error;
use crate::prep::{Csr, remap_sort_rows};

/// Renumbers m2's columns, the per-column norms and the selector patterns;
/// returns the kernel's `col_map` (new -> original id), None for `Original`.
pub(crate) fn layout(
    m2: &mut Csr<f32>,
    norms: &mut Owned,
    patterns: &mut [Option<Pattern>],
    n_cols: usize,
    order: ColumnOrder,
) -> Result<Option<Vec<i32>>, Error> {
    if order == ColumnOrder::Original {
        return Ok(None);
    }
    let mut counts = vec![0u32; n_cols]; // < 2^31: one per m2 row at most
    m2.indices.iter().for_each(|&c| counts[c as usize] += 1);
    let mut new_to_old: Vec<i32> = (0..n_cols as i32).collect();
    new_to_old.sort_by_key(|&c| std::cmp::Reverse(counts[c as usize])); // stable
    let mut old_to_new = vec![0i32; n_cols];
    for (new, &old) in new_to_old.iter().enumerate() {
        old_to_new[old as usize] = new as i32;
    }
    norms.permute_y(&new_to_old);
    for p in patterns.iter_mut().flatten() {
        remap_sort_rows(&p.indptr, &mut p.indices, None, &old_to_new)?;
    }
    remap_sort_rows(&m2.indptr, &mut m2.indices, Some(&mut m2.data), &old_to_new)?;
    Ok(Some(new_to_old))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popularity_order_stable_and_remapped() {
        // column counts: 0 -> 1, 1 -> 2, 2 -> 0, 3 -> 2
        let mut m2 = Csr {
            indptr: vec![0, 3, 5],
            indices: vec![0, 1, 3, 1, 3],
            data: vec![10.0, 11.0, 13.0, 21.0, 23.0],
        };
        let mut norms = Owned {
            y_cosine: Some(vec![0.0, 1.0, 2.0, 3.0]),
            ..Owned::default()
        };
        let mut patterns = [
            Some(Pattern {
                indptr: vec![0, 2],
                indices: vec![0, 2],
            }),
            None,
        ];
        let map = layout(
            &mut m2,
            &mut norms,
            &mut patterns,
            4,
            ColumnOrder::Popularity,
        )
        .unwrap();
        // ties (1, 3) by lower id; 0 before the empty 2
        assert_eq!(map.unwrap(), [1, 3, 0, 2]);
        assert_eq!(norms.y_cosine.unwrap(), [1.0, 3.0, 0.0, 2.0]);
        // rows re-sorted by new id, values carried
        assert_eq!(m2.indices, [0, 1, 2, 0, 1]);
        assert_eq!(m2.data, [11.0, 13.0, 10.0, 21.0, 23.0]);
        assert_eq!(patterns[0].as_ref().unwrap().indices, [2, 3]);
    }

    #[test]
    fn original_order_changes_nothing() {
        let mut m2 = Csr {
            indptr: vec![0, 1],
            indices: vec![1],
            data: vec![1.0],
        };
        let mut norms = Owned::default();
        let map = layout(
            &mut m2,
            &mut norms,
            &mut [None, None],
            2,
            ColumnOrder::Original,
        )
        .unwrap();
        assert!(map.is_none());
        assert_eq!(m2.indices, [1]);
    }
}
