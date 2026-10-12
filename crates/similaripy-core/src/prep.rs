//! Input prep for the blocked kernel: column renumbering + per-row sort.

use crate::Error;

/// Entries per sequential leaf; rows of ~100 entries sort in L1.
const LEAF_NNZ: usize = 1 << 16;

/// Renumbers columns (`c -> forward[c]`, skipped when `forward` is empty) and
/// sorts every row by column, carrying `data` along, in place and in parallel
/// on the current rayon pool. Already-sorted rows are left as they are.
///
/// Replaces numpy's gather + scipy's single-threaded `sort_indices`: 2.1 s of
/// the 3.3 s recommendation call on Yambda 50M (S with 93M entries, Step 3).
pub fn remap_sort_rows(
    indptr: &[i32],
    indices: &mut [i32],
    data: &mut [f32],
    forward: &[i32],
) -> Result<(), Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("remap_sort_rows: {m}")));
    if indptr.first() != Some(&0) || indptr.windows(2).any(|w| w[0] > w[1]) {
        return bad("indptr not monotone from 0");
    }
    if *indptr.last().unwrap() as usize != indices.len() || data.len() != indices.len() {
        return bad("indptr[-1], len(indices) and len(data) differ");
    }
    if !forward.is_empty()
        && indices
            .iter()
            .any(|&c| c < 0 || c as usize >= forward.len())
    {
        return bad("column index out of range of forward");
    }
    rows_from(indptr, indices, data, forward, LEAF_NNZ);
    Ok(())
}

/// Splits rows by entries (as `normalize::par_rows`), so skewed rows balance.
fn rows_from(indptr: &[i32], indices: &mut [i32], data: &mut [f32], forward: &[i32], leaf: usize) {
    let n = indptr.len() - 1;
    let base = indptr[0] as usize;
    if n > 1 && indices.len() > leaf {
        let half = (base + indices.len() / 2) as i32;
        let mid = indptr.partition_point(|&p| p <= half).clamp(1, n - 1);
        let at = indptr[mid] as usize - base;
        let (il, ir) = indices.split_at_mut(at);
        let (dl, dr) = data.split_at_mut(at);
        rayon::join(
            || rows_from(&indptr[..=mid], il, dl, forward, leaf),
            || rows_from(&indptr[mid..], ir, dr, forward, leaf),
        );
        return;
    }
    if !forward.is_empty() {
        for c in indices.iter_mut() {
            *c = forward[*c as usize];
        }
    }
    let mut pairs: Vec<(i32, f32)> = Vec::new();
    for w in indptr.windows(2) {
        let (a, z) = (w[0] as usize - base, w[1] as usize - base);
        let (cols, vals) = (&mut indices[a..z], &mut data[a..z]);
        if cols.is_sorted() {
            continue;
        }
        pairs.clear();
        pairs.extend(cols.iter().copied().zip(vals.iter().copied()));
        pairs.sort_unstable_by_key(|p| p.0);
        for ((c, v), (pc, pv)) in cols.iter_mut().zip(vals.iter_mut()).zip(&pairs) {
            (*c, *v) = (*pc, *pv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // Same rows as a remap + sort per row, each value still on its column, however it splits.
        #[test]
        fn matches_per_row_sort(
            rows in proptest::collection::vec(proptest::collection::vec(0i32..20, 0..8), 1..30),
            remap in any::<bool>(),
            leaf in 0usize..8,
        ) {
            let forward: Vec<i32> = if remap { (0..20).rev().collect() } else { vec![] };
            let mut indptr = vec![0];
            let mut indices = vec![];
            for r in &rows {
                indices.extend(r);
                indptr.push(indices.len() as i32);
            }
            // value = original column, so pairs can be checked after the move
            let mut data: Vec<f32> = indices.iter().map(|&c| c as f32).collect();
            // small leaves exercise the parallel split
            rows_from(&indptr, &mut indices, &mut data, &forward, leaf);
            for (i, r) in rows.iter().enumerate() {
                let (a, z) = (indptr[i] as usize, indptr[i + 1] as usize);
                let mut want: Vec<i32> = r.iter().map(|&c| if remap { forward[c as usize] } else { c }).collect();
                want.sort();
                prop_assert_eq!(&indices[a..z], &want[..]);
                for (&c, &v) in indices[a..z].iter().zip(&data[a..z]) {
                    let orig = if remap { forward[c as usize] } else { c }; // forward is its own inverse
                    prop_assert_eq!(v, orig as f32);
                }
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let (mut i, mut d) = (vec![0, 5], vec![1.0, 2.0]);
        assert!(remap_sort_rows(&[0, 2], &mut i, &mut d, &[0, 1]).is_err()); // 5 >= len(forward)
        assert!(remap_sort_rows(&[0, 1], &mut i, &mut d, &[]).is_err()); // indptr[-1] != nnz
        assert!(remap_sort_rows(&[0, 2], &mut i, &mut d[..1], &[]).is_err());
    }
}
