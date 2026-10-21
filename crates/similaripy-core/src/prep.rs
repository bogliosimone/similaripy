//! Input prep: canonical rows (sorted, duplicates summed) and the blocked
//! kernel's column renumbering + per-row sort.

use crate::Error;
use crate::csr::check_indptr;
use crate::normalize::Value;

/// Entries per sequential leaf; rows of ~100 entries sort in L1.
const LEAF_NNZ: usize = 1 << 16;

/// Renumbers columns (`c -> forward[c]`) and sorts every row by column,
/// carrying `data` along (None: the pattern only), in place and in parallel
/// on the current rayon pool. Already-sorted rows are left as they are.
///
/// Replaces numpy's gather + scipy's single-threaded `sort_indices`: 2.1 s of
/// the 3.3 s recommendation call on Yambda 50M (S with 93M entries).
pub fn remap_sort_rows(
    indptr: &[i32],
    indices: &mut [i32],
    data: Option<&mut [f32]>,
    forward: &[i32],
) -> Result<(), Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("remap_sort_rows: {m}")));
    check_indptr("remap_sort_rows", indptr, indices.len())?;
    if data.as_ref().is_some_and(|d| d.len() != indices.len()) {
        return bad("len(data) != len(indices)");
    }
    if indices
        .iter()
        .any(|&c| c < 0 || c as usize >= forward.len())
    {
        return bad("column index out of range of forward");
    }
    match data {
        Some(data) => remap_sort(indptr, indices, data, forward),
        // zero-sized values: no allocation
        None => remap_sort(indptr, indices, &mut vec![(); indices.len()], forward),
    }
    Ok(())
}

fn remap_sort<V: Copy + Send>(
    indptr: &[i32],
    indices: &mut [i32],
    data: &mut [V],
    forward: &[i32],
) {
    par_leaves(indptr, indices, data, LEAF_NNZ, &|ip, cols, vals| {
        for c in cols.iter_mut() {
            *c = forward[*c as usize];
        }
        sort_rows(ip, cols, vals);
    });
}

/// Makes every row canonical, as scipy's `sum_duplicates` + `eliminate_zeros`:
/// sorted by column, duplicate columns summed into one entry, zeros (explicit,
/// or sums that cancel) dropped; float32 or float64 values. Rows are processed in parallel, then compacted
/// to the front of `indices`/`data` in one sequential pass; returns the new
/// indptr (the first `new_indptr[-1]` entries are the result).
///
/// Replaces scipy's single-threaded COO -> CSR (2.2 s on a 93M-entry COO
/// item model) and sums duplicate entries of CSR inputs, which would otherwise
/// reach the kernel unsummed (wrong norms and binary counts).
pub fn canonicalize_rows<V: Value>(
    indptr: &[i32],
    indices: &mut [i32],
    data: &mut [V],
) -> Result<Vec<i32>, Error> {
    canonicalize_with_leaf(indptr, indices, data, LEAF_NNZ)
}

fn canonicalize_with_leaf<V: Value>(
    indptr: &[i32],
    indices: &mut [i32],
    data: &mut [V],
    leaf: usize,
) -> Result<Vec<i32>, Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("canonicalize_rows: {m}")));
    check_indptr("canonicalize_rows", indptr, indices.len())?;
    if data.len() != indices.len() {
        return bad("len(data) != len(indices)");
    }
    if indices.iter().any(|&c| c < 0) {
        return bad("negative column index");
    }
    // Each row keeps its result at its front; the rest is marked with column -1.
    par_leaves(indptr, indices, data, leaf, &|ip, cols, vals| {
        sort_rows(ip, cols, vals);
        let base = ip[0] as usize;
        for w in ip.windows(2) {
            let (a, z) = (w[0] as usize - base, w[1] as usize - base);
            let (cols, vals) = (&mut cols[a..z], &mut vals[a..z]);
            let mut n = 0; // entries kept so far, sums still open
            for j in 0..cols.len() {
                if n > 0 && cols[n - 1] == cols[j] {
                    vals[n - 1] = vals[n - 1] + vals[j];
                } else {
                    (cols[n], vals[n]) = (cols[j], vals[j]);
                    n += 1;
                }
            }
            let mut kept = 0;
            for j in 0..n {
                if vals[j] != V::ZERO {
                    (cols[kept], vals[kept]) = (cols[j], vals[j]);
                    kept += 1;
                }
            }
            cols[kept..].fill(-1);
        }
    });
    let mut new_indptr = Vec::with_capacity(indptr.len());
    new_indptr.push(0);
    let mut out = 0;
    for w in indptr.windows(2) {
        let (a, z) = (w[0] as usize, w[1] as usize);
        let kept = indices[a..z].iter().position(|&c| c < 0).unwrap_or(z - a);
        indices.copy_within(a..a + kept, out);
        data.copy_within(a..a + kept, out);
        out += kept;
        new_indptr.push(out as i32);
    }
    Ok(new_indptr)
}

/// Sorts every row of a leaf by column, carrying the values; sorted rows are skipped.
fn sort_rows<V: Copy>(indptr: &[i32], indices: &mut [i32], data: &mut [V]) {
    let base = indptr[0] as usize;
    let mut pairs: Vec<(i32, V)> = Vec::new();
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

/// Runs `f(indptr, indices, data)` on leaves of whole rows, in parallel. Splits
/// by entries (as `normalize::par_rows`), so skewed rows balance; a leaf's
/// `indptr` slice is absolute, its `indices`/`data` start at `indptr[0]`.
fn par_leaves<V: Send, F>(indptr: &[i32], indices: &mut [i32], data: &mut [V], leaf: usize, f: &F)
where
    F: Fn(&[i32], &mut [i32], &mut [V]) + Sync,
{
    let n = indptr.len() - 1;
    let base = indptr[0] as usize;
    if n > 1 && indices.len() > leaf {
        let half = (base + indices.len() / 2) as i32;
        let mid = indptr.partition_point(|&p| p <= half).clamp(1, n - 1);
        let at = indptr[mid] as usize - base;
        let (il, ir) = indices.split_at_mut(at);
        let (dl, dr) = data.split_at_mut(at);
        rayon::join(
            || par_leaves(&indptr[..=mid], il, dl, leaf, f),
            || par_leaves(&indptr[mid..], ir, dr, leaf, f),
        );
    } else {
        f(indptr, indices, data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // Miri runs ~1000x slower: 8 cases (256 made the Miri job 80-105 min).
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
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
            // the public entry, pattern only (zero-sized values), as reference for the columns
            let mut pattern = indices.clone();
            let identity: Vec<i32> = (0..20).collect();
            remap_sort_rows(&indptr, &mut pattern, None, if remap { &forward } else { &identity }).unwrap();
            // small leaves exercise the parallel split
            par_leaves(&indptr, &mut indices, &mut data, leaf, &|ip, c, d| {
                if remap {
                    for x in c.iter_mut() {
                        *x = forward[*x as usize];
                    }
                }
                sort_rows(ip, c, d);
            });
            for (i, r) in rows.iter().enumerate() {
                let (a, z) = (indptr[i] as usize, indptr[i + 1] as usize);
                let mut want: Vec<i32> = r.iter().map(|&c| if remap { forward[c as usize] } else { c }).collect();
                want.sort();
                prop_assert_eq!(&indices[a..z], &want[..]);
                prop_assert_eq!(&pattern[a..z], &want[..]);
                for (&c, &v) in indices[a..z].iter().zip(&data[a..z]) {
                    let orig = if remap { forward[c as usize] } else { c }; // forward is its own inverse
                    prop_assert_eq!(v, orig as f32);
                }
            }
        }
    }

    proptest! {
        // Miri runs ~1000x slower: 8 cases (256 made the Miri job 80-105 min).
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 256 }))]
        // Same as a dense per-row sum: every column once, sorted, non-zero sums only.
        #[test]
        fn canonicalize_matches_dense_sum(
            rows in proptest::collection::vec(
                proptest::collection::vec((0i32..6, prop_oneof![Just(0.0f32), Just(1.0), Just(-1.0), Just(2.5)]), 0..10),
                1..if cfg!(miri) { 10 } else { 30 }, // Miri: each rayon split is costly
            ),
            leaf in 0usize..8,
        ) {
            let mut indptr = vec![0];
            let (mut indices, mut data) = (vec![], vec![]);
            for r in &rows {
                for &(c, v) in r {
                    indices.push(c);
                    data.push(v);
                }
                indptr.push(indices.len() as i32);
            }
            // float64 too: the values are exact in both types
            let (mut indices64, mut data64): (Vec<i32>, Vec<f64>) = (indices.clone(), data.iter().map(|&v| v as f64).collect());
            let new_indptr = canonicalize_with_leaf(&indptr, &mut indices, &mut data, leaf).unwrap();
            prop_assert_eq!(&canonicalize_with_leaf(&indptr, &mut indices64, &mut data64, leaf).unwrap(), &new_indptr);
            let nnz = new_indptr[new_indptr.len() - 1] as usize;
            prop_assert_eq!(&indices64[..nnz], &indices[..nnz]);
            prop_assert!(data64[..nnz].iter().zip(&data[..nnz]).all(|(&a, &b)| a == b as f64));
            prop_assert_eq!(new_indptr.len(), indptr.len());
            for (i, r) in rows.iter().enumerate() {
                let mut dense = [0.0f32; 6];
                for &(c, v) in r {
                    dense[c as usize] += v;
                }
                let want: Vec<(i32, f32)> = (0..6).filter(|&c| dense[c] != 0.0).map(|c| (c as i32, dense[c])).collect();
                let (a, z) = (new_indptr[i] as usize, new_indptr[i + 1] as usize);
                let got: Vec<(i32, f32)> = indices[a..z].iter().copied().zip(data[a..z].iter().copied()).collect();
                prop_assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let (mut i, mut d) = (vec![0, 5], vec![1.0, 2.0]);
        assert!(remap_sort_rows(&[0, 2], &mut i, Some(&mut d), &[0, 1]).is_err()); // 5 >= len(forward)
        assert!(remap_sort_rows(&[0, 2], &mut i, None, &[0, 1]).is_err());
        assert!(remap_sort_rows(&[0, 1], &mut i, Some(&mut d), &[]).is_err()); // indptr[-1] != nnz
        assert!(remap_sort_rows(&[0, 2], &mut i, Some(&mut d[..1]), &[]).is_err());
        assert!(canonicalize_rows(&[0, 1], &mut i, &mut d).is_err());
        assert!(canonicalize_rows(&[0, 2], &mut [0, -1], &mut d).is_err());
    }
}
