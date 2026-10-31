//! Input prep: canonical rows (sorted, duplicates summed), the blocked
//! kernel's column renumbering + per-row sort, and interaction logs grouped
//! into a CSR (`group_interactions`).

use rayon::prelude::*;

use crate::Error;
use crate::csr::{CsrView, check_indptr};
use crate::normalize::Value;

/// Entries per sequential leaf; rows of ~100 entries sort in L1.
pub(crate) const LEAF_NNZ: usize = 1 << 16;

/// Renumbers columns (`c -> forward[c]`) and sorts every row by column,
/// carrying `data` along (None: the pattern only), in place and in parallel
/// on the current rayon pool. Already-sorted rows are left as they are.
///
/// Replaces numpy's gather + scipy's single-threaded `sort_indices`: 2.1 s of
/// the 3.3 s recommendation call on Yambda 50M (S with 93M entries).
pub fn remap_sort_rows(
    indptr: &[i64],
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
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [V],
    forward: &[i32],
) {
    par_leaves(indptr, indices, data, LEAF_NNZ, &|_, ip, cols, vals| {
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
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [V],
) -> Result<Vec<i64>, Error> {
    canonicalize_with_leaf(indptr, indices, data, LEAF_NNZ)
}

fn canonicalize_with_leaf<V: Value>(
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [V],
    leaf: usize,
) -> Result<Vec<i64>, Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("canonicalize_rows: {m}")));
    check_indptr("canonicalize_rows", indptr, indices.len())?;
    if data.len() != indices.len() {
        return bad("len(data) != len(indices)");
    }
    if indices.iter().any(|&c| c < 0) {
        return bad("negative column index");
    }
    // Each row keeps its result at its front; the rest is marked with column -1.
    par_leaves(indptr, indices, data, leaf, &|_, ip, cols, vals| {
        let base = ip[0] as usize;
        let mut pairs = Vec::new();
        for w in ip.windows(2) {
            let (a, z) = (w[0] as usize - base, w[1] as usize - base);
            let kept = canonical_row(&mut cols[a..z], &mut vals[a..z], &mut pairs);
            cols[a + kept..z].fill(-1);
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
        new_indptr.push(out as i64);
    }
    Ok(new_indptr)
}

/// One row made canonical in place, as scipy's `sum_duplicates` +
/// `eliminate_zeros`: sorted by column, duplicates summed (in sorted order),
/// zeros dropped. Returns the kept length; the result is at the front.
/// `pairs`: scratch reused across rows.
pub(crate) fn canonical_row<V: Value>(
    cols: &mut [i32],
    vals: &mut [V],
    pairs: &mut Vec<(i32, V)>,
) -> usize {
    sort_row(cols, vals, pairs);
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
    kept
}

/// Per row of `s` its canonical length, per column its canonical entry count
/// (`canonical_row` on each row in one small buffer: one sequential O(nnz)
/// pass, no copy of `s`).
pub(crate) fn canonical_counts(s: &CsrView) -> (Vec<u32>, Vec<u32>) {
    let (mut lens, mut counts) = (vec![0u32; s.n_rows()], vec![0u32; s.n_cols()]);
    let (mut cols, mut vals, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
    for (r, len) in lens.iter_mut().enumerate() {
        let k = canonical_copy(s.arrays(), r, &mut cols, &mut vals, &mut pairs);
        *len = k as u32;
        for &c in &cols[..k] {
            counts[c as usize] += 1;
        }
    }
    (lens, counts)
}

/// Row `r` of (indptr, indices, data) made canonical (`canonical_row`) in
/// `cols`/`vals`; returns its length.
pub(crate) fn canonical_copy(
    (indptr, indices, data): (&[i64], &[i32], &[f32]),
    r: usize,
    cols: &mut Vec<i32>,
    vals: &mut Vec<f32>,
    pairs: &mut Vec<(i32, f32)>,
) -> usize {
    let (a, b) = (indptr[r] as usize, indptr[r + 1] as usize);
    cols.clear();
    cols.extend_from_slice(&indices[a..b]);
    vals.clear();
    vals.extend_from_slice(&data[a..b]);
    canonical_row(cols, vals, pairs)
}

/// Sorts every row of a leaf by column, carrying the values; sorted rows are skipped.
pub(crate) fn sort_rows<V: Copy>(indptr: &[i64], indices: &mut [i32], data: &mut [V]) {
    let base = indptr[0] as usize;
    let mut pairs: Vec<(i32, V)> = Vec::new();
    for w in indptr.windows(2) {
        let (a, z) = (w[0] as usize - base, w[1] as usize - base);
        sort_row(&mut indices[a..z], &mut data[a..z], &mut pairs);
    }
}

fn sort_row<V: Copy>(cols: &mut [i32], vals: &mut [V], pairs: &mut Vec<(i32, V)>) {
    if cols.is_sorted() {
        return;
    }
    pairs.clear();
    pairs.extend(cols.iter().copied().zip(vals.iter().copied()));
    pairs.sort_unstable_by_key(|p| p.0);
    for ((c, v), (pc, pv)) in cols.iter_mut().zip(vals.iter_mut()).zip(pairs.iter()) {
        (*c, *v) = (*pc, *pv);
    }
}

/// Runs `f(first_row, indptr, indices, data)` on leaves of whole rows, in
/// parallel. Splits by entries (as `normalize::par_rows`), so skewed rows
/// balance; a leaf's `indptr` slice is absolute, its `indices`/`data` start at
/// `indptr[0]`.
pub(crate) fn par_leaves<V: Send, F>(
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [V],
    leaf: usize,
    f: &F,
) where
    F: Fn(usize, &[i64], &mut [i32], &mut [V]) + Sync,
{
    par_leaves_from(0, indptr, indices, data, leaf, f);
}

fn par_leaves_from<V: Send, F>(
    row0: usize,
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [V],
    leaf: usize,
    f: &F,
) where
    F: Fn(usize, &[i64], &mut [i32], &mut [V]) + Sync,
{
    let n = indptr.len() - 1;
    let base = indptr[0] as usize;
    if n > 1 && indices.len() > leaf {
        let half = (base + indices.len() / 2) as i64;
        let mid = indptr.partition_point(|&p| p <= half).clamp(1, n - 1);
        let at = indptr[mid] as usize - base;
        let (il, ir) = indices.split_at_mut(at);
        let (dl, dr) = data.split_at_mut(at);
        rayon::join(
            || par_leaves_from(row0, &indptr[..=mid], il, dl, leaf, f),
            || par_leaves_from(row0 + mid, &indptr[mid..], ir, dr, leaf, f),
        );
    } else {
        f(row0, indptr, indices, data);
    }
}

/// What a repeated (row, col) pair keeps in `group_interactions`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Duplicates {
    Sum,
    First,
    Last,
}

/// `group_interactions`' output: a canonical CSR (f32 values, zeros dropped).
pub struct Grouped {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
}

/// Interaction logs (one row code, col code, value, timestamp per event)
/// grouped into a canonical CSR: a counting
/// sort by row (stable: input order), then per row, in parallel, a stable
/// sort by (col, timestamp) and one pass per pair: values summed in that
/// order, or the first / last value (timestamps only order first / last).
/// Pairs whose value is 0 in f32 aren't stored. Peak: the per-row buckets
/// (col + value + timestamp per event) plus the output.
pub fn group_interactions(
    rows: &[i32],
    cols: &[i32],
    values: Option<&[f64]>,
    t: Option<&[f64]>,
    n_rows: usize,
    n_cols: usize,
    duplicates: Duplicates,
) -> Result<Grouped, Error> {
    let bad = |m: &str| Err(Error::InvalidInput(format!("group_interactions: {m}")));
    let n = rows.len();
    if cols.len() != n || values.is_some_and(|v| v.len() != n) || t.is_some_and(|t| t.len() != n) {
        return bad("rows, cols, values and t must have the same length");
    }
    if rows
        .iter()
        .zip(cols)
        .any(|(&r, &c)| r < 0 || c < 0 || r as usize >= n_rows || c as usize >= n_cols)
    {
        return bad("code out of range");
    }
    // Counting sort by row: bucket offsets, then a stable scatter.
    let mut start = vec![0usize; n_rows + 1];
    for &r in rows {
        start[r as usize + 1] += 1;
    }
    for r in 0..n_rows {
        start[r + 1] += start[r];
    }
    let mut b_col = vec![0i32; n];
    let mut b_val = values.map(|_| vec![0f64; n]);
    let mut b_t = t.map(|_| vec![0f64; n]);
    let mut next = start.clone();
    for e in 0..n {
        let p = next[rows[e] as usize];
        next[rows[e] as usize] += 1;
        b_col[p] = cols[e];
        if let (Some(b), Some(v)) = (b_val.as_mut(), values) {
            b[p] = v[e];
        }
        if let (Some(b), Some(t)) = (b_t.as_mut(), t) {
            b[p] = t[e];
        }
    }
    drop(next);

    // Per row: (cols, values) of its stored pairs.
    let row_out: Vec<(Vec<i32>, Vec<f32>)> = (0..n_rows)
        .into_par_iter()
        .map(|r| {
            let (a, z) = (start[r], start[r + 1]);
            let c = &b_col[a..z];
            let v = b_val.as_deref().map(|b| &b[a..z]);
            let ts = b_t.as_deref().map(|b| &b[a..z]);
            let mut order: Vec<usize> = (0..z - a).collect();
            order.sort_by(|&x, &y| {
                c[x].cmp(&c[y]).then_with(|| match (ts, duplicates) {
                    (Some(ts), Duplicates::First | Duplicates::Last) => ts[x].total_cmp(&ts[y]),
                    _ => std::cmp::Ordering::Equal,
                })
            });
            let (mut oc, mut ov) = (Vec::new(), Vec::new());
            let mut g = 0;
            while g < order.len() {
                let mut h = g;
                while h < order.len() && c[order[h]] == c[order[g]] {
                    h += 1;
                }
                let group = &order[g..h];
                let val = |e: usize| v.map_or(1.0, |v| v[e]);
                let value = match duplicates {
                    Duplicates::Sum => group.iter().map(|&e| val(e)).sum::<f64>(),
                    Duplicates::First => val(group[0]),
                    Duplicates::Last => val(group[group.len() - 1]),
                };
                if value as f32 != 0.0 {
                    oc.push(c[group[0]]);
                    ov.push(value as f32);
                }
                g = h;
            }
            (oc, ov)
        })
        .collect();
    drop((b_col, b_val, b_t));

    let nnz: usize = row_out.iter().map(|o| o.0.len()).sum();
    let mut out = Grouped {
        indptr: Vec::with_capacity(n_rows + 1),
        indices: Vec::with_capacity(nnz),
        data: Vec::with_capacity(nnz),
    };
    out.indptr.push(0);
    for (oc, ov) in row_out {
        out.indices.extend_from_slice(&oc);
        out.data.extend_from_slice(&ov);
        out.indptr.push(out.indices.len() as i64);
    }
    Ok(out)
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
                indptr.push(indices.len() as i64);
            }
            // value = original column, so pairs can be checked after the move
            let mut data: Vec<f32> = indices.iter().map(|&c| c as f32).collect();
            // the public entry, pattern only (zero-sized values), as reference for the columns
            let mut pattern = indices.clone();
            let identity: Vec<i32> = (0..20).collect();
            remap_sort_rows(&indptr, &mut pattern, None, if remap { &forward } else { &identity }).unwrap();
            // small leaves exercise the parallel split
            par_leaves(&indptr, &mut indices, &mut data, leaf, &|_, ip, c, d| {
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
                indptr.push(indices.len() as i64);
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
    fn group_interactions_rules() {
        // events: (row, col, value, t); row 1: a +1/-1 pair
        let rows = [0, 0, 0, 1, 1, 0];
        let cols = [2, 0, 2, 1, 1, 2];
        let vals = [1.0, 5.0, 2.0, 1.0, -1.0, 3.0];
        let t = [7.0, 1.0, 9.0, 0.0, 0.0, 7.0];
        let g = |d| group_interactions(&rows, &cols, Some(&vals), Some(&t), 2, 3, d).unwrap();
        let sum = g(Duplicates::Sum);
        assert_eq!(
            (sum.indptr, sum.indices, sum.data),
            (vec![0, 2, 2], vec![0, 2], vec![5.0, 6.0])
        );
        // (0, 2) by time: 1.0 @7, 3.0 @7 (tie: input order), 2.0 @9
        assert_eq!(g(Duplicates::First).data, [5.0, 1.0, 1.0]); // row 1: first = +1 (tie: input order)
        assert_eq!(g(Duplicates::Last).data, [5.0, 2.0, -1.0]);
        let no_t = group_interactions(&rows, &cols, None, None, 2, 3, Duplicates::Sum).unwrap();
        assert_eq!(
            (no_t.data, no_t.indptr),
            (vec![1.0, 3.0, 2.0], vec![0, 2, 3])
        );
        assert!(group_interactions(&[0], &[3], None, None, 1, 3, Duplicates::Sum).is_err());
        assert!(group_interactions(&[-1], &[0], None, None, 1, 3, Duplicates::Sum).is_err());
    }

    // Each leaf's first row index matches its slices, however the rows split.
    #[test]
    fn par_leaves_first_row() {
        let indptr: Vec<i64> = vec![0, 1, 1, 3, 4, 6, 6, 7];
        let mut indices = vec![0; 7];
        // value = row of the entry
        let mut data: Vec<usize> = vec![0, 2, 2, 3, 4, 4, 6];
        let seen = std::sync::Mutex::new(vec![false; 7]);
        par_leaves(&indptr, &mut indices, &mut data, 0, &|row0, ip, _, d| {
            let base = ip[0] as usize;
            for (i, w) in ip.windows(2).enumerate() {
                seen.lock().unwrap()[row0 + i] = true;
                for &r in &d[w[0] as usize - base..w[1] as usize - base] {
                    assert_eq!(r, row0 + i);
                }
            }
        });
        assert!(seen.into_inner().unwrap().iter().all(|&s| s));
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
