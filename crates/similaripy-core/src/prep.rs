//! Input prep: canonical rows (sorted, duplicates summed), the blocked
//! kernel's column renumbering + per-row sort, and interaction logs grouped
//! into a CSR (`group_interactions`).

use rayon::prelude::*;

use crate::Error;
use crate::csr::check_indptr;

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
/// or sums that cancel) dropped; float32 values. Rows are processed in parallel, then compacted
/// to the front of `indices`/`data` in one sequential pass; returns the new
/// indptr (the first `new_indptr[-1]` entries are the result).
///
/// The canonical pass of `canonical` (parallel; scipy's `sum_duplicates` is
/// single-threaded): duplicate entries would otherwise reach the kernel
/// unsummed (wrong norms and binary counts).
pub fn canonicalize_rows(
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [f32],
) -> Result<Vec<i64>, Error> {
    canonicalize_with_leaf(indptr, indices, data, LEAF_NNZ)
}

fn canonicalize_with_leaf(
    indptr: &[i64],
    indices: &mut [i32],
    data: &mut [f32],
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
fn canonical_row(cols: &mut [i32], vals: &mut [f32], pairs: &mut Vec<(i32, f32)>) -> usize {
    sort_row(cols, vals, pairs);
    let mut n = 0; // entries kept so far, sums still open
    for j in 0..cols.len() {
        if n > 0 && cols[n - 1] == cols[j] {
            vals[n - 1] += vals[j];
        } else {
            (cols[n], vals[n]) = (cols[j], vals[j]);
            n += 1;
        }
    }
    let mut kept = 0;
    for j in 0..n {
        if vals[j] != 0.0 {
            (cols[kept], vals[kept]) = (cols[j], vals[j]);
            kept += 1;
        }
    }
    kept
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
/// parallel. Splits by entries, so skewed rows
/// balance; a leaf's `indptr` slice is absolute, its `indices`/`data` start at
/// `indptr[0]`.
pub(crate) fn par_leaves<I: Send, V: Send, F>(
    indptr: &[i64],
    indices: &mut [I],
    data: &mut [V],
    leaf: usize,
    f: &F,
) where
    F: Fn(usize, &[i64], &mut [I], &mut [V]) + Sync,
{
    par_leaves_from(0, indptr, indices, data, leaf, f);
}

/// `par_leaves` over one array of row entries (e.g. (column, value) records).
pub(crate) fn par_leaves_one<T: Send, F>(indptr: &[i64], out: &mut [T], leaf: usize, f: &F)
where
    F: Fn(usize, &[i64], &mut [T]) + Sync,
{
    // Zero-sized: split alongside `out`, never allocated.
    let mut none = vec![(); out.len()];
    par_leaves(indptr, out, &mut none, leaf, &|row0, ip, o, _| {
        f(row0, ip, o)
    });
}

fn par_leaves_from<I: Send, V: Send, F>(
    row0: usize,
    indptr: &[i64],
    indices: &mut [I],
    data: &mut [V],
    leaf: usize,
    f: &F,
) where
    F: Fn(usize, &[i64], &mut [I], &mut [V]) + Sync,
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
            // Within a column, by timestamp only where first/last needs it.
            let by_t = ts.filter(|_| matches!(duplicates, Duplicates::First | Duplicates::Last));
            let mut order: Vec<usize> = (0..z - a).collect();
            order.sort_by(|&x, &y| {
                c[x].cmp(&c[y])
                    .then_with(|| by_t.map_or(std::cmp::Ordering::Equal, |t| t[x].total_cmp(&t[y])))
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

/// A caller's sparse matrix, borrowed (scipy's arrays): `indptr` + `indices`,
/// or COO `rows` + `cols`, of one index type `I` (i32 | i64: scipy never mixes
/// them), values `V` (f32 | f64).
pub enum Sparse<'a, I, V> {
    Csr {
        indptr: &'a [I],
        indices: &'a [I],
        data: &'a [V],
    },
    Csc {
        indptr: &'a [I],
        indices: &'a [I],
        data: &'a [V],
    },
    Coo {
        rows: &'a [I],
        cols: &'a [I],
        data: &'a [V],
    },
}

/// An owned canonical CSR: rows sorted by column, duplicates summed, no
/// stored zeros. `indices`/`data` may have spare capacity past `indptr[-1]`.
pub struct Csr<T> {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<T>,
}

/// The one door in: a caller's `n_rows x n_cols` matrix copied once into an
/// owned canonical CSR of f32 values. The cast comes before duplicates are
/// summed (a float64 value past float32's range becomes inf, a tiny one 0 and
/// is dropped). `canonical`: the caller's hint that the input is sorted with
/// no duplicates (scipy's `has_canonical_format`); zeros are dropped either way.
///
/// CSR: a cast copy. CSC and COO: one serial, stable counting sort by row (as
/// scipy's `tocsr`): CSC rows come out sorted with duplicates in stored order,
/// COO rows keep their stored order. Then `canonicalize_rows` (on the current
/// rayon pool) unless the hint is true and no value is 0. Peak: the output +
/// O(rows + cols). The bindings call it with the GIL released: under
/// free-threading, a concurrent write to the input arrays races the copy.
pub fn canonical<I, V>(
    m: Sparse<'_, I, V>,
    (n_rows, n_cols): (usize, usize),
    canonical: bool,
) -> Result<Csr<f32>, Error>
where
    I: Copy + Into<i64>,
    V: Copy + Into<f64>,
{
    let bad = |m: &str| Error::InvalidInput(format!("canonical: {m}"));
    if n_cols > i32::MAX as usize + 1 {
        return Err(bad("more than 2^31 columns (column ids are int32)"));
    }
    let in_range = |ids: &[I], n: usize| ids.iter().all(|&x| (0..n as i64).contains(&x.into()));
    let cast = |v: &V| (*v).into() as f32;
    let (mut indptr, mut indices, mut data) = match m {
        Sparse::Csr { indices, data, .. } | Sparse::Csc { indices, data, .. }
            if data.len() != indices.len() =>
        {
            return Err(bad("len(data) != len(indices)"));
        }
        Sparse::Coo { rows, cols, data }
            if rows.len() != cols.len() || data.len() != rows.len() =>
        {
            return Err(bad("rows, cols and data must have the same length"));
        }
        Sparse::Csr {
            indptr,
            indices,
            data,
        } => {
            let indptr = offsets(indptr, n_rows, indices.len())?;
            if !in_range(indices, n_cols) {
                return Err(bad("column index out of range"));
            }
            let indices = indices.iter().map(|&c| c.into() as i32).collect();
            (indptr, indices, data.iter().map(cast).collect())
        }
        Sparse::Csc {
            indptr,
            indices,
            data,
        } => {
            let colptr = offsets(indptr, n_cols, indices.len())?;
            if !in_range(indices, n_rows) {
                return Err(bad("row index out of range"));
            }
            let cols = colptr
                .windows(2)
                .enumerate()
                .flat_map(|(j, w)| std::iter::repeat_n(j as i32, (w[1] - w[0]) as usize));
            scatter(indices, cols, data, n_rows, cast)
        }
        Sparse::Coo { rows, cols, data } => {
            if !in_range(rows, n_rows) || !in_range(cols, n_cols) {
                return Err(bad("row or column index out of range"));
            }
            scatter(
                rows,
                cols.iter().map(|&c| c.into() as i32),
                data,
                n_rows,
                cast,
            )
        }
    };
    if !canonical || data.contains(&0.0) {
        indptr = canonicalize_rows(&indptr, &mut indices, &mut data)?;
        let nnz = indptr[n_rows] as usize;
        indices.truncate(nnz);
        data.truncate(nnz);
    }
    Ok(Csr {
        indptr,
        indices,
        data,
    })
}

/// A caller's `n_major + 1` offsets widened to i64 and checked (`check_indptr`).
fn offsets<I: Copy + Into<i64>>(ptr: &[I], n_major: usize, nnz: usize) -> Result<Vec<i64>, Error> {
    if ptr.len() != n_major + 1 {
        return Err(Error::InvalidInput(
            "canonical: len(indptr) != n + 1".into(),
        ));
    }
    let ptr: Vec<i64> = ptr.iter().map(|&p| p.into()).collect();
    check_indptr("canonical", &ptr, nnz)?;
    Ok(ptr)
}

/// Stable counting sort of entries by row (ids already checked): each row's
/// entries in input order, values cast on the way.
fn scatter<I: Copy + Into<i64>, V>(
    rows: &[I],
    cols: impl Iterator<Item = i32>,
    data: &[V],
    n_rows: usize,
    cast: impl Fn(&V) -> f32,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let row = |r: I| r.into() as usize;
    let mut indptr = vec![0i64; n_rows + 1];
    for &r in rows {
        indptr[row(r) + 1] += 1;
    }
    for i in 0..n_rows {
        indptr[i + 1] += indptr[i];
    }
    let mut next = indptr.clone();
    let (mut out_cols, mut out_vals) = (vec![0i32; rows.len()], vec![0f32; rows.len()]);
    for ((&r, c), v) in rows.iter().zip(cols).zip(data) {
        let p = &mut next[row(r)];
        out_cols[*p as usize] = c;
        out_vals[*p as usize] = cast(v);
        *p += 1;
    }
    (indptr, out_cols, out_vals)
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
            let new_indptr = canonicalize_with_leaf(&indptr, &mut indices, &mut data, leaf).unwrap();
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

    /// (indptr, indices) of `entries` (row, col, value) grouped by `major`
    /// (rows for CSR, cols for CSC), stable: entries keep their order.
    fn compressed(
        entries: &[(i64, i64, f64)],
        n_major: usize,
        by_col: bool,
    ) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut order: Vec<usize> = (0..entries.len()).collect();
        let key = |e: &(i64, i64, f64)| if by_col { e.1 } else { e.0 };
        order.sort_by_key(|&i| key(&entries[i]));
        let mut indptr = vec![0i64; n_major + 1];
        for e in entries {
            indptr[key(e) as usize + 1] += 1;
        }
        for i in 0..n_major {
            indptr[i + 1] += indptr[i];
        }
        let minor = order
            .iter()
            .map(|&i| if by_col { entries[i].0 } else { entries[i].1 })
            .collect();
        (indptr, minor, order.iter().map(|&i| entries[i].2).collect())
    }

    /// Every format and index type of `entries`, canonical with `hint`.
    fn every_form(entries: &[(i64, i64, f64)], shape: (usize, usize), hint: bool) -> Vec<Csr<f32>> {
        let (rp, rc, rv) = compressed(entries, shape.0, false);
        let (cp, cr, cv) = compressed(entries, shape.1, true);
        let (rows, cols, vals): (Vec<i64>, Vec<i64>, Vec<f64>) = (
            entries.iter().map(|e| e.0).collect(),
            entries.iter().map(|e| e.1).collect(),
            entries.iter().map(|e| e.2).collect(),
        );
        let n32 = |v: &[i64]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
        let (rp32, rc32, cp32, cr32, rows32, cols32) = (
            n32(&rp),
            n32(&rc),
            n32(&cp),
            n32(&cr),
            n32(&rows),
            n32(&cols),
        );
        let vals32: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        vec![
            canonical(
                Sparse::Csr {
                    indptr: &rp,
                    indices: &rc,
                    data: &rv,
                },
                shape,
                hint,
            )
            .unwrap(),
            canonical(
                Sparse::Csc {
                    indptr: &cp,
                    indices: &cr,
                    data: &cv,
                },
                shape,
                hint,
            )
            .unwrap(),
            canonical(
                Sparse::Coo {
                    rows: &rows,
                    cols: &cols,
                    data: &vals,
                },
                shape,
                hint,
            )
            .unwrap(),
            canonical(
                Sparse::Csr {
                    indptr: &rp32,
                    indices: &rc32,
                    data: &rv,
                },
                shape,
                hint,
            )
            .unwrap(),
            canonical(
                Sparse::Csc {
                    indptr: &cp32,
                    indices: &cr32,
                    data: &cv,
                },
                shape,
                hint,
            )
            .unwrap(),
            canonical(
                Sparse::Coo {
                    rows: &rows32,
                    cols: &cols32,
                    data: &vals32,
                },
                shape,
                hint,
            )
            .unwrap(),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]
        // Every format and index type: the dense per-cell sum (values exact in
        // any order), sorted rows, no zeros, spare capacity cut.
        #[test]
        fn canonical_matches_dense_sum(
            entries in proptest::collection::vec(
                (0i64..7, 0i64..5, prop_oneof![Just(0.0f64), Just(1.0), Just(-1.0), Just(2.5)]),
                0..if cfg!(miri) { 12 } else { 60 },
            ),
            empty_tail in 0usize..3,
        ) {
            let shape = (7 + empty_tail, 5 + empty_tail);
            let mut dense = vec![vec![0.0f64; shape.1]; shape.0];
            for &(r, c, v) in &entries {
                dense[r as usize][c as usize] += v;
            }
            let mut want = (vec![0i64], vec![], vec![]);
            for row in &dense {
                for (c, &v) in row.iter().enumerate().filter(|(_, v)| **v != 0.0) {
                    want.1.push(c as i32);
                    want.2.push(v);
                }
                want.0.push(want.1.len() as i64);
            }
            for out in every_form(&entries, shape, false) {
                prop_assert_eq!(&out.indptr, &want.0);
                prop_assert_eq!(&out.indices, &want.1);
                prop_assert!(out.data.iter().zip(&want.2).all(|(&a, &b)| a as f64 == b));
            }
        }
    }

    #[test]
    fn canonical_csc_sums_in_stored_order() {
        // f32 sums that depend on the order; each column holds row 2 three times, summed in stored
        // order (as scipy's tocsr): column 0 -1e8 + 1e8 + 1 = 1, column 1 1e8 + 1 - 1e8 = 0 (dropped).
        let (indptr, indices) = ([0i64, 3, 6], [2i64, 2, 2, 2, 2, 2]);
        let data = [-1e8, 1e8, 1.0, 1e8, 1.0, -1e8];
        let out: Csr<f32> = canonical(
            Sparse::Csc {
                indptr: &indptr,
                indices: &indices,
                data: &data,
            },
            (3, 2),
            false,
        )
        .unwrap();
        assert_eq!(
            (out.indptr, out.indices, out.data),
            (vec![0, 0, 0, 1], vec![0], vec![1.0])
        );
    }

    #[test]
    fn canonical_casts_before_summing() {
        // f64 past f32's range -> inf (kept, the similarities report it); 1e-50 -> 0 in f32 (dropped);
        // 0.1 + 0.2 summed in f32 after the cast
        let (rows, cols) = ([0i64, 0, 1, 1], [0i64, 1, 0, 0]);
        let data = [1e300, 1e-50, 0.1, 0.2];
        let coo = || Sparse::Coo {
            rows: &rows,
            cols: &cols,
            data: &data,
        };
        let out: Csr<f32> = canonical(coo(), (2, 2), false).unwrap();
        assert_eq!((out.indptr, out.indices), (vec![0, 1, 2], vec![0, 0]));
        assert_eq!(out.data, [f32::INFINITY, 0.1f32 + 0.2f32]);
    }

    #[test]
    fn canonical_hint_skips_the_pass() {
        // A wrong hint isn't checked (scipy's flag is trusted, as before): rows stay as given.
        // Zeros are dropped whatever the hint.
        let (indptr, indices, data) = ([0i32, 2], [1i32, 0], [1.0f32, 2.0]);
        let csr = || Sparse::Csr {
            indptr: &indptr,
            indices: &indices,
            data: &data,
        };
        assert_eq!(canonical(csr(), (1, 2), true).unwrap().indices, [1, 0]);
        assert_eq!(canonical(csr(), (1, 2), false).unwrap().indices, [0, 1]);
        let zero = [1.0f32, 0.0];
        let out: Csr<f32> = canonical(
            Sparse::Csr {
                indptr: &indptr,
                indices: &indices,
                data: &zero,
            },
            (1, 2),
            true,
        )
        .unwrap();
        assert_eq!((out.indptr, out.indices), (vec![0, 1], vec![1]));
    }

    #[test]
    fn canonical_empty() {
        for shape in [(0, 0), (0, 3), (3, 0), (4, 4)] {
            let ptr = vec![0i64; shape.0 + 1];
            let out: Csr<f32> = canonical(
                Sparse::Csr {
                    indptr: &ptr,
                    indices: &[],
                    data: &[0f32; 0],
                },
                shape,
                false,
            )
            .unwrap();
            assert_eq!(out.indptr, ptr);
            let ptr = vec![0i64; shape.1 + 1];
            let out: Csr<f32> = canonical(
                Sparse::Csc {
                    indptr: &ptr,
                    indices: &[],
                    data: &[0f32; 0],
                },
                shape,
                false,
            )
            .unwrap();
            assert_eq!((out.indptr.len(), out.indices.len()), (shape.0 + 1, 0));
            let out: Csr<f32> = canonical(
                Sparse::Coo {
                    rows: &[0i32; 0],
                    cols: &[],
                    data: &[0f32; 0],
                },
                shape,
                false,
            )
            .unwrap();
            assert_eq!((out.indptr.len(), out.data.len()), (shape.0 + 1, 0));
        }
    }

    #[test]
    fn canonical_rejects_bad_inputs() {
        let c = |m: Sparse<'_, i64, f32>, shape| canonical(m, shape, false).is_err();
        let d = [1.0f32, 2.0];
        assert!(c(
            Sparse::Csr {
                indptr: &[0, 2],
                indices: &[0, 2],
                data: &d
            },
            (1, 2)
        )); // col 2 >= 2
        assert!(c(
            Sparse::Csr {
                indptr: &[0, 2],
                indices: &[0, -1],
                data: &d
            },
            (1, 2)
        ));
        assert!(c(
            Sparse::Csr {
                indptr: &[0, 2],
                indices: &[0, 1],
                data: &d[..1]
            },
            (1, 2)
        ));
        assert!(c(
            Sparse::Csr {
                indptr: &[0, 1],
                indices: &[0, 1],
                data: &d
            },
            (1, 2)
        )); // indptr[-1] != nnz
        assert!(c(
            Sparse::Csr {
                indptr: &[0, 2],
                indices: &[0, 1],
                data: &d
            },
            (2, 2)
        )); // len(indptr)
        assert!(c(
            Sparse::Csr {
                indptr: &[1, 2],
                indices: &[0, 1],
                data: &d
            },
            (1, 2)
        ));
        assert!(c(
            Sparse::Csc {
                indptr: &[0, 2, 1],
                indices: &[0, 0],
                data: &d
            },
            (1, 2)
        )); // not monotone
        assert!(c(
            Sparse::Csc {
                indptr: &[0, 1, 2],
                indices: &[0, 1],
                data: &d
            },
            (1, 2)
        )); // row 1 >= 1
        assert!(c(
            Sparse::Coo {
                rows: &[0, 1],
                cols: &[0, 0],
                data: &d
            },
            (1, 2)
        ));
        assert!(c(
            Sparse::Coo {
                rows: &[0, 0],
                cols: &[0, 5],
                data: &d
            },
            (1, 2)
        ));
        assert!(c(
            Sparse::Coo {
                rows: &[0],
                cols: &[0, 0],
                data: &d
            },
            (1, 2)
        ));
        assert!(c(
            Sparse::Coo {
                rows: &[],
                cols: &[],
                data: &[]
            },
            (1, (1 << 31) + 1)
        )); // > 2^31 columns
        assert!(!c(
            Sparse::Coo {
                rows: &[],
                cols: &[],
                data: &[]
            },
            (1, 1 << 31)
        ));
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
