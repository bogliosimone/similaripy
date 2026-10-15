//! Property test: the kernel against a naive dense f64 reference.
//! Scores must match within rtol; the kept columns must be a valid top-K
//! (any column whose reference score ties the k-th within tolerance may swap).

use proptest::prelude::*;
use similaripy_core::{CsrPattern, CsrView, Inputs, Norms, Params, compute};

const RTOL: f64 = 1e-5;
const ATOL: f64 = 1e-6;

struct Csr {
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
}

fn to_csr(dense: &[Vec<f32>]) -> Csr {
    let mut m = Csr {
        indptr: vec![0],
        indices: vec![],
        data: vec![],
    };
    for row in dense {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                m.indices.push(c as i32);
                m.data.push(v);
            }
        }
        m.indptr.push(m.indices.len() as i32);
    }
    m
}

fn dense(rows: usize, cols: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
    // Small integers keep every dot product exact in f32; negative ones (signed
    // input, D45) make sums cancel to exactly 0.0 (re-pushed columns, dropped
    // zero scores) and give negative scores, never results.
    let cell = prop_oneof![
        3 => Just(0.0f32),
        2 => (1u8..4).prop_map(f32::from),
        1 => (-3i8..0).prop_map(f32::from),
    ];
    prop::collection::vec(prop::collection::vec(cell, cols), rows)
}

fn mask(rows: usize, cols: usize) -> impl Strategy<Value = Option<Vec<Vec<f32>>>> {
    prop::option::of(prop::collection::vec(
        prop::collection::vec(prop_oneof![Just(0.0f32), Just(1.0f32)], cols),
        rows,
    ))
}

#[derive(Debug)]
struct Case {
    m1: Vec<Vec<f32>>,
    m2: Vec<Vec<f32>>,
    norms: [Vec<f32>; 6],
    params: [f32; 8], // a1 l1 l2 l3 t1 t2 stabilized bayesian
    threshold: f32,
    filter: Option<Vec<Vec<f32>>>,
    target: Option<Vec<Vec<f32>>>,
    k: usize,
    /// Target rows: any subset/order, duplicates allowed.
    targets: Vec<usize>,
    /// Reverse each m2 row's indices (allowed on the single-block path).
    unsorted_m2: bool,
    /// 0 = no blocking; otherwise blocks of this many columns.
    block_size: usize,
}

fn case() -> impl Strategy<Value = Case> {
    (1usize..7, 1usize..7, 1usize..10).prop_flat_map(|(r, m, n)| {
        let pos = |len| prop::collection::vec(0.1f32..3.0, len);
        let weight = prop_oneof![Just(0.0f32), 0.1f32..2.0];
        (
            dense(r, m),
            dense(m, n),
            [pos(r), pos(n), pos(r), pos(n), pos(r), pos(n)],
            (
                prop_oneof![Just(1.0f32), Just(0.5)],
                weight.clone(),
                weight.clone(),
                weight.clone(),
                0.0f32..1.0,
                0.0f32..1.0,
                weight.clone(),
                weight,
            ),
            mask(r, n),
            mask(r, n),
            1..=n,
            prop::collection::vec(0..r, 1..=2 * r),
            any::<bool>(),
            prop_oneof![Just(0usize), 1..=n],
            prop_oneof![Just(0.0f32), Just(0.5), Just(1.0)],
        )
            .prop_map(
                |(
                    m1,
                    m2,
                    norms,
                    p,
                    filter,
                    target,
                    k,
                    targets,
                    unsorted_m2,
                    block_size,
                    threshold,
                )| Case {
                    m1,
                    m2,
                    norms,
                    params: [p.0, p.1, p.2, p.3, p.4, p.5, p.6, p.7],
                    threshold,
                    filter,
                    target,
                    k,
                    targets,
                    unsorted_m2,
                    block_size,
                },
            )
    })
}

/// Independent f64 version of the 0.7.0 scoring formula (with the F1/F14 fixes). Returns the score
/// and its absolute tolerance: the f32 kernel loses about `eps * cond` relative
/// accuracy when the denominator terms cancel (cond = sum|terms| / |den|).
fn ref_score(c: &Case, row: usize, col: usize, xy: f64) -> (f64, f64) {
    let [a1, l1, l2, l3, t1, t2, ss, bs] = c.params.map(f64::from);
    let n = |i: usize, j: usize| f64::from(c.norms[i][j]);
    let p = xy.powf(a1);
    if l1 == 0.0 && l2 == 0.0 && l3 == 0.0 && ss == 0.0 && bs == 0.0 {
        return (p, ATOL + RTOL * p.abs());
    }
    let tv = l1 * (t1 * (n(0, row) - xy) + t2 * (n(1, col) - xy) + xy);
    let cs = l2 * n(2, row) * n(3, col);
    let dp = l3 * n(4, row) * n(5, col);
    let den = tv + cs + dp + ss;
    let has_den = l1 != 0.0 || l2 != 0.0 || l3 != 0.0 || ss != 0.0;
    let mut v = if !has_den {
        p
    } else if den != 0.0 {
        p / den
    } else {
        0.0
    };
    if bs != 0.0 {
        v *= p / (p + bs);
    }
    let terms = l1 * (t1 * (n(0, row) + xy.abs()) + t2 * (n(1, col) + xy.abs()) + xy.abs())
        + cs.abs()
        + dp.abs()
        + ss;
    let cond = if den != 0.0 { terms / den.abs() } else { 1.0 };
    (
        v,
        ATOL + v.abs() * (RTOL + 4.0 * f64::from(f32::EPSILON) * cond),
    )
}

fn pattern(m: &Option<Csr>, n: usize) -> Option<CsrPattern<'_>> {
    m.as_ref()
        .map(|m| CsrPattern::new("sel", &m.indptr, &m.indices, n).unwrap())
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn check(c: &Case) -> Result<(), TestCaseError> {
    let n = c.m2[0].len();
    let (m1, mut m2) = (to_csr(&c.m1), to_csr(&c.m2));
    if c.unsorted_m2 {
        for w in m2.indptr.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            m2.indices[a..b].reverse();
            m2.data[a..b].reverse();
        }
    }
    let (filter, target) = (
        c.filter.as_deref().map(to_csr),
        c.target.as_deref().map(to_csr),
    );
    let targets: Vec<i32> = c.targets.iter().map(|&t| t as i32).collect();
    let r = targets.len();
    let [a1, l1, l2, l3, t1, t2, stabilized_shrink, bayesian_shrink] = c.params;
    let inp = Inputs {
        m1: CsrView::new("m1", &m1.indptr, &m1.indices, &m1.data, c.m2.len()).unwrap(),
        m2: CsrView::new("m2", &m2.indptr, &m2.indices, &m2.data, n).unwrap(),
        targets: &targets,
        norms: Norms {
            x_tversky: &c.norms[0],
            y_tversky: &c.norms[1],
            x_cosine: &c.norms[2],
            y_cosine: &c.norms[3],
            x_depop: &c.norms[4],
            y_depop: &c.norms[5],
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
            threshold: c.threshold,
        },
        filter: pattern(&filter, n),
        target_cols: pattern(&target, n),
        col_map: None,
    };
    let k = c.k;
    let res = compute(&inp, k, c.block_size, &|| {});
    let blocked = c.block_size > 0 && n > c.block_size;
    // Unsorted m2 rows only reject when the blocked path would binary-search them.
    if blocked && c.unsorted_m2 && m2.indptr.windows(2).any(|w| w[1] - w[0] > 1) {
        prop_assert!(res.is_err(), "unsorted m2 accepted on the blocked path");
        return Ok(());
    }
    let out = res.unwrap();
    prop_assert_eq!(out.counts.len(), r);
    let mut start = 0;

    for (i, &t) in c.targets.iter().enumerate() {
        // Reference scores of every eligible (touched, selected) column.
        let mut eligible: Vec<(usize, f64, f64)> = vec![];
        for col in 0..n {
            let touched = (0..c.m2.len()).any(|u| c.m1[t][u] != 0.0 && c.m2[u][col] != 0.0);
            let xy: f64 = (0..c.m2.len())
                .map(|u| f64::from(c.m1[t][u]) * f64::from(c.m2[u][col]))
                .sum();
            let filtered = c.filter.as_ref().is_some_and(|f| f[t][col] != 0.0);
            let untargeted = c.target.as_ref().is_some_and(|g| g[t][col] == 0.0);
            let (score, tol) = ref_score(c, t, col, xy);
            let threshold = f64::from(c.threshold);
            // xy == 0.0 is exact in f32 too (a cancelled sum): score 0, dropped.
            // NaN (a negative xy ** 0.5) is never a result either.
            if touched && !filtered && !untargeted && xy != 0.0 {
                // f32 vs f64: within tolerance of the threshold (or of 0) either side is right.
                prop_assume!(!close(score, threshold, tol) && !close(score, 0.0, tol));
                if score > 0.0 && score >= threshold {
                    eligible.push((col, score, tol));
                }
            }
        }
        let expected = k.min(eligible.len());
        let mut sorted: Vec<f64> = eligible.iter().map(|e| e.1).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        // Packed rows: exactly the expected count, sorted by column, no duplicates.
        prop_assert_eq!(out.counts[i] as usize, expected, "row {}: count", t);
        let span = start..start + expected;
        start = span.end;
        let got: Vec<(i32, f32)> = out.cols[span.clone()]
            .iter()
            .copied()
            .zip(out.values[span].iter().copied())
            .collect();
        prop_assert!(
            got.windows(2).all(|w| w[0].0 < w[1].0),
            "row {}: not sorted",
            t
        );
        for &(col, score) in &got {
            let reference = eligible.iter().find(|e| e.0 == col as usize);
            prop_assert!(reference.is_some(), "row {t}: col {col} not eligible");
            let &(_, reference, tol) = reference.unwrap();
            prop_assert!(
                close(f64::from(score), reference, tol),
                "row {t} col {col}: {score} vs {reference}"
            );
            // Near-ties with the k-th score may swap.
            let kth = eligible
                .iter()
                .filter(|e| e.1 == sorted[expected - 1])
                .map(|e| e.2)
                .fold(0.0, f64::max);
            prop_assert!(
                reference >= sorted[expected - 1] - tol - kth,
                "row {t}: col {col} below k-th"
            );
        }
    }
    prop_assert_eq!(out.cols.len(), start, "padding left");
    prop_assert_eq!(out.values.len(), start, "padding left");
    Ok(())
}

proptest! {
    // Miri runs ~1000x slower: a few cases still exercise every unsafe path.
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 2000 }))]
    #[test]
    fn matches_dense_reference(c in case()) {
        check(&c)?;
    }
}

// Output column ids are i32 (D11): wider m2 is rejected before any allocation.
#[test]
fn rejects_more_than_i32_max_columns() {
    let (indptr, indices, data) = ([0, 0], [], []);
    let empty: [f32; 0] = [];
    let inp = Inputs {
        m1: CsrView::new("m1", &indptr, &indices, &data, 1).unwrap(),
        m2: CsrView::new("m2", &indptr, &indices, &data, i32::MAX as usize + 1).unwrap(),
        targets: &[0],
        norms: Norms {
            x_tversky: &empty,
            y_tversky: &empty,
            x_cosine: &empty,
            y_cosine: &empty,
            x_depop: &empty,
            y_depop: &empty,
        },
        params: Params {
            a1: 1.0,
            l1: 0.0,
            l2: 0.0,
            l3: 0.0,
            t1: 1.0,
            t2: 1.0,
            stabilized_shrink: 0.0,
            bayesian_shrink: 0.0,
            threshold: 0.0,
        },
        filter: None,
        target_cols: None,
        col_map: None,
    };
    assert!(compute(&inp, 1, 0, &|| {}).is_err());
}

// The selection bitsets are m2.n_cols wide: a wider selection would index past them.
#[test]
fn rejects_selection_wider_than_m2() {
    let (indptr, indices, data) = ([0, 1], [0], [1.0]);
    let (sel_indptr, sel_indices) = ([0, 1], [4]);
    let inp = Inputs {
        m1: CsrView::new("m1", &indptr, &indices, &data, 1).unwrap(),
        m2: CsrView::new("m2", &indptr, &indices, &data, 2).unwrap(),
        targets: &[0],
        norms: Norms::default(),
        params: Params::default(),
        filter: Some(CsrPattern::new("filter", &sel_indptr, &sel_indices, 5).unwrap()),
        target_cols: None,
        col_map: None,
    };
    assert!(compute(&inp, 1, 0, &|| {}).is_err());
}

/// Top-`k` of `m1 · m2` (dot product): per-row counts and the packed
/// (col, score bits) entries, exactly as the kernel returns them.
#[allow(clippy::too_many_arguments)]
fn top_scores(
    m1: &Csr,
    m2: &Csr,
    m: usize,
    n: usize,
    r: usize,
    k: usize,
    block: usize,
    col_map: Option<&[i32]>,
) -> (Vec<u32>, Vec<(i32, u32)>) {
    let targets: Vec<i32> = (0..r as i32).collect();
    let inp = Inputs {
        m1: CsrView::new("m1", &m1.indptr, &m1.indices, &m1.data, m).unwrap(),
        m2: CsrView::new("m2", &m2.indptr, &m2.indices, &m2.data, n).unwrap(),
        targets: &targets,
        norms: Norms::default(),
        params: Params::default(),
        filter: None,
        target_cols: None,
        col_map,
    };
    let out = compute(&inp, k, block, &|| {}).unwrap();
    let got = out
        .cols
        .iter()
        .zip(&out.values)
        .map(|(&c, &v)| (c, v.to_bits()))
        .collect();
    (out.counts, got)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 1000 }))]
    // m2 with columns renumbered (kernel col j = original col map[j]) + col_map
    // gives the same packed rows as the original m2. k = n keeps every column,
    // so the tie rule (on kernel ids) can't pick different ones.
    #[test]
    fn col_map_restores_original_ids(
        (m1, m2, n, map, block) in (1usize..6, 1usize..8, 2usize..40).prop_flat_map(|(r, m, n)| {
            let ids: Vec<i32> = (0..n as i32).collect();
            (dense(r, m), dense(m, n), Just(n), Just(ids).prop_shuffle(), 0..n)
        })
    ) {
        let permuted: Vec<Vec<f32>> = m2
            .iter()
            .map(|row| map.iter().map(|&o| row[o as usize]).collect())
            .collect();
        let (a, b, bp) = (to_csr(&m1), to_csr(&m2), to_csr(&permuted));
        let m = m2.len();
        prop_assert_eq!(
            top_scores(&a, &bp, m, n, m1.len(), n, block, Some(&map)),
            top_scores(&a, &b, m, n, m1.len(), n, block, None)
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 1000 }))]
    // Non-integer values make f32 sums order-dependent: blocking must add each
    // column's terms in the same order as the single-block path (bit-equal).
    #[test]
    fn blocked_matches_single_block_bitwise(
        (m1, m2, n, block) in (1usize..6, 1usize..8, 2usize..40).prop_flat_map(|(r, m, n)| {
            let cell = prop_oneof![2 => Just(0.0f32), 1 => 0.1f32..3.0];
            let dense = |a, b| prop::collection::vec(prop::collection::vec(cell.clone(), b), a);
            (dense(r, m), dense(m, n), Just(n), 1..n)
        })
    ) {
        let (a, b) = (to_csr(&m1), to_csr(&m2));
        let m = m2.len();
        prop_assert_eq!(
            top_scores(&a, &b, m, n, m1.len(), n, block, None),
            top_scores(&a, &b, m, n, m1.len(), n, 0, None)
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 1000 }))]
    // Small integers make ties common; with k < n the tie rule (lower column
    // wins) decides which tied columns stay, so blocked == single-block only
    // if the result is independent of the order columns reach the heap (F4).
    #[test]
    fn blocked_matches_single_block_with_ties(
        (m1, m2, n, k, block) in (1usize..6, 1usize..8, 2usize..40).prop_flat_map(|(r, m, n)| {
            (dense(r, m), dense(m, n), Just(n), 1..n, 1..n)
        })
    ) {
        let (a, b) = (to_csr(&m1), to_csr(&m2));
        let m = m2.len();
        prop_assert_eq!(
            top_scores(&a, &b, m, n, m1.len(), k, block, None),
            top_scores(&a, &b, m, n, m1.len(), k, 0, None)
        );
    }
}
