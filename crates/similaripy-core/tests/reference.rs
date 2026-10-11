//! Property test: the kernel against a naive dense f64 reference.
//! Scores must match within rtol; the kept columns must be a valid top-K
//! (any column whose reference score ties the k-th within tolerance may swap).

use proptest::prelude::*;
use similaripy_core::{CsrPattern, CsrView, Inputs, Norms, Output, Params, Schedule, compute};

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
    // Small integers keep every dot product exact in f32.
    let cell = prop_oneof![3 => Just(0.0f32), 2 => (1u8..4).prop_map(f32::from)];
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
    filter: Option<Vec<Vec<f32>>>,
    target: Option<Vec<Vec<f32>>>,
    k: usize,
    schedule: Schedule,
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
            prop_oneof![Just(Schedule::Adaptive), Just(Schedule::Dynamic)],
            prop::collection::vec(0..r, 1..=2 * r),
            any::<bool>(),
            prop_oneof![Just(0usize), 1..=n],
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
                    schedule,
                    targets,
                    unsorted_m2,
                    block_size,
                )| Case {
                    m1,
                    m2,
                    norms,
                    params: [p.0, p.1, p.2, p.3, p.4, p.5, p.6, p.7],
                    filter,
                    target,
                    k,
                    schedule,
                    targets,
                    unsorted_m2,
                    block_size,
                },
            )
    })
}

/// Independent f64 version of the 0.7.0 scoring formula. Returns the score
/// and its absolute tolerance: the f32 kernel loses about `eps * cond` relative
/// accuracy when the denominator terms cancel (cond = sum|terms| / |den|).
fn ref_score(c: &Case, row: usize, col: usize, xy: f64) -> (f64, f64) {
    let [a1, l1, l2, l3, t1, t2, ss, bs] = c.params.map(f64::from);
    let n = |i: usize, j: usize| f64::from(c.norms[i][j]);
    if l1 == 0.0 && l2 == 0.0 && l3 == 0.0 && ss == 0.0 && bs == 0.0 {
        return (xy, ATOL + RTOL * xy); // no normalization: a1 is ignored, as in 0.7.0
    }
    let tv = l1 * (t1 * (n(0, row) - xy) + t2 * (n(1, col) - xy) + xy);
    let cs = l2 * n(2, row) * n(3, col);
    let dp = l3 * n(4, row) * n(5, col);
    let p = xy.powf(a1);
    let den = tv + cs + dp + ss;
    let mut v = if den != 0.0 { p / den } else { 0.0 };
    if bs != 0.0 {
        v *= p / (p + bs);
    }
    let terms =
        l1 * (t1 * (n(0, row) + xy) + t2 * (n(1, col) + xy) + xy) + cs.abs() + dp.abs() + ss;
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
            threshold: 0.0,
        },
        filter: pattern(&filter, n),
        target_cols: pattern(&target, n),
    };
    let k = c.k;
    let (mut rows, mut cols, mut vals) = (vec![0; r * k], vec![0; r * k], vec![0.0; r * k]);
    let out = Output {
        rows: &mut rows,
        cols: &mut cols,
        values: &mut vals,
    };
    let res = compute(&inp, k, c.block_size, c.schedule, out);
    let blocked = c.block_size > 0 && n > c.block_size;
    // Unsorted m2 rows only reject when the blocked path would binary-search them.
    if blocked && c.unsorted_m2 && m2.indptr.windows(2).any(|w| w[1] - w[0] > 1) {
        prop_assert!(res.is_err(), "unsorted m2 accepted on the blocked path");
        return Ok(());
    }
    res.unwrap();

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
            if touched && !filtered && !untargeted && score >= 0.0 {
                eligible.push((col, score, tol)); // threshold 0
            }
        }
        let expected = k.min(eligible.len());
        let mut sorted: Vec<f64> = eligible.iter().map(|e| e.1).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let span = i * k..(i + 1) * k;
        let got: Vec<(i32, f32)> = cols[span.clone()]
            .iter()
            .copied()
            .zip(vals[span.clone()].iter().copied())
            .collect();
        for (j, &(col, score)) in got.iter().enumerate() {
            if j >= expected {
                prop_assert_eq!(
                    (rows[i * k + j], col, score),
                    (0, 0, 0.0),
                    "padding touched"
                );
                continue;
            }
            prop_assert_eq!(rows[i * k + j], t as i32);
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
        let mut kept: Vec<i32> = got[..expected].iter().map(|g| g.0).collect();
        kept.sort();
        kept.dedup();
        prop_assert_eq!(kept.len(), expected, "row {}: duplicate columns", t);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]
    #[test]
    fn matches_dense_reference(c in case()) {
        check(&c)?;
    }
}
