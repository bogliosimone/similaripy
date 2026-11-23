//! Property test: the kernel against a naive dense f64 reference.
//! Scores must match within rtol; the kept columns must be a valid top-K
//! (any column whose reference score ties the k-th within tolerance may swap).

use proptest::prelude::*;
use similaripy_core::{CsrPattern, CsrView, Inputs, Norms, Params, compute};

const RTOL: f64 = 1e-5;
const ATOL: f64 = 1e-6;

struct Csr {
    indptr: Vec<i64>,
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
        m.indptr.push(m.indices.len() as i64);
    }
    m
}

fn dense(rows: usize, cols: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
    // Small integers keep every dot product exact in f32; negative ones (signed
    // input) make sums cancel to exactly 0.0 (re-pushed columns, dropped
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
        // Positive float ranges start at 0.125, a multiple of ulp(3.0): from 0.1,
        // proptest 1.12's sampler can overshoot `low` and trip its own (inverted)
        // debug_assert, ~1 in 4e7 draws in debug builds.
        let pos = |len| prop::collection::vec(0.125f32..3.0, len);
        let weight = prop_oneof![Just(0.0f32), 0.125f32..2.0];
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

/// Independent f64 version of the s_plus scoring formula. Returns the score and
/// its absolute tolerance: the f32 kernel loses about `eps * cond` relative
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
            x_tversky: Some(&c.norms[0]),
            y_tversky: Some(&c.norms[1]),
            x_cosine: Some(&c.norms[2]),
            y_cosine: Some(&c.norms[3]),
            x_depop: Some(&c.norms[4]),
            y_depop: Some(&c.norms[5]),
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
    // Unsorted m2 rows only reject on the blocked path, which walks each row by column.
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

// Output column ids are i32: wider m2 is rejected before any allocation.
// The selection bitsets are m2.n_cols wide: a wider selection would index past them.
// Every `validate` error but the slot-count overflow: these guard the unchecked
// row accesses for direct core callers (the Python layer never builds such
// inputs). The overflow can't happen on 64-bit: `compute` clamps k to n_cols
// first, so len(targets) * k < 2^62; it guards 32-bit targets.
#[test]
fn rejects_each_invalid_input() {
    // m1: 2 x 2, m2: 2 x 3 with sorted rows.
    let (i1, c1, d1) = ([0, 1, 2], [0, 1], [1.0, 1.0]);
    let (i2, c2, d2) = ([0, 2, 3], [0, 2, 1], [1.0, 1.0, 1.0]);
    let one_row = ([0, 0], []);
    let m1 = CsrView::new("m1", &i1, &c1, &d1, 2).unwrap();
    let m1_wide = CsrView::new("m1", &i1, &c1, &d1, 3).unwrap();
    let m2 = CsrView::new("m2", &i2, &c2, &d2, 3).unwrap();
    let m2_too_wide = CsrView::new("m2", &i2, &c2, &d2, i32::MAX as usize + 1).unwrap();
    let short_sel = CsrPattern::new("filter", &one_row.0, &one_row.1, 3).unwrap();
    let wide_sel_rows = ([0, 1, 1], [4]);
    let wide_sel = CsrPattern::new("filter", &wide_sel_rows.0, &wide_sel_rows.1, 5).unwrap();
    let base = Inputs {
        m1,
        m2,
        targets: &[0, 1],
        norms: Norms::default(),
        params: Params::default(),
        filter: None,
        target_cols: None,
        col_map: None,
    };
    let err = |inp: Inputs, k: usize| compute(&inp, k, 0, &|| {}).err().unwrap().to_string();
    assert!(compute(&base, 2, 0, &|| {}).is_ok());
    let cases: [(Inputs, usize, &str); 10] = [
        (
            Inputs {
                m1: m1_wide,
                ..base
            },
            2,
            "m1 columns != m2 rows",
        ),
        (
            Inputs {
                targets: &[-1],
                ..base
            },
            2,
            "negative target row",
        ),
        (
            Inputs {
                targets: &[2],
                ..base
            },
            2,
            "target row out of range",
        ),
        (
            Inputs {
                col_map: Some(&[0, 1]),
                ..base
            },
            2,
            "col_map length",
        ),
        (
            Inputs {
                params: Params {
                    l2: 1.0,
                    ..Params::default()
                },
                ..base
            },
            2,
            "cosine norms missing",
        ),
        (
            Inputs {
                norms: Norms {
                    x_cosine: Some(&[1.0, 1.0]),
                    ..Norms::default()
                },
                params: Params {
                    l2: 1.0,
                    ..Params::default()
                },
                ..base
            },
            2,
            "cosine norms missing",
        ),
        (
            Inputs {
                norms: Norms {
                    x_cosine: Some(&[1.0, 1.0]),
                    y_cosine: Some(&[1.0, 1.0]), // 3 columns
                    ..Norms::default()
                },
                params: Params {
                    l2: 1.0,
                    ..Params::default()
                },
                ..base
            },
            2,
            "cosine norms too short",
        ),
        (
            Inputs {
                filter: Some(short_sel),
                ..base
            },
            2,
            "fewer rows than targets need",
        ),
        (
            Inputs {
                filter: Some(wide_sel),
                ..base
            },
            2,
            "column selection columns != m2 columns",
        ),
        (
            Inputs {
                m2: m2_too_wide,
                ..base
            },
            2,
            "more than i32::MAX output columns",
        ),
    ];
    for (inp, k, msg) in cases {
        let e = err(inp, k);
        assert!(e.contains(msg), "{e:?} should contain {msg:?}");
    }
}

// The blocked path walks each m2 row front to back, block by block, so it needs
// sorted rows: unsorted rows are rejected when blocked, accepted when not.
#[test]
fn rejects_unsorted_m2_only_when_blocked() {
    let (i1, c1, d1) = ([0, 2], [0, 1], [1.0, 2.0]);
    let (i2, c2, d2) = ([0, 2, 3], [2, 0, 1], [1.0, 3.0, 1.0]);
    let inp = Inputs {
        m1: CsrView::new("m1", &i1, &c1, &d1, 2).unwrap(),
        m2: CsrView::new("m2", &i2, &c2, &d2, 3).unwrap(),
        targets: &[0],
        norms: Norms::default(),
        params: Params::default(),
        filter: None,
        target_cols: None,
        col_map: None,
    };
    let e = compute(&inp, 3, 1, &|| {}).err().unwrap().to_string();
    assert!(e.contains("indices not sorted within rows"), "{e:?}");
    let out = compute(&inp, 3, 0, &|| {}).unwrap();
    assert_eq!((out.cols, out.values), (vec![0, 1, 2], vec![3.0, 2.0, 1.0]));
}

#[test]
fn k_above_n_cols_gives_the_same_rows() {
    let (m1, m2) = (
        to_csr(&[vec![1.0, 2.0]]),
        to_csr(&[vec![1.0, 0.0, 3.0], vec![0.5, 1.0, 0.0]]),
    );
    let want = top_scores(&m1, &m2, 2, 3, 1, 3, 0, None);
    assert_eq!(top_scores(&m1, &m2, 2, 3, 1, 1000, 0, None), want);
    assert_eq!(want.0, vec![3]);
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
            let cell = prop_oneof![2 => Just(0.0f32), 1 => 0.125f32..3.0];
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
    // if the result is independent of the order columns reach the top-K.
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

// ---- The similarity driver (`similarity::similarity`) against a dense f64
// reference of the whole pipeline: binary, random walk, norms, selectors,
// target rows, every column order and block size. ----

mod driver {
    use super::{dense, to_csr};
    use proptest::prelude::*;
    use similaripy_core::Params;
    use similaripy_core::block::BlockSize;
    use similaripy_core::prep::Csr as Owned;
    use similaripy_core::similarity::{ColumnOrder, Options, Pop, Selector, Walk, similarity};

    const RTOL: f64 = 1e-4;
    const EPS: f64 = f32::EPSILON as f64;

    #[derive(Debug, Clone)]
    enum Sel {
        Ids(Vec<i64>),
        Matrix(Vec<Vec<f32>>),
    }

    #[derive(Debug, Clone)]
    enum PopSpec {
        None,
        Sum,
        Weights(Vec<f32>),
    }

    /// One similarity's options, as the Python wrappers set them.
    #[derive(Debug, Clone)]
    struct Config {
        name: &'static str,
        params: P,
        c: (f32, f32),
        p: (f32, f32),
        additive: f32,
        pop: (PopSpec, PopSpec),
        walk: Option<Walk>,
    }

    /// `Params` printed on failure (the kernel type has no Debug).
    #[derive(Clone, Copy)]
    struct P(Params);

    impl std::fmt::Debug for P {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let p = &self.0;
            write!(
                f,
                "a1 {} l {} {} {} t {} {} shrink s {} b {} thr {}",
                p.a1,
                p.l1,
                p.l2,
                p.l3,
                p.t1,
                p.t2,
                p.stabilized_shrink,
                p.bayesian_shrink,
                p.threshold
            )
        }
    }

    #[derive(Debug)]
    struct Case {
        m1: Vec<Vec<f32>>,
        m2: Vec<Vec<f32>>,
        cfg: Config,
        binary: bool,
        k: usize,
        targets: Option<Vec<i64>>,
        filter: Option<Sel>,
        target: Option<Sel>,
        block: BlockSize,
        order: ColumnOrder,
    }

    fn pop(n: usize) -> impl Strategy<Value = PopSpec> {
        prop_oneof![
            Just(PopSpec::None),
            Just(PopSpec::Sum),
            prop::collection::vec(0.5f32..2.0, n).prop_map(PopSpec::Weights),
        ]
    }

    fn config(r: usize, n: usize, signed: bool) -> impl Strategy<Value = Config> {
        let w = || prop_oneof![Just(0.0f32), 0.125f32..2.0];
        let frac = 0.125f32..0.875;
        (
            0usize..9,
            (w(), w(), w(), w(), w(), w()),
            (frac.clone(), 0.125f32..1.5, 0.125f32..1.5),
            (prop_oneof![Just(1.0f32), Just(0.5)], 0.0f32..1.0),
            (pop(r), pop(n)),
        )
            .prop_map(
                move |(kind, (l1, l2, l3, ss, bs, add), (a, c1, c2), (pw, thr), pops)| {
                    let mut p = Params::default();
                    let mut cfg = Config {
                        name: "",
                        params: P(p),
                        c: (0.5, 0.5),
                        p: (0.0, 0.0),
                        additive: 0.0,
                        pop: (PopSpec::None, PopSpec::None),
                        walk: None,
                    };
                    p.threshold = if thr < 0.3 { 0.0 } else { thr };
                    match kind {
                        0 => cfg.name = "dot_product",
                        1 => {
                            cfg.name = "cosine";
                            (p.l2, cfg.additive) = (1.0, add);
                        }
                        2 => {
                            cfg.name = "asymmetric_cosine";
                            p.l2 = 1.0;
                            cfg.c = (a, 1.0 - a);
                        }
                        3 => {
                            cfg.name = "tversky";
                            (p.l1, p.t1, p.t2) = (1.0, a, 1.0 - a);
                        }
                        4 => {
                            cfg.name = "jaccard";
                            p.l1 = 1.0;
                        }
                        5 => {
                            cfg.name = "dice";
                            (p.l1, p.t1, p.t2) = (1.0, 0.5, 0.5);
                        }
                        6 => {
                            cfg.name = "p3alpha";
                            cfg.walk = Some(Walk {
                                alpha: pw,
                                popularity: false,
                            });
                        }
                        7 => {
                            cfg.name = "rp3beta";
                            p.l3 = 1.0;
                            cfg.p = (0.0, c2);
                            cfg.walk = Some(Walk {
                                alpha: pw,
                                popularity: true,
                            });
                        }
                        _ => {
                            cfg.name = "s_plus";
                            (p.l1, p.l2, p.l3, p.stabilized_shrink, p.bayesian_shrink) =
                                (l1, l2, l3, ss, bs);
                            p.a1 = pw;
                            cfg.c = (c1, c2);
                            cfg.p = (c1, c2);
                            cfg.additive = add;
                            cfg.pop = pops;
                        }
                    }
                    if signed {
                        (p.a1, p.bayesian_shrink) = (1.0, 0.0);
                    }
                    cfg.params = P(p);
                    cfg
                },
            )
    }

    fn sel(r: usize, n: usize, any_empty: bool) -> impl Strategy<Value = Option<Sel>> {
        let ids = prop::collection::vec(0..n as i64, if any_empty { 0 } else { 1 }..=n);
        let mask = prop::collection::vec(
            prop::collection::vec(prop_oneof![Just(0.0f32), Just(1.0f32), Just(-1.0f32)], n),
            r,
        );
        prop::option::of(prop_oneof![
            ids.prop_map(Sel::Ids),
            mask.prop_map(Sel::Matrix)
        ])
    }

    fn case() -> impl Strategy<Value = Case> {
        (1usize..6, 1usize..6, 1usize..9, any::<bool>()).prop_flat_map(|(r, m, n, signed)| {
            let abs = move |d: Vec<Vec<f32>>| {
                d.into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|v| if signed { v } else { v.abs() })
                            .collect()
                    })
                    .collect::<Vec<Vec<f32>>>()
            };
            (
                dense(r, m).prop_map(abs),
                dense(m, n).prop_map(abs),
                config(r, n, signed),
                any::<bool>(),
                1..=n,
                prop::option::of(prop::collection::vec(0..r as i64, 0..=2 * r)),
                (sel(r, n, true), sel(r, n, false)),
                prop_oneof![
                    Just(BlockSize::Off),
                    Just(BlockSize::Auto),
                    (1..=n).prop_map(BlockSize::Size)
                ],
                prop_oneof![Just(ColumnOrder::Popularity), Just(ColumnOrder::Original)],
            )
                .prop_map(
                    |(m1, m2, cfg, binary, k, targets, (filter, target), block, order)| Case {
                        m1,
                        m2,
                        cfg,
                        binary,
                        k,
                        targets,
                        filter,
                        target,
                        block,
                        order,
                    },
                )
        })
    }

    fn owned(d: &[Vec<f32>]) -> Owned<f32> {
        let c = to_csr(d);
        Owned {
            indptr: c.indptr,
            indices: c.indices,
            data: c.data,
        }
    }

    fn pop_of(p: &PopSpec) -> Pop<'_> {
        match p {
            PopSpec::None => Pop::None,
            PopSpec::Sum => Pop::Sum,
            PopSpec::Weights(w) => Pop::Weights(w),
        }
    }

    fn selector(s: &Option<Sel>) -> Option<Selector<'_>> {
        s.as_ref().map(|s| match s {
            Sel::Ids(ids) => Selector::Ids(ids),
            Sel::Matrix(m) => Selector::Matrix(owned(m)),
        })
    }

    /// Column `c` of row `t` selected by a selector (None = no constraint).
    fn selected(s: &Option<Sel>, t: usize, c: usize) -> Option<bool> {
        s.as_ref().map(|s| match s {
            Sel::Ids(ids) => ids.contains(&(c as i64)),
            Sel::Matrix(m) => m[t][c] != 0.0,
        })
    }

    /// The pipeline in f64 on dense matrices: values after binary + walk, and
    /// the x / y terms [tversky, cosine, depop].
    type Dense = Vec<Vec<f64>>;

    fn prepare(c: &Case) -> (Dense, Dense, [Vec<f64>; 3], [Vec<f64>; 3]) {
        let f64s = |d: &[Vec<f32>]| -> Vec<Vec<f64>> {
            d.iter()
                .map(|r| {
                    r.iter()
                        .map(|&v| {
                            if c.binary && v != 0.0 {
                                1.0
                            } else {
                                f64::from(v)
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let (mut a, mut b) = (f64s(&c.m1), f64s(&c.m2));
        let (m, n) = (b.len(), b[0].len());
        let col_abs = |b: &[Vec<f64>]| -> Vec<f64> {
            (0..n)
                .map(|j| (0..m).map(|u| b[u][j].abs()).sum())
                .collect()
        };
        let mut walk_pop = None;
        if let Some(w) = c.cfg.walk {
            if w.popularity {
                walk_pop = Some(col_abs(&b));
            }
            for row in a.iter_mut().chain(b.iter_mut()) {
                let s: f64 = row.iter().map(|x| x.abs()).sum();
                for x in row.iter_mut() {
                    if s != 0.0 {
                        *x /= s;
                    }
                    *x = x.abs().powf(f64::from(w.alpha)).copysign(*x);
                }
            }
        }
        let (c1, c2) = (f64::from(c.cfg.c.0), f64::from(c.cfg.c.1));
        let (p1, p2) = (f64::from(c.cfg.p.0), f64::from(c.cfg.p.1));
        let add = f64::from(c.cfg.additive);
        let xsq: Vec<f64> = a
            .iter()
            .map(|row| row.iter().map(|x| x * x).sum())
            .collect();
        let ysq: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|u| b[u][j] * b[u][j]).sum())
            .collect();
        let depop = |p: &PopSpec, sums: Vec<f64>, e: f64| -> Vec<f64> {
            match p {
                PopSpec::None => vec![1.0; sums.len()],
                PopSpec::Sum => sums.iter().map(|s| s.powf(e)).collect(),
                PopSpec::Weights(w) => w.iter().map(|&w| f64::from(w).powf(e)).collect(),
            }
        };
        let xd = depop(
            &c.cfg.pop.0,
            a.iter()
                .map(|row| row.iter().map(|x| x.abs()).sum())
                .collect(),
            p1,
        );
        let yd = match walk_pop {
            Some(s) => s.iter().map(|s| s.powf(p2)).collect(),
            None => depop(&c.cfg.pop.1, col_abs(&b), p2),
        };
        let x = [
            xsq.clone(),
            xsq.iter().map(|s| (s + add).powf(c1)).collect(),
            xd,
        ];
        let y = [
            ysq.clone(),
            ysq.iter().map(|s| (s + add).powf(c2)).collect(),
            yd,
        ];
        (a, b, x, y)
    }

    /// A reference score and its tolerance; None: too close to 0 to judge
    /// (the f32 sum may land on either side).
    fn score(
        p: &Params,
        x: [f64; 3],
        y: [f64; 3],
        xy: f64,
        xy_abs: f64,
        m: usize,
    ) -> Option<(f64, f64)> {
        let tol_xy = 64.0 * EPS * (m as f64 + 4.0) * xy_abs;
        if xy.abs() <= tol_xy {
            return None;
        }
        let [a1, l1, l2, l3, t1, t2, ss, bs] = [
            p.a1,
            p.l1,
            p.l2,
            p.l3,
            p.t1,
            p.t2,
            p.stabilized_shrink,
            p.bayesian_shrink,
        ]
        .map(f64::from);
        let pw = xy.powf(a1);
        let tv = l1 * (t1 * (x[0] - xy) + t2 * (y[0] - xy) + xy);
        let (cs, dp) = (l2 * x[1] * y[1], l3 * x[2] * y[2]);
        let den = tv + cs + dp + ss;
        let has_den = l1 != 0.0 || l2 != 0.0 || l3 != 0.0 || ss != 0.0;
        let mut v = if !has_den {
            pw
        } else if den != 0.0 {
            pw / den
        } else {
            0.0
        };
        if bs != 0.0 {
            v *= pw / (pw + bs);
        }
        let terms = l1 * (t1 * (x[0] + xy.abs()) + t2 * (y[0] + xy.abs()) + xy.abs())
            + cs.abs()
            + dp.abs()
            + ss;
        let cond = if has_den && den != 0.0 {
            terms / den.abs()
        } else {
            1.0
        };
        let rel = RTOL + 64.0 * EPS * cond + (2.0 * a1 + 1.0 + cond) * tol_xy / xy.abs();
        Some((v, 1e-6 + v.abs() * rel))
    }

    fn check(c: &Case) -> Result<(), TestCaseError> {
        let (r, n) = (c.m1.len(), c.m2[0].len());
        let p = c.cfg.params.0;
        let opts = Options {
            k: c.k,
            params: p,
            c1: c.cfg.c.0,
            c2: c.cfg.c.1,
            p1: c.cfg.p.0,
            p2: c.cfg.p.1,
            additive_shrink: c.cfg.additive,
            pop1: pop_of(&c.cfg.pop.0),
            pop2: pop_of(&c.cfg.pop.1),
            binary: c.binary,
            walk: c.cfg.walk,
            target_rows: c.targets.as_deref(),
            filter_cols: selector(&c.filter),
            target_cols: selector(&c.target),
            block_size: c.block,
            column_order: c.order,
        };
        let res = similarity(owned(&c.m1), owned(&c.m2), n, opts, &|| {}, |_| {});
        let selects_nothing = match &c.target {
            Some(Sel::Matrix(m)) => m.iter().flatten().all(|&v| v == 0.0),
            _ => false,
        };
        if selects_nothing {
            let e = res.err().map(|e| e.to_string()).unwrap_or_default();
            prop_assert!(e.contains("target_cols is empty"), "{e:?}");
            return Ok(());
        }
        let out = res.map_err(|e| TestCaseError::fail(format!("{}: {e}", c.cfg.name)))?;
        let csr = &out.csr;
        prop_assert_eq!(csr.indptr.len(), r + 1);
        let (a, b, x, y) = prepare(c);
        let m = b.len();
        let thr = f64::from(p.threshold);
        for t in 0..r {
            let got: Vec<(usize, f64)> = (csr.indptr[t] as usize..csr.indptr[t + 1] as usize)
                .map(|j| (csr.indices[j] as usize, f64::from(csr.data[j])))
                .collect();
            let targeted = c.targets.as_ref().is_none_or(|ts| ts.contains(&(t as i64)));
            if !targeted {
                prop_assert!(got.is_empty(), "{}: row {t} not targeted", c.cfg.name);
                continue;
            }
            prop_assert!(
                got.windows(2).all(|w| w[0].0 < w[1].0),
                "row {t}: not sorted"
            );
            // per eligible column: Some((score, tol)) or None (score ~ 0)
            let mut ref_of = vec![None; n];
            let (mut sure, mut maybe) = (vec![], vec![]);
            for col in 0..n {
                let touched = (0..m).any(|u| a[t][u] != 0.0 && b[u][col] != 0.0);
                let ok = selected(&c.filter, t, col) != Some(true)
                    && selected(&c.target, t, col) != Some(false);
                if !touched || !ok {
                    continue;
                }
                let xy: f64 = (0..m).map(|u| a[t][u] * b[u][col]).sum();
                let xy_abs: f64 = (0..m).map(|u| (a[t][u] * b[u][col]).abs()).sum();
                let s = score(
                    &p,
                    [x[0][t], x[1][t], x[2][t]],
                    [y[0][col], y[1][col], y[2][col]],
                    xy,
                    xy_abs,
                    m,
                );
                ref_of[col] = Some(s);
                match s {
                    None => maybe.push(col),
                    Some((v, tol)) if v > thr.max(0.0) + tol => sure.push(col),
                    Some((v, tol)) if (v - thr).abs() <= tol || v.abs() <= tol => maybe.push(col),
                    Some(_) => {}
                }
            }
            let name = c.cfg.name;
            let (lo, hi) = (c.k.min(sure.len()), c.k.min(sure.len() + maybe.len()));
            prop_assert!(
                (lo..=hi).contains(&got.len()),
                "{name} row {t}: {} results, want {lo}..={hi}",
                got.len()
            );
            let value = |col: usize| ref_of[col].flatten().unwrap_or((0.0, 0.0));
            for &(col, v) in &got {
                prop_assert!(
                    sure.contains(&col) || maybe.contains(&col),
                    "{name} row {t}: col {col} not eligible ({v})"
                );
                prop_assert!(
                    v > 0.0 && v >= thr - 1e-9,
                    "{name} row {t}: col {col} = {v} below threshold"
                );
                if let Some(Some((want, tol))) = ref_of[col] {
                    prop_assert!(
                        (v - want).abs() <= tol,
                        "{name} row {t} col {col}: {v} vs {want} (tol {tol})"
                    );
                }
            }
            for &s in sure.iter().filter(|s| !got.iter().any(|g| g.0 == **s)) {
                prop_assert!(
                    got.len() == c.k,
                    "{name} row {t}: sure col {s} missing with room left"
                );
                let (vs, ts) = value(s);
                for &(o, _) in &got {
                    let (vo, to) = value(o);
                    prop_assert!(
                        vs <= vo + ts + to,
                        "{name} row {t}: col {s} ({vs}) lost to col {o} ({vo})"
                    );
                }
            }
        }
        Ok(())
    }

    proptest! {
        // Miri runs ~1000x slower: a few cases still go through every step.
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 2000 }))]
        #[test]
        fn similarity_matches_dense_reference(c in case()) {
            check(&c)?;
        }
    }

    fn csr(rows: &[&[f32]]) -> Owned<f32> {
        owned(&rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
    }

    fn err(m1: Owned<f32>, m2: Owned<f32>, n: usize, o: Options) -> String {
        match similarity(m1, m2, n, o, &|| {}, |_| {}) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    // Finite is checked before binary and the walk: p3alpha(binary=True)
    // no longer turns NaN into 1.0.
    #[test]
    fn non_finite_rejected_before_binary_and_walk() {
        for walk in [
            None,
            Some(Walk {
                alpha: 1.0,
                popularity: false,
            }),
        ] {
            let o = Options {
                binary: true,
                walk,
                ..Options::default()
            };
            let e = err(csr(&[&[f32::NAN, 1.0]]), csr(&[&[1.0], &[1.0]]), 1, o);
            assert!(e.contains("matrix1 values must be finite"), "{e}");
        }
    }

    #[test]
    fn phases_rows_and_kernel_time() {
        let phases = std::sync::Mutex::new(vec![]);
        let rows = std::sync::atomic::AtomicUsize::new(0);
        let o = Options {
            target_rows: Some(&[1, 1]),
            ..Options::default()
        };
        let out = similarity(
            csr(&[&[1.0, 0.0], &[0.0, 2.0]]),
            csr(&[&[1.0, 1.0], &[3.0, 0.0]]),
            2,
            o,
            &|| {
                rows.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            },
            |p| phases.lock().unwrap().push(p),
        )
        .unwrap();
        use similaripy_core::similarity::Phase;
        assert_eq!(
            *phases.lock().unwrap(),
            [Phase::Prep, Phase::Compute { rows: 1 }]
        );
        assert_eq!(rows.into_inner(), 1, "duplicate target rows computed once");
        assert_eq!(out.csr.indptr, [0, 0, 1]);
        assert_eq!((out.csr.indices, out.csr.data), (vec![0], vec![6.0]));
        assert!(out.kernel_seconds >= 0.0);
    }

    // binary + a walk: the values after the walk are probabilities, so the
    // tversky rule still applies (x = 1, y = 0.25: den = 2 * 0.0625 - 0.25 < 0).
    #[test]
    fn tversky_rule_applies_after_a_binary_walk() {
        let o = Options {
            binary: true,
            walk: Some(Walk {
                alpha: 1.0,
                popularity: false,
            }),
            params: Params {
                l1: 1.0,
                t1: 0.0,
                t2: 2.0,
                ..Params::default()
            },
            ..Options::default()
        };
        let m2 = csr(&[&[2.0, 3.0, 4.0, 5.0]]);
        let e = err(csr(&[&[7.0]]), m2, 4, o);
        assert!(e.contains("denominator <= 0 on some pairs"), "{e}");
    }

    #[test]
    fn signed_and_shape_rules_reach_the_caller() {
        let signed = || csr(&[&[1.0, -1.0]]);
        let o = Options {
            params: Params {
                a1: 0.5,
                ..Params::default()
            },
            ..Options::default()
        };
        assert!(
            err(signed(), csr(&[&[1.0], &[1.0]]), 1, o).contains("a1 (alpha) must be 1 on signed")
        );
        let o = Options {
            target_rows: Some(&[1]),
            ..Options::default()
        };
        assert!(
            err(signed(), csr(&[&[1.0], &[1.0]]), 1, o)
                .contains("target_rows ids must be in [0, 1)")
        );
        let o = Options {
            target_cols: Some(Selector::Ids(&[])),
            ..Options::default()
        };
        assert!(err(signed(), csr(&[&[1.0], &[1.0]]), 1, o).contains("target_cols is empty"));
        // m1 column 1 has no m2 row
        let e = err(signed(), csr(&[&[1.0]]), 1, Options::default());
        assert!(e.contains("m1"), "{e}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 500 }))]
    // Depop-only weights (rp3beta) through the cosine slots (the driver's
    // `depop_as_cosine`: COSINE mode instead of FULL) give the same bits, with
    // a shrink, a power and zero weights (empty rows / columns), blocked or not.
    #[test]
    fn depop_in_cosine_slots_same_bits(
        (m1, m2, x, y, block) in (1usize..6, 1usize..8, 2usize..30).prop_flat_map(|(r, m, n)| {
            let w = |len| prop::collection::vec(prop_oneof![Just(0.0f32), 0.01f32..4.0], len);
            (dense(r, m), dense(m, n), w(r), w(n), 0..n)
        }),
        l in 0.1f32..2.0,
        a1 in prop_oneof![Just(1.0f32), 0.3f32..2.0],
        (stabilized, bayesian) in prop_oneof![Just((0.0f32, 0.0f32)), (0.1f32..5.0).prop_map(|s| (s, 0.0)), (0.1f32..5.0).prop_map(|b| (0.0, b))],
    ) {
        // Not under Miri: it adds random error to powf, so the two runs' powers differ there.
        let a1 = if cfg!(miri) { 1.0 } else { a1 };
        let (a, b) = (to_csr(&m1), to_csr(&m2));
        let (m, n) = (m2.len(), y.len());
        let targets: Vec<i32> = (0..m1.len() as i32).collect();
        let run = |depop: bool| {
            let p = Params { a1, stabilized_shrink: stabilized, bayesian_shrink: bayesian, ..Params::default() };
            let (params, norms) = if depop {
                (Params { l3: l, ..p }, Norms { x_depop: Some(&x), y_depop: Some(&y), ..Norms::default() })
            } else {
                (Params { l2: l, ..p }, Norms { x_cosine: Some(&x), y_cosine: Some(&y), ..Norms::default() })
            };
            let inp = Inputs {
                m1: CsrView::new("m1", &a.indptr, &a.indices, &a.data, m).unwrap(),
                m2: CsrView::new("m2", &b.indptr, &b.indices, &b.data, n).unwrap(),
                targets: &targets,
                norms,
                params,
                filter: None,
                target_cols: None,
                col_map: None,
            };
            let out = compute(&inp, n, block, &|| {}).unwrap();
            let bits: Vec<u32> = out.values.iter().map(|v| v.to_bits()).collect();
            (out.counts, out.cols, bits)
        };
        prop_assert_eq!(run(true), run(false));
    }
}
