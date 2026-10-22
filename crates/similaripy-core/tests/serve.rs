//! Property tests: serving against a naive dense f64 reference, every variant.
//! Small integers keep every sum exact, so results must match exactly; float
//! values check the bit-identity claims (blocked == unblocked, batch == single,
//! history order and duplicates don't matter).

use proptest::prelude::*;
use similaripy_core::serve::{
    Lookup, Mask, Model, Request, Seen, Variant, recommend, recommend_batch,
};

#[derive(Debug, Clone)]
struct Case {
    /// S in internal ids (row = col).
    s: Vec<Vec<f32>>,
    new_to_old: Vec<i32>,
    ext: Vec<i64>,
    /// Indices into ext (known) or `None` (an unknown id).
    hist: Vec<Option<usize>>,
    weights: Option<Vec<f32>>,
    k: usize,
    exclude_seen: bool,
    /// Allowed flag per internal id.
    allowed: Option<Vec<bool>>,
}

fn csr(s: &[Vec<f32>]) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let (mut indptr, mut indices, mut data) = (vec![0], vec![], vec![]);
    for row in s {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                indices.push(c as i32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as i32);
    }
    (indptr, indices, data)
}

fn model(c: &Case) -> Model {
    let (indptr, indices, data) = csr(&c.s);
    Model::new(indptr, indices, data, &c.new_to_old, &c.ext).unwrap()
}

/// External id of original column `i`, or one no item has: past the end for
/// identity ids (dense lookup), else 0 (the key of an empty hash slot).
fn ids(c: &Case) -> Vec<i64> {
    let unknown = if c.ext[0] == 0 { c.ext.len() as i64 } else { 0 };
    c.hist
        .iter()
        .map(|h| h.map_or(unknown, |i| c.ext[i]))
        .collect()
}

fn case(float: bool) -> impl Strategy<Value = Case> {
    (1usize..12).prop_flat_map(move |n| {
        let cell = if float {
            prop_oneof![3 => Just(0.0f32), 2 => 0.1f32..3.0, 1 => -3.0f32..-0.1].boxed()
        } else {
            prop_oneof![
                3 => Just(0.0f32),
                2 => (1u8..4).prop_map(f32::from),
                1 => (-3i8..0).prop_map(f32::from),
            ]
            .boxed()
        };
        // Some trailing all-zero columns: ids >= n_active (G2).
        let empty_tail = 0..=n / 2;
        (
            prop::collection::vec(prop::collection::vec(cell, n), n),
            empty_tail,
            Just((0..n as i32).collect::<Vec<_>>()).prop_shuffle(),
            prop::bool::ANY,
            prop::collection::vec(prop::option::weighted(0.9, 0..n), 0..2 * n),
            prop::option::of(prop::collection::vec((-2i8..4).prop_map(f32::from), 2 * n)),
            0..n + 2,
            prop::bool::ANY,
            prop::option::of(prop::collection::vec(prop::bool::weighted(0.6), n)),
        )
            .prop_map(
                move |(mut s, tail, new_to_old, identity, hist, w, k, exclude_seen, allowed)| {
                    for row in &mut s {
                        for v in &mut row[n - tail..] {
                            *v = 0.0;
                        }
                    }
                    let ext = if identity {
                        (0..n as i64).collect()
                    } else {
                        (0..n as i64)
                            .map(|i| i * 7_919_000_017 - 3_000_000_000_000)
                            .collect()
                    };
                    let weights = w.map(|w| w[..hist.len()].to_vec());
                    Case {
                        s,
                        new_to_old,
                        ext,
                        hist,
                        weights,
                        k,
                        exclude_seen,
                        allowed,
                    }
                },
            )
    })
}

fn variant() -> impl Strategy<Value = Variant> {
    (
        prop::sample::select(vec![
            Lookup::Binary,
            Lookup::Eytzinger,
            Lookup::Hash,
            Lookup::HashBatched,
        ]),
        prop::sample::select(vec![Seen::NegInf, Seen::Bitset, Seen::Later]),
        prop::sample::select(vec![Mask::AfterFloor, Mask::BeforeFloor, Mask::Later]),
        0usize..6,
        prop::bool::ANY,
        prop::bool::ANY,
    )
        .prop_map(
            |(lookup, seen, mask, block, canonicalize, reuse_scratch)| Variant {
                lookup,
                seen,
                mask,
                block,
                canonicalize,
                reuse_scratch,
            },
        )
}

fn bits(c: &Case) -> Option<Vec<u64>> {
    c.allowed.as_ref().map(|a| {
        let mut b = vec![0u64; a.len().div_ceil(64)];
        for (i, _) in a.iter().enumerate().filter(|x| *x.1) {
            b[i / 64] |= 1 << (i % 64);
        }
        b
    })
}

fn run(c: &Case, m: &Model, v: &Variant) -> (Vec<i64>, Vec<f32>) {
    let (ids, allowed) = (ids(c), bits(c));
    let req = Request {
        ids: &ids,
        weights: c.weights.as_deref(),
        k: c.k,
        exclude_seen: c.exclude_seen,
        allowed: allowed.as_deref(),
    };
    let mut out = (vec![], vec![]);
    recommend(m, &req, v, &mut out.0, &mut out.1).unwrap();
    out
}

/// The engine's semantics in f64: in-drain filters, top `fetch`, later filters, top k.
fn reference(c: &Case, v: &Variant) -> (Vec<i64>, Vec<f32>) {
    let n = c.s.len();
    let old_to_new = |old: usize| {
        c.new_to_old
            .iter()
            .position(|&o| o as usize == old)
            .unwrap()
    };
    let hist: Vec<(usize, f64)> = c
        .hist
        .iter()
        .enumerate()
        .filter_map(|(p, h)| {
            h.map(|old| {
                (
                    old_to_new(old),
                    c.weights.as_ref().map_or(1.0, |w| f64::from(w[p])),
                )
            })
        })
        .collect();
    let mut seen = vec![false; n];
    for &(i, _) in &hist {
        seen[i] = true;
    }
    let n_hist = if v.canonicalize {
        seen.iter().filter(|&&s| s).count()
    } else {
        hist.len()
    };
    let n_active = (0..n)
        .rev()
        .find(|&j| c.s.iter().any(|r| r[j] != 0.0))
        .map_or(0, |j| j + 1);
    let allowed = |j: usize| c.allowed.as_ref().is_none_or(|a| a[j]);
    let (seen_later, mask_later) = (
        c.exclude_seen && v.seen == Seen::Later,
        c.allowed.is_some() && v.mask == Mask::Later,
    );
    let mut cand: Vec<(usize, f64)> = (0..n)
        .map(|j| {
            (
                j,
                hist.iter()
                    .map(|&(i, w)| w * f64::from(c.s[i][j]))
                    .sum::<f64>(),
            )
        })
        .filter(|&(j, s)| {
            s > 0.0 && !(c.exclude_seen && !seen_later && seen[j]) && (mask_later || allowed(j))
        })
        .collect();
    cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = c.k.min(n_active);
    let fetch =
        (k + if seen_later { n_hist } else { 0 } + if mask_later { k } else { 0 }).min(n_active);
    let out: Vec<(usize, f64)> = cand
        .into_iter()
        .take(fetch)
        .filter(|&(j, _)| !(seen_later && seen[j]) && !(mask_later && !allowed(j)))
        .take(k)
        .collect();
    let ext = |j: usize| c.ext[c.new_to_old[j] as usize];
    (
        out.iter().map(|&(j, _)| ext(j)).collect(),
        out.iter().map(|&(_, s)| s as f32).collect(),
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 16 } else { 3000 }))]

    #[test]
    fn matches_reference(c in case(false), v in variant()) {
        let m = model(&c);
        prop_assert_eq!(run(&c, &m, &v), reference(&c, &v));
        // Dense lookup on the identity ids.
        if c.ext.iter().enumerate().all(|(i, &e)| e == i as i64) {
            let dense = Variant { lookup: Lookup::Dense, ..v };
            prop_assert_eq!(run(&c, &m, &dense), reference(&c, &v));
        }
    }

    #[test]
    fn bits_independent_of_blocking_order_and_batch(c in case(true), block in 1usize..6, rev in prop::bool::ANY) {
        let m = model(&c);
        let v = Variant { lookup: Lookup::Binary, ..Variant::default() };
        let want = run(&c, &m, &v);
        prop_assert_eq!(&run(&c, &m, &Variant { block, ..v }), &want);
        // History order and duplicate split don't change bits (weights 1.0 only:
        // merged weights are exact sums then).
        if c.weights.is_none() {
            let mut d = c.clone();
            if rev {
                d.hist.reverse();
            }
            d.hist.extend(c.hist.iter().take(1).copied());
            let mut e = c.clone();
            e.hist.extend(c.hist.iter().take(1).copied());
            prop_assert_eq!(run(&d, &m, &v), run(&e, &m, &v));
        }
        // Batch of two copies == single (no mask in batch).
        if c.allowed.is_none() {
            let ids = ids(&c);
            let hist: Vec<i64> = ids.iter().chain(&ids).copied().collect();
            let w = c.weights.as_ref().map(|w| w.iter().chain(w).copied().collect::<Vec<_>>());
            let indptr = [0, ids.len() as i64, 2 * ids.len() as i64];
            let (bp, bi, bs) = recommend_batch(&m, &indptr, &hist, w.as_deref(), c.k, c.exclude_seen, &v).unwrap();
            let l = want.0.len();
            prop_assert_eq!(bp, vec![0, l as i64, 2 * l as i64]);
            prop_assert_eq!(&bi[..l], &want.0[..]);
            prop_assert_eq!(&bi[l..], &want.0[..]);
            prop_assert_eq!(bs[..l].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            prop_assert_eq!(&bs[l..], &bs[..l]);
        }
    }
}

#[test]
fn rejects_bad_input() {
    let s = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let (indptr, indices, data) = csr(&s);
    let new = |p: &[i32], e: &[i64]| {
        Model::new(indptr.clone(), indices.clone(), data.clone(), p, e).err()
    };
    assert!(new(&[0, 0], &[0, 1]).is_some()); // not a permutation
    assert!(new(&[0, 1], &[5, 5]).is_some()); // duplicate ids
    assert!(new(&[0], &[0]).is_some()); // not square
    let m = Model::new(indptr, indices, data, &[1, 0], &[0, 1]).unwrap();
    let mut out = (vec![], vec![]);
    let req = |w, a| Request {
        ids: &[0],
        weights: w,
        k: 1,
        exclude_seen: true,
        allowed: a,
    };
    let v = Variant::default();
    assert!(
        recommend(
            &m,
            &req(Some(&[f32::NAN]), None),
            &v,
            &mut out.0,
            &mut out.1
        )
        .is_err()
    );
    assert!(
        recommend(
            &m,
            &req(Some(&[1.0, 2.0]), None),
            &v,
            &mut out.0,
            &mut out.1
        )
        .is_err()
    );
    assert!(recommend(&m, &req(None, Some(&[1, 1])), &v, &mut out.0, &mut out.1).is_err());
    assert!(recommend(&m, &req(None, None), &v, &mut out.0, &mut out.1).is_ok());
    // id 0 = original column 0 = internal 1; its only neighbour is internal 0 = id 1.
    assert_eq!(out, (vec![1], vec![1.0]));
}
