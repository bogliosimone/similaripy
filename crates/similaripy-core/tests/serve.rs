//! Property tests: serving against a naive dense f64 reference. Small
//! integers keep every sum exact, so results must match exactly; float values
//! check the bit-identity claims (batch == single for any block size, baked
//! mask == per-call mask, history order doesn't matter).

use proptest::prelude::*;
use similaripy_core::CsrView;
use similaripy_core::serve::{
    BlockSize, Ids, ItemMask, Model, Request, Unknown, recommend, recommend_batch,
};

#[derive(Debug, Clone)]
struct Case {
    /// S over the original ids.
    s: Vec<Vec<f32>>,
    /// A random order of the kept items (internal id → original id).
    new_to_old: Vec<i32>,
    /// External id per original id; None = column indices.
    ext: Option<Vec<i64>>,
    /// Original ids, or `None` for an id no item has.
    hist: Vec<Option<usize>>,
    weights: Option<Vec<f32>>,
    k: usize,
    exclude_seen: bool,
    /// Allowed flag per original id.
    allowed: Option<Vec<bool>>,
}

struct Csr {
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
}

fn csr(s: &[Vec<f32>]) -> Csr {
    let mut m = Csr {
        indptr: vec![0],
        indices: vec![],
        data: vec![],
    };
    for row in s {
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

fn build_blocked(
    s: &[Vec<f32>],
    new_to_old: &[i32],
    ext: Option<&[i64]>,
    block: usize,
) -> Result<Model, similaripy_core::Error> {
    let m = csr(s);
    let view = CsrView::new("S", &m.indptr, &m.indices, &m.data, s.len())?;
    let block = if block == 0 {
        BlockSize::Off
    } else {
        BlockSize::Size(block)
    };
    Model::with_order(view, new_to_old.to_vec(), ext.map(<[i64]>::to_vec), block)
}

/// S as a caller may give it: rows reversed, each entry split into two halves
/// (exact), an explicit zero and a pair that cancels at column 0.
fn messy(s: &[Vec<f32>]) -> Csr {
    let mut m = Csr {
        indptr: vec![0],
        indices: vec![],
        data: vec![],
    };
    for row in s {
        for (c, &v) in row.iter().enumerate().rev() {
            if v != 0.0 {
                m.indices.extend([c as i32, c as i32]);
                m.data.extend([v / 2.0, v / 2.0]);
            }
        }
        m.indices.push(0);
        m.data.push(0.0);
        if row[0] == 0.0 {
            m.indices.extend([0, 0]);
            m.data.extend([1.5, -1.5]);
        }
        m.indptr.push(m.indices.len() as i32);
    }
    m
}

/// Popularity order: descending column count, ties to the lower column;
/// items with no row and no column left out.
fn popularity(s: &[Vec<f32>]) -> Vec<i32> {
    let n = s.len();
    let count = |j: usize| s.iter().filter(|r| r[j] != 0.0).count();
    let mut order: Vec<i32> = (0..n as i32).collect();
    order.sort_by_key(|&j| std::cmp::Reverse(count(j as usize)));
    order.retain(|&j| count(j as usize) > 0 || s[j as usize].iter().any(|&v| v != 0.0));
    order
}

/// A model rebuilt from its own parts (the pickle / load path).
fn from_own_parts(m: &Model) -> Model {
    let p = m.parts();
    Model::from_parts(
        p.indptr.to_vec(),
        p.indices.to_vec(),
        p.data.to_vec(),
        p.new_to_old.to_vec(),
        p.n_cols,
        p.ext.map(<[i64]>::to_vec),
        p.allowed.map(<[u64]>::to_vec),
        p.block_size,
    )
    .unwrap()
}

fn build(
    s: &[Vec<f32>],
    new_to_old: &[i32],
    ext: Option<&[i64]>,
) -> Result<Model, similaripy_core::Error> {
    build_blocked(s, new_to_old, ext, 0)
}

fn model(c: &Case) -> Model {
    build(&c.s, &c.new_to_old, c.ext.as_deref()).unwrap()
}

fn ext_of(c: &Case, i: usize) -> i64 {
    c.ext.as_ref().map_or(i as i64, |e| e[i])
}

/// An id no item has: past the end for column-index ids, else 0 (the key of
/// an empty hash slot; never a real id below).
fn unknown_id(c: &Case) -> i64 {
    if c.ext.is_none() { c.s.len() as i64 } else { 0 }
}

fn ids(c: &Case) -> Vec<i64> {
    c.hist
        .iter()
        .map(|h| h.map_or(unknown_id(c), |i| ext_of(c, i)))
        .collect()
}

fn mask(c: &Case, m: &Model) -> Option<ItemMask> {
    c.allowed.as_ref().map(|a| {
        let ids: Vec<i64> = (0..a.len())
            .filter(|&j| a[j])
            .map(|j| ext_of(c, j))
            .collect();
        m.mask(Ids::External(&ids), Unknown::Skip).unwrap()
    })
}

fn kept(c: &Case, i: usize) -> bool {
    c.new_to_old.contains(&(i as i32))
}

fn case(float: bool) -> impl Strategy<Value = Case> {
    // Mostly small; sometimes past 64 items: bitsets span several words.
    let sizes = if cfg!(miri) {
        (1usize..10).boxed()
    } else {
        prop_oneof![4 => 1usize..12, 1 => 60usize..90].boxed()
    };
    sizes.prop_flat_map(move |n| {
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
        (
            prop::collection::vec(prop::collection::vec(cell, n), n),
            // per item: 0 normal, 1 no column, 2 no row and no column (dropped)
            prop::collection::vec(
                prop_oneof![6 => Just(0u8), 1 => Just(1u8), 1 => Just(2u8)],
                n,
            ),
            Just((0..n as i32).collect::<Vec<_>>()).prop_shuffle(),
            prop::bool::ANY,
            prop::collection::vec(prop::option::weighted(0.9, 0..n), 0..2 * n),
            prop::option::of(prop::collection::vec((-2i8..4).prop_map(f32::from), 2 * n)),
            0..n + 2,
            prop::bool::ANY,
            prop::option::of(prop::collection::vec(prop::bool::weighted(0.6), n)),
        )
            .prop_map(
                move |(mut s, kind, order, identity, hist, w, k, exclude_seen, allowed)| {
                    for (i, &t) in kind.iter().enumerate() {
                        for row in &mut s {
                            if t >= 1 {
                                row[i] = 0.0;
                            }
                        }
                        if t == 2 {
                            s[i].iter_mut().for_each(|v| *v = 0.0);
                        }
                    }
                    // Keep every item with entries, and some empty ones.
                    let new_to_old = order
                        .into_iter()
                        .filter(|&i| {
                            let i = i as usize;
                            kind[i] != 2
                                || s[i].iter().any(|&v| v != 0.0)
                                || s.iter().any(|r| r[i] != 0.0)
                        })
                        .collect();
                    let ext = (!identity).then(|| {
                        // Edge keys as real items, never 0 (the unknown id).
                        let edge = [i64::MIN, i64::MAX, -1];
                        (0..n as i64)
                            .map(|i| {
                                edge.get(i as usize)
                                    .copied()
                                    .unwrap_or(i * 7_919_000_017 - 3_000_000_000_000)
                            })
                            .collect()
                    });
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

fn req<'a>(
    c: &'a Case,
    ids: &'a [i64],
    allowed: Option<&'a ItemMask>,
    unknown: Unknown,
) -> Request<'a> {
    Request {
        ids: Ids::External(ids),
        weights: c.weights.as_deref(),
        k: c.k,
        exclude_seen: c.exclude_seen,
        allowed,
        unknown,
    }
}

fn run(c: &Case, m: &Model) -> (Vec<i64>, Vec<f32>) {
    let (ids, allowed) = (ids(c), mask(c, m));
    let mut out = (vec![], vec![]);
    recommend(
        m,
        &req(c, &ids, allowed.as_ref(), Unknown::Skip),
        &mut out.0,
        &mut out.1,
    )
    .unwrap();
    out
}

/// Top k of score = h · S (> 0) over the original ids; ties: lower internal id.
fn reference(c: &Case) -> (Vec<i64>, Vec<f32>) {
    let n = c.s.len();
    let internal = |j: usize| c.new_to_old.iter().position(|&o| o as usize == j);
    let hist: Vec<(usize, f64)> = c
        .hist
        .iter()
        .enumerate()
        .filter_map(|(p, h)| {
            h.filter(|&i| kept(c, i))
                .map(|i| (i, c.weights.as_ref().map_or(1.0, |w| f64::from(w[p]))))
        })
        .collect();
    let seen: Vec<bool> = (0..n).map(|j| hist.iter().any(|&(i, _)| i == j)).collect();
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
            s > 0.0 && !(c.exclude_seen && seen[j]) && c.allowed.as_ref().is_none_or(|a| a[j])
        })
        .collect();
    cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(internal(a.0).cmp(&internal(b.0))));
    cand.truncate(c.k);
    (
        cand.iter().map(|&(j, _)| ext_of(c, j)).collect(),
        cand.iter().map(|&(_, s)| s as f32).collect(),
    )
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 12 } else { 3000 }))]

    #[test]
    fn matches_reference(c in case(false)) {
        let m = model(&c);
        prop_assert_eq!(run(&c, &m), reference(&c));
        // unknown='raise' fails exactly when an id is unknown (never seen or dropped).
        let ids = ids(&c);
        let any_unknown = c.hist.iter().any(|h| h.is_none_or(|i| !kept(&c, i)));
        let mut out = (vec![], vec![]);
        let r = recommend(&m, &req(&c, &ids, None, Unknown::Raise), &mut out.0, &mut out.1);
        prop_assert_eq!(r.is_err(), any_unknown);
    }

    #[test]
    fn bits_batch_blocks_baked_mask_and_order(c in case(true), block in 0usize..6) {
        let m = model(&c);
        let want = run(&c, &m);
        // Trimmed S and a view's default mask == the per-call mask, bit for
        // bit (same layout); view + per-call mask == the per-call AND.
        if let Some(a) = mask(&c, &m) {
            let ids = ids(&c);
            let go = |model: &Model, allowed: Option<&ItemMask>| {
                let mut out = (vec![], vec![]);
                recommend(model, &req(&c, &ids, allowed, Unknown::Skip), &mut out.0, &mut out.1).unwrap();
                out
            };
            let trimmed = m.trimmed(&a).unwrap();
            let view = m.view(Some(&a)).unwrap();
            for got in [go(&trimmed, None), go(&view, None), go(&trimmed, Some(&a)), go(&view, Some(&a))] {
                prop_assert_eq!(&got.0, &want.0);
                prop_assert_eq!(bits(&got.1), bits(&want.1));
            }
            // A view of a view replaces the mask: None widens back to all.
            let wide = go(&view.view(None).unwrap(), None);
            prop_assert_eq!(&wide.0, &go(&m, None).0);
            // Default mask a AND per-call mask "even original ids".
            let even: Vec<i64> = (0..c.s.len()).filter(|j| j % 2 == 0).map(|j| ext_of(&c, j)).collect();
            let both: Vec<i64> = (0..c.s.len())
                .filter(|&j| j % 2 == 0 && c.allowed.as_ref().is_some_and(|al| al[j]))
                .map(|j| ext_of(&c, j))
                .collect();
            let b = m.mask(Ids::External(&even), Unknown::Skip).unwrap();
            let ab = m.mask(Ids::External(&both), Unknown::Skip).unwrap();
            let got = go(&view, Some(&b));
            let and = go(&m, Some(&ab));
            prop_assert_eq!(&got.0, &and.0);
            prop_assert_eq!(bits(&got.1), bits(&and.1));
        }
        // History order doesn't change bits (weights 1.0: duplicates sum exactly).
        if c.weights.is_none() {
            let mut d = c.clone();
            d.hist.reverse();
            prop_assert_eq!(&run(&d, &m).0, &want.0);
            prop_assert_eq!(bits(&run(&d, &m).1), bits(&want.1));
        }
        // Batch of two copies (columns of S, known ids only) == single, any block size.
        let hist: Vec<i32> = c.hist.iter().flatten().map(|&i| i as i32).collect();
        let w: Option<Vec<f32>> = c.weights.as_ref().map(|w| {
            c.hist.iter().zip(w).filter(|(h, _)| h.is_some()).map(|(_, &x)| x).collect()
        });
        let single = {
            let c2 = Case { hist: c.hist.iter().copied().filter(Option::is_some).collect(), weights: w.clone(), ..c.clone() };
            run(&c2, &m)
        };
        let both: Vec<i32> = hist.iter().chain(&hist).copied().collect();
        let w2 = w.as_ref().map(|w| w.iter().chain(w).copied().collect::<Vec<_>>());
        let n = hist.len() as i64;
        // On a model with column blocks: same bits as the one-block single path.
        let mb = build_blocked(&c.s, &c.new_to_old, c.ext.as_deref(), block).unwrap();
        let allowed = mask(&c, &mb);
        let (bp, bc, bs) = recommend_batch(&mb, &[0, n, 2 * n], &both, w2.as_deref(), c.k, c.exclude_seen, allowed.as_ref()).unwrap();
        let l = single.0.len();
        prop_assert_eq!(bp, vec![0, l as i64, 2 * l as i64]);
        let as_ext: Vec<i64> = bc.iter().map(|&j| ext_of(&c, j as usize)).collect();
        prop_assert_eq!(&as_ext[..l], &single.0[..]);
        prop_assert_eq!(&as_ext[l..], &single.0[..]);
        prop_assert_eq!(bits(&bs[..l]), bits(&single.1));
        prop_assert_eq!(bits(&bs[l..]), bits(&single.1));
    }

    // `new` on S as a caller may give it == the canonical S in popularity
    // order, array for array; a model rebuilt from its parts answers the same.
    #[test]
    fn raw_input_popularity_and_parts(c in case(true)) {
        let raw = messy(&c.s);
        let view = CsrView::new("S", &raw.indptr, &raw.indices, &raw.data, c.s.len()).unwrap();
        let m = Model::new(view, c.ext.clone(), BlockSize::Auto).unwrap();
        let want = build(&c.s, &popularity(&c.s), c.ext.as_deref()).unwrap();
        let (p, q) = (m.parts(), want.parts());
        prop_assert_eq!((p.indptr, p.indices, p.new_to_old, p.n_cols), (q.indptr, q.indices, q.new_to_old, q.n_cols));
        prop_assert_eq!(bits(p.data), bits(q.data));
        prop_assert_eq!(p.ext, q.ext);
        let c2 = Case { new_to_old: popularity(&c.s), ..c.clone() };
        let got = run(&c2, &m);
        let view = match mask(&c2, &m) { Some(a) => m.view(Some(&a)).unwrap(), None => m.view(None).unwrap() };
        let back = from_own_parts(&view);
        prop_assert_eq!(back.parts().allowed, view.parts().allowed);
        prop_assert_eq!(back.parts().block_size, BlockSize::Auto);
        let again = run(&Case { allowed: None, ..c2.clone() }, &back);
        prop_assert_eq!(&again.0, &got.0);
        prop_assert_eq!(bits(&again.1), bits(&got.1));
    }
}

#[test]
fn rejects_bad_input() {
    let s = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0], vec![0.0; 3]];
    assert!(build(&s, &[0, 0], None).is_err()); // not distinct
    assert!(build(&s, &[0], None).is_err()); // leaves out item 1 (has entries)
    assert!(build(&[vec![0.0, 1.0], vec![0.0; 2]], &[1], None).is_err()); // leaves out item 0 (a row, no column)
    assert!(build(&s, &[0, 1], Some(&[5, 5, 6])).is_err()); // duplicate external ids
    assert!(build(&s, &[0, 1], Some(&[5, 6])).is_err()); // one id per item
    let m = build(&s, &[1, 0], None).unwrap(); // item 2 dropped
    assert_eq!((m.n_items(), m.n_cols(), m.n_active()), (2, 3, 2));
    let run = |r: &Request| {
        let mut out = (vec![], vec![]);
        recommend(&m, r, &mut out.0, &mut out.1).map(|()| out)
    };
    let r = |ids, w, a, u| Request {
        ids: Ids::External(ids),
        weights: w,
        k: 5,
        exclude_seen: true,
        allowed: a,
        unknown: u,
    };
    assert!(run(&r(&[0], Some(&[f32::NAN]), None, Unknown::Skip)).is_err());
    assert!(run(&r(&[0], Some(&[1.0, 2.0]), None, Unknown::Skip)).is_err());
    assert!(run(&r(&[2], None, None, Unknown::Raise)).is_err()); // dropped = unknown
    assert_eq!(
        run(&r(&[0, 2, -4, 99], None, None, Unknown::Skip)).unwrap(),
        (vec![1], vec![1.0])
    );
    // A mask from another build of the same S: other layout.
    let other = build(&s, &[1, 0], None)
        .unwrap()
        .mask(Ids::External(&[1]), Unknown::Skip)
        .unwrap();
    assert!(run(&r(&[0], None, Some(&other), Unknown::Skip)).is_err());
    assert!(m.trimmed(&other).is_err());
    assert!(m.view(Some(&other)).is_err());
    let own = m.mask(Ids::External(&[1]), Unknown::Skip).unwrap();
    assert_eq!(
        run(&r(&[0], None, Some(&own), Unknown::Skip)).unwrap(),
        (vec![1], vec![1.0])
    );
    // A huge k is clamped before anything is allocated.
    let big = Request {
        k: usize::MAX / 2,
        ..r(&[0], None, None, Unknown::Skip)
    };
    assert_eq!(run(&big).unwrap(), (vec![1], vec![1.0]));
    let (bp, bc, _) = recommend_batch(&m, &[0, 1], &[0], None, usize::MAX / 2, true, None).unwrap();
    assert_eq!((bp, bc), (vec![0, 1], vec![1]));
    // Batch: out-of-range columns and a bad CSR are errors.
    assert!(recommend_batch(&m, &[0, 1], &[3], None, 1, true, None).is_err());
    assert!(recommend_batch(&m, &[0, 2], &[0], None, 1, true, None).is_err());
}

/// Owned parts of a model, to corrupt one field at a time.
#[derive(Clone)]
struct Owned {
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    new_to_old: Vec<i32>,
    n_cols: usize,
    ext: Option<Vec<i64>>,
    allowed: Option<Vec<u64>>,
    block_size: BlockSize,
}

impl Owned {
    fn of(m: &Model) -> Self {
        let p = m.parts();
        Self {
            indptr: p.indptr.to_vec(),
            indices: p.indices.to_vec(),
            data: p.data.to_vec(),
            new_to_old: p.new_to_old.to_vec(),
            n_cols: p.n_cols,
            ext: p.ext.map(<[i64]>::to_vec),
            allowed: p.allowed.map(<[u64]>::to_vec),
            block_size: p.block_size,
        }
    }

    fn load(self) -> Result<Model, similaripy_core::Error> {
        Model::from_parts(
            self.indptr,
            self.indices,
            self.data,
            self.new_to_old,
            self.n_cols,
            self.ext,
            self.allowed,
            self.block_size,
        )
    }
}

#[test]
fn from_parts_checks_every_invariant() {
    // 3 kept items of 4 (item 3 dropped), one row with two entries.
    let s = vec![
        vec![0.0, 1.0, 2.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 3.0, 0.0, 0.0],
        vec![0.0; 4],
    ];
    let m = build(&s, &[1, 0, 2], Some(&[10, 11, 12, 13])).unwrap();
    let a = m.mask(Ids::External(&[11]), Unknown::Skip).unwrap();
    let good = Owned::of(&m.view(Some(&a)).unwrap());
    assert!(good.clone().load().is_ok());
    assert_eq!(good.indptr, vec![0, 1, 3, 4]); // row of item 0 = internal 1
    let bad = |f: &dyn Fn(&mut Owned)| {
        let mut o = good.clone();
        f(&mut o);
        o.load().is_err()
    };
    assert!(bad(&|o| o.indptr[0] = 1));
    assert!(bad(&|o| o.indptr[1] = 4)); // not monotone
    assert!(bad(&|o| *o.indptr.last_mut().unwrap() = 3)); // != nnz
    assert!(bad(&|o| {
        o.indptr.pop();
    })); // one row per item
    assert!(bad(&|o| o.indices[0] = 3)); // column >= n_items
    assert!(bad(&|o| o.indices[0] = -1));
    assert!(bad(&|o| o.indices.swap(1, 2))); // unsorted row
    assert!(bad(&|o| o.indices[2] = o.indices[1])); // duplicate column
    assert!(bad(&|o| {
        o.data.pop();
    }));
    assert!(bad(&|o| o.data[0] = f32::INFINITY));
    assert!(bad(&|o| o.data[0] = f32::NAN));
    assert!(bad(&|o| o.new_to_old[0] = o.new_to_old[1])); // not distinct
    assert!(bad(&|o| o.new_to_old[0] = 4)); // >= n_cols
    assert!(bad(&|o| o.new_to_old[0] = -1));
    assert!(bad(&|o| o.n_cols = 2)); // new_to_old past n_cols
    assert!(bad(&|o| o.n_cols = usize::MAX)); // no huge allocation
    assert!(bad(&|o| o.ext.as_mut().unwrap().truncate(3)));
    assert!(bad(&|o| o.ext.as_mut().unwrap()[0] = 13)); // duplicate id
    assert!(bad(&|o| o.allowed.as_mut().unwrap().push(0))); // one bit per item
    assert!(bad(&|o| o.block_size = BlockSize::Size(0)));
    // Not a corruption: an extra empty item at the end (no row, no column).
    let mut o = good.clone();
    o.n_cols = 5;
    o.ext.as_mut().unwrap().push(14);
    assert!(o.load().is_ok());
}
