//! Property tests: serving against a naive dense f64 reference. Small
//! integers keep every sum exact, so results must match exactly; float values
//! check the bit-identity claims (batch == single for any block size, a
//! view == a per-call include, history order doesn't matter); a trimmed S ==
//! a model of S · diag(keep) in its own order, array for array.

use proptest::prelude::*;
use similaripy_core::CsrView;
use similaripy_core::serve::{
    BatchRequest, BlockSize, Entry, ItemOrder, Lists, MAX_VIEWS, Model, Request, ScoreRequest,
    Stats, Worker, recommend, recommend_batch, score, score_batch,
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
    exclude_history: bool,
    /// Include flag per original id (None = no include rule).
    allowed: Option<Vec<bool>>,
    /// Exclude flag per original id.
    excluded: Option<Vec<bool>>,
    sorted: bool,
}

struct Csr {
    indptr: Vec<i64>,
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
        m.indptr.push(m.indices.len() as i64);
    }
    m
}

/// A test's block (0 = one block) as a setting.
fn block_size(block: usize) -> BlockSize {
    if block == 0 {
        BlockSize::Off
    } else {
        BlockSize::Size(block)
    }
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
        m.indptr.push(m.indices.len() as i64);
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

/// The build's order from the dense S: `popularity`; S's own order (items
/// with no column last); or
/// clustered: the `head` most popular, then the rest by (no column first =
/// false, popularity rank of the most popular neighbour (own rank without a
/// row), own rank).
fn reference_order(s: &[Vec<f32>], order: ItemOrder, head: usize) -> Vec<i32> {
    let mut pop = popularity(s);
    match order {
        ItemOrder::Popularity => pop,
        ItemOrder::Original => {
            let no_col = |j: i32| s.iter().all(|r| r[j as usize] == 0.0);
            pop.sort_by_key(|&j| (no_col(j), j));
            pop
        }
        ItemOrder::Auto | ItemOrder::Clustered => {
            let by_rank = pop.clone();
            let rank = |j: usize| by_rank.iter().position(|&p| p as usize == j).unwrap();
            let count = |j: usize| s.iter().filter(|r| r[j] != 0.0).count();
            let key = |j: usize| {
                let anchor = (0..s.len()).filter(|&i| s[j][i] != 0.0).map(rank).min();
                (count(j) == 0, anchor.unwrap_or(rank(j)), rank(j))
            };
            let mut tail = pop.split_off(head.min(pop.len()));
            tail.sort_by_key(|&j| key(j as usize));
            pop.extend(tail);
            pop
        }
    }
}

fn build(
    s: &[Vec<f32>],
    new_to_old: &[i32],
    ext: Option<&[i64]>,
) -> Result<Model, similaripy_core::Error> {
    let m = csr(s);
    let view = CsrView::new("S", &m.indptr, &m.indices, &m.data, s.len())?;
    Model::with_order(view, new_to_old.to_vec(), ext.map(<[i64]>::to_vec))
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

/// The ids whose flag is set, plus an id no item has (ignored).
fn flagged(c: &Case, flags: &Option<Vec<bool>>) -> Option<Vec<i64>> {
    flags.as_ref().map(|a| {
        (0..a.len())
            .filter(|&j| a[j])
            .map(|j| ext_of(c, j))
            .chain([unknown_id(c)])
            .collect()
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
        // Positive float ranges start at 0.125, a multiple of ulp(3.0): from 0.1,
        // proptest 1.12's sampler can overshoot `low` and trip its own (inverted)
        // debug_assert, ~1 in 4e7 draws in debug builds.
        let cell = if float {
            prop_oneof![3 => Just(0.0f32), 2 => 0.125f32..3.0, 1 => -3.0f32..-0.125].boxed()
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
            prop::option::of(prop::collection::vec(prop::bool::weighted(0.2), n)),
            prop::bool::weighted(0.8),
        )
            .prop_map(
                move |(
                    mut s,
                    kind,
                    order,
                    identity,
                    hist,
                    w,
                    k,
                    exclude_history,
                    allowed,
                    excluded,
                    sorted,
                )| {
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
                        exclude_history,
                        allowed,
                        excluded,
                        sorted,
                    }
                },
            )
    })
}

fn req<'a>(
    c: &'a Case,
    ids: &'a [i64],
    include: Option<&'a [i64]>,
    exclude: Option<&'a [i64]>,
) -> Request<'a> {
    Request {
        ids,
        weights: c.weights.as_deref(),
        k: c.k,
        exclude_history: c.exclude_history,
        include,
        exclude,
        sorted: c.sorted,
        ..Default::default()
    }
}

fn sreq<'a>(c: &'a Case, ids: &'a [i64], items: &'a [i64]) -> ScoreRequest<'a> {
    ScoreRequest {
        ids,
        weights: c.weights.as_deref(),
        items,
        ..Default::default()
    }
}

fn go(
    c: &Case,
    m: &Model,
    include: Option<&[i64]>,
    exclude: Option<&[i64]>,
) -> (Vec<i64>, Vec<f32>) {
    let ids = ids(c);
    let mut out = (vec![], vec![]);
    recommend(
        m,
        0,
        &req(c, &ids, include, exclude),
        &mut out.0,
        &mut out.1,
    )
    .unwrap();
    out
}

fn run(c: &Case, m: &Model) -> (Vec<i64>, Vec<f32>) {
    let (inc, exc) = (flagged(c, &c.allowed), flagged(c, &c.excluded));
    go(c, m, inc.as_deref(), exc.as_deref())
}

/// Unsorted results in the sorted order (ties as `reference`: by position).
fn by_score(
    c: &Case,
    out: (Vec<i64>, Vec<f32>),
    want: &(Vec<i64>, Vec<f32>),
) -> (Vec<i64>, Vec<f32>) {
    if c.sorted {
        return out;
    }
    let mut pairs: Vec<(i64, f32)> = out.0.into_iter().zip(out.1).collect();
    let pos = |id: i64| want.0.iter().position(|&w| w == id).unwrap_or(usize::MAX);
    pairs.sort_by_key(|&(id, _)| pos(id));
    pairs.into_iter().unzip()
}

/// Top k of score = h · S (> 0) over the original ids; ties: lower internal id.
fn reference(c: &Case) -> (Vec<i64>, Vec<f32>) {
    let n = c.s.len();
    let internal = |j: usize| c.new_to_old.iter().position(|&o| o as usize == j);
    let seen = |j: usize| kept(c, j) && c.hist.contains(&Some(j));
    let mut cand: Vec<(usize, f64)> = (0..n)
        .map(|j| (j, hs(c, j)))
        .filter(|&(j, s)| {
            s > 0.0
                && !(c.exclude_history && seen(j))
                && c.allowed.as_ref().is_none_or(|a| a[j])
                && !c.excluded.as_ref().is_some_and(|x| x[j])
        })
        .collect();
    cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(internal(a.0).cmp(&internal(b.0))));
    cand.truncate(c.k);
    (
        cand.iter().map(|&(j, _)| ext_of(c, j)).collect(),
        cand.iter().map(|&(_, s)| s as f32).collect(),
    )
}

/// (h · S)[j] in f64: the history's kept items, weighted.
fn hs(c: &Case, j: usize) -> f64 {
    c.hist
        .iter()
        .enumerate()
        .filter_map(|(p, h)| {
            h.filter(|&i| kept(c, i))
                .map(|i| c.weights.as_ref().map_or(1.0, |w| f64::from(w[p])) * f64::from(c.s[i][j]))
        })
        .sum()
}

/// The history twice (a two-user batch): ids and weights.
fn twice(c: &Case, ids: &[i64]) -> (Vec<i64>, Option<Vec<f32>>) {
    let both = ids.iter().chain(ids).copied().collect();
    let w2 = c
        .weights
        .as_ref()
        .map(|w| w.iter().chain(w).copied().collect());
    (both, w2)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 12 } else { 3000 }))]

    #[test]
    fn matches_reference(c in case(false)) {
        let m = model(&c);
        let want = reference(&c);
        prop_assert_eq!(by_score(&c, run(&c, &m), &want), want);
    }

    #[test]
    fn bits_batch_blocks_views_and_order(c in case(true), block in 0usize..6) {
        let c = Case { sorted: true, ..c };
        let m = model(&c);
        let want = run(&c, &m);
        let (inc, exc) = (flagged(&c, &c.allowed), flagged(&c, &c.excluded));
        // A view == the per-call include / exclude, bit for bit; view +
        // per-call include == the per-call AND (a trimmed S: in
        // `raw_input_orders_and_parts`, its own order).
        if let Some(inc) = &inc {
            let view = m.view(Some(inc), exc.as_deref());
            for got in [go(&c, &view, None, None), go(&c, &view, Some(inc), exc.as_deref())] {
                prop_assert_eq!(&got.0, &want.0);
                prop_assert_eq!(bits(&got.1), bits(&want.1));
            }
            // A view of a view narrows: more rules = the parent's AND them
            // (== the same rules per call).
            let odd: Vec<i64> = (0..c.s.len()).filter(|j| j % 2 == 1).map(|j| ext_of(&c, j)).collect();
            let got = go(&c, &view.view(None, Some(&odd)), None, None);
            let and = go(&c, &view, None, Some(&odd));
            prop_assert_eq!(&got.0, &and.0);
            prop_assert_eq!(bits(&got.1), bits(&and.1));
            // View's include AND the per-call include "even original ids".
            let even: Vec<i64> = (0..c.s.len()).filter(|j| j % 2 == 0).map(|j| ext_of(&c, j)).collect();
            let both: Vec<i64> = (0..c.s.len())
                .filter(|&j| j % 2 == 0 && c.allowed.as_ref().is_some_and(|al| al[j]))
                .map(|j| ext_of(&c, j))
                .collect();
            let got = go(&c, &view, Some(&even), None);
            let and = go(&c, &m, Some(&both), exc.as_deref());
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
        // Batch of [history, empty, history] == single, any block size; the
        // middle user has no include rule (`no_include`) and no history.
        let ids = ids(&c);
        let n = ids.len() as i64;
        let (both, w2) = twice(&c, &ids);
        let per_user = |l: &Option<Vec<i64>>| {
            l.as_ref().map(|l| {
                let k = l.len() as i64;
                (vec![0, k, k, 2 * k], l.iter().chain(l).copied().collect::<Vec<_>>())
            })
        };
        let (pi, pe) = (per_user(&inc), per_user(&exc));
        let (o, got_ids, got_scores) = recommend_batch(&m, block_size(block), &BatchRequest {
            histories: Lists { offsets: &[0, n, n, 2 * n], ids: &both },
            weights: w2.as_deref(),
            k: c.k,
            exclude_history: c.exclude_history,
            include: pi.as_ref().map(|(o, i)| Lists { offsets: o, ids: i }),
            no_include: inc.as_ref().map(|_| &[false, true, false][..]),
            exclude: pe.as_ref().map(|(o, i)| Lists { offsets: o, ids: i }),
            ..Default::default()
        }).unwrap();
        let l = want.0.len();
        prop_assert_eq!(o, vec![0, l as i64, l as i64, 2 * l as i64]);
        prop_assert_eq!(&got_ids[..l], &want.0[..]);
        prop_assert_eq!(&got_ids[l..], &want.0[..]);
        prop_assert_eq!(bits(&got_scores[..l]), bits(&want.1));
        prop_assert_eq!(bits(&got_scores[l..]), bits(&want.1));
    }

    // `new` on S as a caller may give it == the canonical S in the reference
    // order (a small head: the catalogs are smaller than `CLUSTER_HEAD`),
    // array for array; trimmed (`keep` = the include flags) == the same on
    // S · diag(keep), its kept columns first (n_active = their count); a
    // model rebuilt from its parts answers the same.
    #[test]
    fn raw_input_orders_and_parts(c in case(true), o in 0usize..4, head in 0usize..12) {
        let order = [ItemOrder::Auto, ItemOrder::Clustered, ItemOrder::Popularity, ItemOrder::Original][o];
        let raw = messy(&c.s);
        let view = CsrView::new("S", &raw.indptr, &raw.indices, &raw.data, c.s.len()).unwrap();
        let keep = c.allowed.as_deref();
        let m = Model::new_with_head(view, c.ext.clone(), keep, order, head).unwrap();
        let s: Vec<Vec<f32>> = c.s.iter().map(|r| {
            r.iter().enumerate().map(|(j, &v)| if keep.is_none_or(|k| k[j]) { v } else { 0.0 }).collect()
        }).collect();
        let n_cols = (0..s.len()).filter(|&j| s.iter().any(|r| r[j] != 0.0)).count();
        prop_assert_eq!(m.n_active(), n_cols);
        let new_to_old = reference_order(&s, order, head);
        let want = build(&s, &new_to_old, c.ext.as_deref()).unwrap();
        let (p, q) = (m.parts(), want.parts());
        prop_assert_eq!((p.indptr, p.new_to_old, p.n_cols), (q.indptr, q.new_to_old, q.n_cols));
        let cols_bits = |m: &[Entry]| m.iter().map(|e| (e.col, e.val.to_bits())).collect::<Vec<_>>();
        prop_assert_eq!(cols_bits(p.pairs), cols_bits(q.pairs));
        prop_assert_eq!(p.ext, q.ext);
        let c2 = Case { s, new_to_old, ..c.clone() };
        let got = run(&c2, &m);
        let view = m.view(flagged(&c2, &c2.allowed).as_deref(), flagged(&c2, &c2.excluded).as_deref());
        let back = Owned::of(&view).load().unwrap(); // the pickle / load path
        prop_assert_eq!(back.parts().allowed, view.parts().allowed);
        let again = run(&Case { allowed: None, excluded: None, ..c2.clone() }, &back);
        prop_assert_eq!(&again.0, &got.0);
        prop_assert_eq!(bits(&again.1), bits(&got.1));
    }

    // score == h · S per candidate (every item reversed, a duplicate, an
    // unknown id), 0 where the view's mask (the case's include / exclude)
    // says no.
    #[test]
    fn score_matches_reference(c in case(false), block in 0usize..6) {
        let (inc, exc) = (flagged(&c, &c.allowed), flagged(&c, &c.excluded));
        let m = model(&c).view(inc.as_deref(), exc.as_deref());
        let n = c.s.len();
        let items: Vec<i64> = (0..n).rev().chain([0]).map(|j| ext_of(&c, j)).chain([unknown_id(&c)]).collect();
        let got = score(&m, block, &sreq(&c, &ids(&c), &items)).unwrap();
        let want: Vec<f32> = (0..n).rev().chain([0]).map(|j| {
            let ok = kept(&c, j) && c.allowed.as_ref().is_none_or(|a| a[j])
                && !c.excluded.as_ref().is_some_and(|x| x[j]);
            if ok { hs(&c, j) as f32 } else { 0.0 }
        }).chain([0.0]).collect();
        prop_assert_eq!(got, want);
    }

    // score's bits == recommend's scores (any block size), batch == single.
    #[test]
    fn score_bits_recommend_and_batch(c in case(true), block in 0usize..6) {
        let m = model(&c);
        let ids = ids(&c);
        let mut rec = (vec![], vec![]);
        let all = Request { k: c.s.len(), exclude_history: false, include: None, exclude: None, ..req(&c, &ids, None, None) };
        recommend(&m, block, &all, &mut rec.0, &mut rec.1).unwrap();
        let got = score(&m, block, &sreq(&c, &ids, &rec.0)).unwrap();
        prop_assert_eq!(bits(&got), bits(&rec.1));
        // Batch [history, empty history with items, history with no items].
        let items: Vec<i64> = (0..c.s.len()).map(|j| ext_of(&c, j)).collect();
        let single = score(&m, block, &sreq(&c, &ids, &items)).unwrap();
        let (h, k) = (ids.len() as i64, items.len() as i64);
        let (both, w2) = twice(&c, &ids);
        let it2: Vec<i64> = items.iter().chain(&items).copied().collect();
        let flat = score_batch(
            &m,
            block_size(block),
            Lists { offsets: &[0, h, h, 2 * h], ids: &both },
            w2.as_deref(),
            Lists { offsets: &[0, k, 2 * k, 2 * k], ids: &it2 },
            &[],
        ).unwrap();
        prop_assert_eq!(bits(&flat[..k as usize]), bits(&single));
        prop_assert!(flat[k as usize..].iter().all(|&v| v == 0.0));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 1000 }))]

    // One worker (any block) across two models of different sizes and a view,
    // in turns: each answer == the single-call path's ids and bits (sorted),
    // the same items and scores unsorted; scores too. Float cases, so the
    // oracle is the engine, not the f64 `reference`: f32 sums can tie where
    // f64 ones don't, and ties go to the lower internal id.
    #[test]
    fn one_worker_across_models(a in case(true), b in case(true), block in 0usize..6) {
        let (ma, mb) = (model(&a), model(&b));
        let inc = flagged(&b, &b.allowed);
        let vb = mb.view(inc.as_deref(), None);
        let mut w = Worker::new(&ma, block_size(block), 1).unwrap();
        prop_assert_eq!(w.block(), block);
        for (c, m, view) in [(&a, &ma, false), (&b, &mb, false), (&b, &vb, true), (&a, &ma, false)] {
            let c = Case { allowed: if view { c.allowed.clone() } else { None }, excluded: None, ..c.clone() };
            let ids = ids(&c);
            let mut got = (vec![], vec![]);
            w.recommend(m, &req(&c, &ids, None, None), &mut got.0, &mut got.1).unwrap();
            let sorted = run(&Case { sorted: true, ..c.clone() }, m);
            if c.sorted {
                prop_assert_eq!(&got.0, &sorted.0);
                prop_assert_eq!(bits(&got.1), bits(&sorted.1));
            } else {
                // Unsorted: the same items and scores as sorted, any block.
                let mut x: Vec<_> = got.0.iter().zip(bits(&got.1)).collect();
                let mut y: Vec<_> = sorted.0.iter().zip(bits(&sorted.1)).collect();
                x.sort();
                y.sort();
                prop_assert_eq!(x, y);
            }
            let mut sc = vec![1.0];
            w.score(m, &sreq(&c, &ids, &got.0), &mut sc).unwrap();
            prop_assert_eq!(bits(&sc), bits(&score(m, 0, &sreq(&c, &ids, &got.0)).unwrap()));
        }
    }

    // Several views per request (V16) == one combined view (a view of a view
    // ANDs): the served model (the base, or the case's include / exclude
    // view) with 0..4 extra views and maybe the case's include per call ==
    // the served model narrowed by every view, the same ids and bits (any
    // order: the same pushes), any block; batch and score too. 0..6 masks:
    // the two-mask drains and `drain_many`, in every order of selectivity.
    #[test]
    fn views_match_one_combined_view(
        (c, extra) in case(true).prop_flat_map(|c| {
            let n = c.s.len();
            let flags = move |p| prop::option::of(prop::collection::vec(prop::bool::weighted(p), n));
            (Just(c), prop::collection::vec((flags(0.7), flags(0.2)), 0..5))
        }),
        served_view in prop::bool::ANY,
        call_include in prop::bool::ANY,
        block in 0usize..6,
    ) {
        let base = model(&c);
        let (inc, exc) = (flagged(&c, &c.allowed), flagged(&c, &c.excluded));
        let served = if served_view { base.view(inc.as_deref(), exc.as_deref()) } else { base.view(None, None) };
        let rules: Vec<_> = extra.iter().map(|(i, e)| (flagged(&c, i), flagged(&c, e))).collect();
        let views: Vec<Model> = rules.iter().map(|(i, e)| base.view(i.as_deref(), e.as_deref())).collect();
        let views: Vec<&Model> = views.iter().collect();
        let combined = rules.iter().fold(served.view(None, None), |m, (i, e)| m.view(i.as_deref(), e.as_deref()));
        let ids = ids(&c);
        let include = if call_include { inc.as_deref() } else { None };
        let r = Request { include, ..req(&c, &ids, None, None) };
        let (mut got, mut want) = ((vec![], vec![]), (vec![], vec![]));
        recommend(&served, block, &Request { views: &views, ..r }, &mut got.0, &mut got.1).unwrap();
        recommend(&combined, block, &r, &mut want.0, &mut want.1).unwrap();
        prop_assert_eq!(&got.0, &want.0);
        prop_assert_eq!(bits(&got.1), bits(&want.1));
        // Batch of [history, history]: views shared by every user.
        let n = ids.len() as i64;
        let (both, w2) = twice(&c, &ids);
        let offsets = [0, n, 2 * n];
        let b = |views| BatchRequest {
            histories: Lists { offsets: &offsets, ids: &both },
            weights: w2.as_deref(),
            k: c.k,
            exclude_history: c.exclude_history,
            sorted: c.sorted,
            views,
            ..Default::default()
        };
        let (got_b, want_b) = (
            recommend_batch(&served, block_size(block), &b(&views)).unwrap(),
            recommend_batch(&combined, block_size(block), &b(&[])).unwrap(),
        );
        prop_assert_eq!(&got_b.1, &want_b.1);
        prop_assert_eq!(bits(&got_b.2), bits(&want_b.2));
        // Score: an item a view excludes scores 0.0.
        let items: Vec<i64> = (0..c.s.len()).map(|j| ext_of(&c, j)).collect();
        let s = sreq(&c, &ids, &items);
        let got_s = score(&served, block, &ScoreRequest { views: &views, ..s }).unwrap();
        let want_s = score(&combined, block, &s).unwrap();
        prop_assert_eq!(bits(&got_s), bits(&want_s));
        let k = items.len() as i64;
        let it2: Vec<i64> = items.iter().chain(&items).copied().collect();
        let flat = score_batch(
            &served,
            block_size(block),
            Lists { offsets: &offsets, ids: &both },
            w2.as_deref(),
            Lists { offsets: &[0, k, 2 * k], ids: &it2 },
            &views,
        ).unwrap();
        prop_assert_eq!(bits(&flat), bits(&[&want_s[..], &want_s[..]].concat()));
    }
}

// Views per request: at most MAX_VIEWS, each of the served model's base (a
// model built again from the same S is another model), counted as errors;
// `n_allowed` counts items, not the last word's spare bits.
#[test]
fn views_checked_and_counted() {
    let n = 70; // two mask words, 6 spare bits
    let s: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if (i + j) % 3 == 0 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let order: Vec<i32> = (0..n as i32).collect();
    let (m, other) = (
        build(&s, &order, None).unwrap(),
        build(&s, &order, None).unwrap(),
    );
    assert_eq!(m.n_allowed(), n);
    assert_eq!(m.view(None, Some(&[0, 1, 99])).n_allowed(), n - 2);
    let v = m.view(Some(&[3, 4, 5]), None);
    assert_eq!(
        (v.n_allowed(), v.view(None, Some(&[4])).n_allowed()),
        (3, 2)
    );
    let mut w = Worker::new(&m, BlockSize::Off, 1).unwrap();
    let mut out = (vec![], vec![]);
    let foreign = other.view(None, None);
    let many: Vec<&Model> = vec![&v; MAX_VIEWS + 1];
    for (views, ok) in [
        (&[&foreign][..], false),
        (&many[..MAX_VIEWS], true),
        (&many[..], false),
    ] {
        let r = Request {
            ids: &[0],
            views,
            ..Default::default()
        };
        assert_eq!(w.recommend(&m, &r, &mut out.0, &mut out.1).is_ok(), ok);
        let sr = ScoreRequest {
            ids: &[0],
            items: &[3],
            views,
            ..Default::default()
        };
        assert_eq!(w.score(&m, &sr, &mut vec![]).is_ok(), ok);
    }
    assert_eq!((w.stats().requests, w.stats().errors), (6, 4));
}

// More candidates than the drain head (8 x k) takes, in one block and across
// blocks (the head spans several drains): the branchy tail after it runs on
// every request, k 1..4; == the reference (integer sums, exact). Each history
// row has its own columns (j % 10 == i % 10) and few distinct scores, so the
// head cuts and then meets ties with lower columns from the next row. With
// 0, 1 (the include) and 2 masks (a view + the include), allowing 75% / 50%
// (the masked head: masked-out candidates in it and past it) or ~5% of
// the items (below `MASKED_HEAD_DIV`'s share: the gated loop only).
#[test]
fn past_the_drain_head() {
    let n = 300;
    let s: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if j % 10 == i % 10 {
                        ((i * 7 + j * 13) % 4 + 1) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    for k in 1..5 {
        let c = Case {
            s: s.clone(),
            new_to_old: (0..n).rev().collect(),
            ext: None,
            hist: vec![Some(3), Some(250), Some(17)],
            weights: None,
            k,
            exclude_history: true,
            allowed: None,
            excluded: None,
            sorted: true,
        };
        let (m, ids) = (model(&c), ids(&c));
        let flags = |f: fn(usize) -> bool| Some((0..n as usize).map(f).collect::<Vec<_>>());
        let (a, b, e, few) = (
            flagged(&c, &flags(|j| j % 3 != 0)),
            flagged(&c, &flags(|j| j % 4 != 1)),
            flagged(&c, &flags(|j| j % 11 == 5)),
            flagged(&c, &flags(|j| j % 40 == 3 || j % 40 == 7)),
        );
        let view = m.view(a.as_deref(), e.as_deref());
        for masks in 0..5 {
            let (served, include, c) = match masks {
                0 => (&m, None, c.clone()),
                1 | 3 => (
                    &m,
                    if masks == 1 {
                        b.as_deref()
                    } else {
                        few.as_deref()
                    },
                    Case {
                        allowed: flags(if masks == 1 {
                            |j| j % 4 != 1
                        } else {
                            |j| j % 40 == 3 || j % 40 == 7
                        }),
                        ..c.clone()
                    },
                ),
                _ => (
                    &view,
                    if masks == 2 {
                        b.as_deref()
                    } else {
                        few.as_deref()
                    },
                    Case {
                        allowed: flags(if masks == 2 {
                            |j| j % 3 != 0 && j % 4 != 1
                        } else {
                            |j| j % 3 != 0 && (j % 40 == 3 || j % 40 == 7)
                        }),
                        excluded: flags(|j| j % 11 == 5),
                        ..c.clone()
                    },
                ),
            };
            let want = reference(&c);
            for block in [0, 7, 64] {
                let mut got = (vec![], vec![]);
                recommend(
                    served,
                    block,
                    &req(&c, &ids, include, None),
                    &mut got.0,
                    &mut got.1,
                )
                .unwrap();
                assert_eq!(got, want, "k {k}, masks {masks}, block {block}");
            }
        }
    }
}

#[test]
fn worker_block_and_stats() {
    // Item 2 has no column; item 3 is dropped (unknown).
    let s = vec![
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 3.0, 0.0, 0.0],
        vec![0.0; 4],
    ];
    let m = build(&s, &[1, 0, 2], None).unwrap();
    assert_eq!(m.n_active(), 2);
    let new = |b, t| Worker::new(&m, b, t).map(|w| w.block());
    assert_eq!(new(BlockSize::Off, 8).unwrap(), 0);
    assert_eq!(new(BlockSize::Size(7), 8).unwrap(), 7); // clamped per request, kept as given
    assert!(new(BlockSize::Size(0), 1).is_err());
    assert!(new(BlockSize::Auto, 1).is_ok());
    let mut w = Worker::new(&m, BlockSize::Off, 1).unwrap();
    let r = |ids, weights| Request {
        ids,
        weights,
        k: 5,
        exclude_history: false,
        ..Default::default()
    };
    let mut out = (vec![], vec![]);
    // ids 0, 0, 2 known, 3 and 9 unknown.
    w.recommend(&m, &r(&[0, 3, 0, 2, 9], None), &mut out.0, &mut out.1)
        .unwrap();
    // The caller's outputs are appended to, the worker's own buffers reused.
    w.recommend(&m, &r(&[1], None), &mut out.0, &mut out.1)
        .unwrap();
    assert_eq!(out, (vec![1, 0, 0], vec![5.0, 1.0, 1.0]));
    assert!(
        w.recommend(&m, &r(&[1], Some(&[f32::NAN])), &mut out.0, &mut out.1)
            .is_err()
    );
    let mut sc = vec![];
    let sr = |ids, items| ScoreRequest {
        ids,
        items,
        ..Default::default()
    };
    w.score(&m, &sr(&[0, 9], &[1, 2, 3]), &mut sc).unwrap();
    assert_eq!(sc, vec![1.0, 0.0, 0.0]);
    w.score(&m, &sr(&[0], &[2]), &mut sc).unwrap(); // no candidate column: still looked up
    assert_eq!(sc, vec![0.0]);
    assert_eq!(
        w.stats(),
        Stats {
            requests: 5,
            errors: 1,
            ids: 5 + 1 + 1 + 2 + 1,
            known_ids: 3 + 1 + 1 + 1,
        }
    );
}

#[test]
fn rejects_bad_input() {
    let s = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0], vec![0.0; 3]];
    assert!(build(&s, &[0], None).is_err()); // leaves out item 1 (has entries)
    assert!(build(&[vec![0.0, 1.0], vec![0.0; 2]], &[1], None).is_err()); // leaves out item 0 (a row, no column)
    let m = build(&s, &[1, 0], None).unwrap(); // item 2 dropped
    assert_eq!((m.n_items(), m.n_cols(), m.n_active()), (2, 3, 2));
    let run = |r: &Request| {
        let mut out = (vec![], vec![]);
        recommend(&m, 0, r, &mut out.0, &mut out.1).map(|()| out)
    };
    let r = |ids, w| Request {
        ids,
        weights: w,
        k: 5,
        ..Default::default()
    };
    assert!(run(&r(&[0], Some(&[f32::NAN]))).is_err());
    assert!(run(&r(&[0], Some(&[1.0, 2.0]))).is_err());
    // Unknown ids (never seen, or dropped: item 2) are ignored, everywhere.
    assert_eq!(
        run(&r(&[0, 2, -4, 99], None)).unwrap(),
        (vec![1], vec![1.0])
    );
    let inc = Request {
        include: Some(&[1, 2, 99]),
        exclude: Some(&[2, -1]),
        ..r(&[0], None)
    };
    assert_eq!(run(&inc).unwrap(), (vec![1], vec![1.0]));
    assert_eq!(
        run(&Request {
            include: Some(&[]),
            ..r(&[0], None)
        })
        .unwrap(),
        (vec![], vec![])
    );
    assert_eq!(
        run(&Request {
            exclude: Some(&[1]),
            ..r(&[0], None)
        })
        .unwrap(),
        (vec![], vec![])
    );
    // A huge k is clamped before anything is allocated.
    let big = Request {
        k: usize::MAX / 2,
        ..r(&[0], None)
    };
    assert_eq!(run(&big).unwrap(), (vec![1], vec![1.0]));
    let batch = |offsets: &[i64], ids: &[i64], weights: Option<&[f32]>| {
        recommend_batch(
            &m,
            BlockSize::Off,
            &BatchRequest {
                histories: Lists { offsets, ids },
                weights,
                k: usize::MAX / 2,
                ..Default::default()
            },
        )
    };
    assert_eq!(
        batch(&[0, 1], &[0], None).unwrap(),
        (vec![0, 1], vec![1], vec![1.0])
    );
    assert_eq!(batch(&[0], &[], None).unwrap(), (vec![0], vec![], vec![]));
    // Batch structure: offsets must cover the ids, weights align with them.
    assert!(batch(&[0, 2], &[0], None).is_err());
    assert!(batch(&[1, 1], &[0], None).is_err());
    assert!(batch(&[], &[], None).is_err());
    assert!(batch(&[0, 1], &[0], Some(&[1.0, 2.0])).is_err());
}

/// Owned parts of a model, to corrupt one field at a time.
#[derive(Clone, Debug, PartialEq)]
struct Owned {
    indptr: Vec<i64>,
    pairs: Vec<Entry>,
    new_to_old: Vec<i32>,
    n_cols: usize,
    ext: Option<Vec<i64>>,
    allowed: Option<Vec<u64>>,
}

impl Owned {
    fn of(m: &Model) -> Self {
        let p = m.parts();
        Self {
            indptr: p.indptr.to_vec(),
            pairs: p.pairs.to_vec(),
            new_to_old: p.new_to_old.to_vec(),
            n_cols: p.n_cols,
            ext: p.ext.map(<[i64]>::to_vec),
            allowed: p.allowed.map(<[u64]>::to_vec),
        }
    }

    fn load(self) -> Result<Model, similaripy_core::Error> {
        Model::from_parts(
            self.indptr,
            self.pairs,
            self.new_to_old,
            self.n_cols,
            self.ext,
            self.allowed,
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
    let good = Owned::of(&m.view(Some(&[11]), None));
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
    assert!(bad(&|o| o.pairs[0].col = 3)); // column >= n_items
    assert!(bad(&|o| o.pairs[0].col = -1));
    assert!(bad(&|o| o.pairs.swap(1, 2))); // unsorted row
    assert!(bad(&|o| o.pairs[2].col = o.pairs[1].col)); // duplicate column
    assert!(bad(&|o| {
        o.pairs.pop();
    })); // != nnz
    assert!(bad(&|o| o.pairs[0].val = f32::INFINITY));
    assert!(bad(&|o| o.pairs[0].val = f32::NAN));
    assert!(bad(&|o| o.new_to_old[0] = o.new_to_old[1])); // not distinct
    assert!(bad(&|o| o.new_to_old[0] = 4)); // >= n_cols
    assert!(bad(&|o| o.new_to_old[0] = -1));
    assert!(bad(&|o| o.n_cols = 2)); // new_to_old past n_cols
    assert!(bad(&|o| o.n_cols = usize::MAX)); // no huge allocation
    assert!(bad(&|o| o.ext.as_mut().unwrap().truncate(3)));
    assert!(bad(&|o| o.ext.as_mut().unwrap()[0] = 13)); // duplicate id
    assert!(bad(&|o| o.allowed.as_mut().unwrap().push(0))); // one bit per item
    // Not a corruption: an extra empty item at the end (no row, no column).
    let mut o = good.clone();
    o.n_cols = 5;
    o.ext.as_mut().unwrap().push(14);
    assert!(o.load().is_ok());
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("similaripy-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn save_load_round_trip_and_versions() {
    let s = vec![
        vec![0.0, 1.0, 2.0, 0.0],
        vec![1.0, 0.0, 0.5, 0.0],
        vec![0.0, 3.0, 0.0, 0.0],
        vec![0.0; 4],
    ];
    let m = build(&s, &[1, 0, 2], Some(&[10, 11, 12, 13])).unwrap();
    let v = m.view(Some(&[11, 12]), None);
    let dir = tmp("model");
    let err = v.save(&dir, "test").unwrap_err().to_string();
    assert!(err.contains("save the base model"), "{err}");
    m.save(&dir, "test").unwrap();
    let rec = |m: &Model| {
        let mut out = (vec![], vec![]);
        let r = Request {
            ids: &[10],
            k: 5,
            ..Default::default()
        };
        recommend(m, 2, &r, &mut out.0, &mut out.1).unwrap();
        out
    };
    // Miri can't map files; plain reads run under it
    for mmap in [false, true].into_iter().filter(|&m| !(cfg!(miri) && m)) {
        let back = Model::load(&dir, mmap).unwrap();
        assert_eq!(Owned::of(&back), Owned::of(&m));
        assert_eq!(rec(&back), rec(&m));
    }
    // never into a non-empty directory (a saved model is never replaced); an empty one is fine
    assert!(m.save(&dir, "test").is_err());
    let other = tmp("other");
    std::fs::create_dir_all(&other).unwrap();
    m.save(&other, "test").unwrap();
    assert_eq!(
        Owned::of(&Model::load(&other, false).unwrap()),
        Owned::of(&m)
    );
    // Forward compatibility (spec: docs/saved-format.md): keys and files a
    // reader doesn't know are ignored, `written_by` is optional; a newer
    // format is a clear error.
    let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&meta).unwrap();
    let keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
    assert_eq!(
        keys,
        [
            "format",
            "item_ids",
            "kind",
            "n_cols",
            "n_items",
            "nnz",
            "written_by"
        ]
    );
    let o = v.as_object_mut().unwrap();
    o.remove("written_by");
    for (k, x) in [
        ("future", serde_json::json!({"a": [1, "x"], "b": null})),
        ("saved_at", serde_json::json!("2030-01-01T00:00:00Z")),
        ("a_number", serde_json::json!(1.5)),
        ("a_null", serde_json::Value::Null),
        ("block_size", serde_json::json!(7)), // written by pre-1.0 builds
    ] {
        o.insert(k.into(), x);
    }
    std::fs::write(dir.join("meta.json"), v.to_string()).unwrap();
    std::fs::write(dir.join("README.txt"), "notes").unwrap();
    std::fs::write(dir.join("extra.npy"), "not even numpy").unwrap();
    assert_eq!(Owned::of(&Model::load(&dir, false).unwrap()), Owned::of(&m));
    for (from, to, says) in [
        ("\"format\": 1", "\"format\": 2", "newer"),
        ("\"nnz\": 5", "\"nnz\": 4", "match"),
    ] {
        assert!(meta.contains(from), "{from}");
        std::fs::write(dir.join("meta.json"), meta.replace(from, to)).unwrap();
        let e = Model::load(&dir, false).err().unwrap().to_string();
        assert!(e.contains(says), "{e}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&other).unwrap();
}

/// Savers racing on one path: at most one wins, and a loser's cleanup never
/// removes the winner's files (it used to `remove_dir_all` the directory it
/// created: about one round in five left no model, some after an `Ok`).
#[test]
#[cfg_attr(miri, ignore)] // threads x rounds of file I/O
fn concurrent_saves_to_one_path_keep_the_winners_model() {
    let s = vec![
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.0, 0.5],
        vec![0.0, 3.0, 0.0],
    ];
    let m = build(&s, &[0, 1, 2], Some(&[10, 11, 12])).unwrap();
    let base = tmp("race");
    for round in 0..200 {
        let dir = base.join(round.to_string());
        let ok = std::thread::scope(|sc| {
            let saves: Vec<_> = (0..8).map(|_| sc.spawn(|| m.save(&dir, "test"))).collect();
            saves
                .into_iter()
                .map(|h| h.join().unwrap().is_ok())
                .filter(|&ok| ok)
                .count()
        });
        assert_eq!(ok, 1, "round {round}: {ok} savers returned Ok");
        let back = Model::load(&dir, false).unwrap_or_else(|e| panic!("round {round}: {e}"));
        assert_eq!(Owned::of(&back), Owned::of(&m));
    }
    std::fs::remove_dir_all(&base).unwrap();
}

/// 1.x loads every older 1.x format: these files were written by format 1 and
/// stay as they are (tests/test_serve.py loads them too). Changing what the
/// files mean bumps the format instead.
#[test]
fn loads_the_committed_format_1_model() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/sparseknn-format1");
    for mmap in [false, true].into_iter().filter(|&m| !(cfg!(miri) && m)) {
        let m = Model::load(&dir, mmap).unwrap();
        let mut out = (vec![], vec![]);
        let r = Request {
            ids: &[20, 40],
            k: 3,
            ..Default::default()
        };
        recommend(&m, 2, &r, &mut out.0, &mut out.1).unwrap();
        assert_eq!(out, (vec![30, 10, 50], vec![1.75, 0.75, 0.5]));
    }
}
