//! The kernel's normalization terms (x per m1 row, y per m2 column) and the
//! random walk of p3alpha / rp3beta. Sums accumulate in f64 in data order and
//! each term is finished in f64, cast once to f32 (RC-3).

use rayon::prelude::*;

use super::{Pop, Walk};
use crate::Error;
use crate::kernel::{Norms, Params};
use crate::normalize::{Axis, Norm, normalize_checked};
use crate::prep::Csr;

/// The terms the weights use; the others stay None.
#[derive(Default)]
pub(crate) struct Owned {
    pub x_tversky: Option<Vec<f32>>,
    pub y_tversky: Option<Vec<f32>>,
    pub x_cosine: Option<Vec<f32>>,
    pub y_cosine: Option<Vec<f32>>,
    pub x_depop: Option<Vec<f32>>,
    pub y_depop: Option<Vec<f32>>,
}

impl Owned {
    pub fn view(&self) -> Norms<'_> {
        Norms {
            x_tversky: self.x_tversky.as_deref(),
            y_tversky: self.y_tversky.as_deref(),
            x_cosine: self.x_cosine.as_deref(),
            y_cosine: self.y_cosine.as_deref(),
            x_depop: self.x_depop.as_deref(),
            y_depop: self.y_depop.as_deref(),
        }
    }

    /// The per-column arrays the kernel's drain reads (hot bytes, `block_size='auto'`).
    pub fn y_arrays(&self) -> usize {
        [&self.y_tversky, &self.y_cosine, &self.y_depop]
            .iter()
            .filter(|y| y.is_some())
            .count()
    }

    /// Per-column terms in a new column order (`new_to_old`).
    pub fn permute_y(&mut self, new_to_old: &[i32]) {
        for y in [&mut self.y_tversky, &mut self.y_cosine, &mut self.y_depop]
            .into_iter()
            .flatten()
        {
            *y = new_to_old.iter().map(|&c| y[c as usize]).collect();
        }
    }
}

/// The options the terms need besides `Params`.
pub(crate) struct Terms<'a> {
    pub c1: f32,
    pub c2: f32,
    pub p1: f32,
    pub p2: f32,
    pub additive_shrink: f32,
    pub pop1: Pop<'a>,
    pub pop2: Pop<'a>,
}

/// Row sums of `f(x)` in f64, rows in parallel (each row sequential).
fn row_sums(m: &Csr<f32>, f: impl Fn(f32) -> f64 + Sync) -> Vec<f64> {
    m.indptr
        .par_windows(2)
        .map(|w| {
            let row = &m.data[w[0] as usize..w[1] as usize];
            row.iter().map(|&x| f(x)).sum()
        })
        .collect()
}

/// Column sums of `f(x)` in f64, in data order.
// ponytail: serial O(nnz) (~0.3 s est. on Y500M, U3); per-chunk partials
// combined in a fixed order if a profile shows it
fn col_sums(m: &Csr<f32>, n_cols: usize, f: impl Fn(f32) -> f64) -> Vec<f64> {
    let mut s = vec![0.0f64; n_cols];
    for (&c, &x) in m.indices.iter().zip(&m.data) {
        s[c as usize] += f(x);
    }
    s
}

fn cast(v: impl IntoIterator<Item = f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

/// The terms `p` uses: tversky = sums of squares; cosine = (sums of squares +
/// additive)^c; depop = weights^p ('none': ones, 'sum': sums of |x|, or
/// `walk_pop` for m2 when the walk computed it).
pub(crate) fn norms(
    m1: &Csr<f32>,
    m2: &Csr<f32>,
    n_cols: usize,
    p: &Params,
    t: &Terms,
    walk_pop: Option<Vec<f64>>,
) -> Owned {
    let mut out = Owned::default();
    if p.l1 != 0.0 || p.l2 != 0.0 {
        let sq = |x: f32| f64::from(x) * f64::from(x);
        let (x_sq, y_sq) = (row_sums(m1, sq), col_sums(m2, n_cols, sq));
        if p.l2 != 0.0 {
            let a = f64::from(t.additive_shrink);
            let pow = |s: &[f64], c: f32| cast(s.iter().map(|&s| (s + a).powf(f64::from(c))));
            out.x_cosine = Some(pow(&x_sq, t.c1));
            out.y_cosine = Some(pow(&y_sq, t.c2));
        }
        if p.l1 != 0.0 {
            out.x_tversky = Some(cast(x_sq));
            out.y_tversky = Some(cast(y_sq));
        }
    }
    if p.l3 != 0.0 {
        let abs = |x: f32| f64::from(x.abs());
        let n_rows = m1.indptr.len() - 1;
        out.x_depop = Some(depop(t.pop1, t.p1, n_rows, || row_sums(m1, abs)));
        out.y_depop = Some(depop(t.pop2, t.p2, n_cols, || {
            walk_pop.unwrap_or_else(|| col_sums(m2, n_cols, abs))
        }));
    }
    out
}

/// Depop weights `pop^power`: ones, `sums()` or the custom weights.
fn depop(pop: Pop, power: f32, n: usize, sums: impl FnOnce() -> Vec<f64>) -> Vec<f32> {
    let power = f64::from(power);
    match pop {
        Pop::None => vec![1.0; n],
        Pop::Sum => cast(sums().into_iter().map(|s| s.powf(power))),
        Pop::Weights(w) => cast(w.iter().map(|&w| f64::from(w).powf(power))),
    }
}

/// Depop-only weights (rp3beta) through the cosine slots: the kernel's COSINE
/// mode instead of FULL, same bits (both compute `l * (x * y)`; FULL's den
/// adds only zeros to it). A bayesian shrink stays FULL (kernel's choice).
pub(crate) fn depop_as_cosine(p: &mut Params, n: &mut Owned) {
    if p.l1 == 0.0 && p.l2 == 0.0 && p.l3 != 0.0 {
        (p.l2, p.l3) = (p.l3, 0.0);
        n.x_cosine = n.x_depop.take();
        n.y_cosine = n.y_depop.take();
    }
}

/// p3alpha / rp3beta prep on the call's copies: rp3beta's popularity (m2's
/// column sums of |x|, the item degree when binary; returned), then every row
/// l1-normalized (transition probabilities) and raised to alpha keeping its
/// sign (`sign(x) |x|^alpha`; alpha = 1 leaves the values as they are).
pub(crate) fn walk(
    m1: &mut Csr<f32>,
    m2: &mut Csr<f32>,
    n_cols: usize,
    w: Walk,
) -> Result<Option<Vec<f64>>, Error> {
    let pop = w
        .popularity
        .then(|| col_sums(m2, n_cols, |x| f64::from(x.abs())));
    // Rows only: the column count is unused (the driver checked the structure)
    for m in [m1, m2] {
        normalize_checked(Norm::L1, Axis::Rows, &mut m.data, &m.indices, &m.indptr, 0);
        if w.alpha != 1.0 {
            m.data
                .par_iter_mut()
                .for_each(|x| *x = x.abs().powf(w.alpha).copysign(*x));
        }
    }
    Ok(pop)
}

#[cfg(test)]
mod tests {
    use super::*;

    // [[1, -2], [0, 3]] (m1 rows; also m2 = 2 x 2, columns 0 and 1)
    fn m() -> Csr<f32> {
        Csr {
            indptr: vec![0, 2, 3],
            indices: vec![0, 1, 1],
            data: vec![1.0, -2.0, 3.0],
        }
    }

    fn terms() -> Terms<'static> {
        Terms {
            c1: 0.5,
            c2: 1.0,
            p1: 1.0,
            p2: 2.0,
            additive_shrink: 1.0,
            pop1: Pop::Sum,
            pop2: Pop::Sum,
        }
    }

    #[test]
    fn terms_from_f64_sums() {
        let p = Params {
            l1: 1.0,
            l2: 1.0,
            l3: 1.0,
            ..Params::default()
        };
        let n = norms(&m(), &m(), 2, &p, &terms(), None);
        assert_eq!(n.x_tversky.unwrap(), [5.0, 9.0]);
        assert_eq!(n.y_tversky.unwrap(), [1.0, 13.0]);
        // (5 + 1)^0.5, (9 + 1)^0.5; columns (1 + 1)^1, (13 + 1)^1
        assert_eq!(
            n.x_cosine.unwrap(),
            [6f64.sqrt() as f32, 10f64.sqrt() as f32]
        );
        assert_eq!(n.y_cosine.unwrap(), [2.0, 14.0]);
        assert_eq!(n.x_depop.unwrap(), [3.0, 3.0]); // sums of |x|, ^1
        assert_eq!(n.y_depop.unwrap(), [1.0, 25.0]); // (1, 5)^2
        // unused terms stay None
        let n = norms(&m(), &m(), 2, &Params::default(), &terms(), None);
        assert!(n.x_tversky.is_none() && n.x_cosine.is_none() && n.y_depop.is_none());
    }

    #[test]
    fn depop_none_custom_and_walk_pop() {
        let p = Params {
            l3: 1.0,
            ..Params::default()
        };
        let w = [2.0f32, 3.0];
        let t = Terms {
            pop1: Pop::None,
            pop2: Pop::Weights(&w),
            ..terms()
        };
        let n = norms(&m(), &m(), 2, &p, &t, None);
        assert_eq!(n.x_depop.unwrap(), [1.0, 1.0]);
        assert_eq!(n.y_depop.unwrap(), [4.0, 9.0]);
        let n = norms(&m(), &m(), 2, &p, &terms(), Some(vec![2.0, 4.0]));
        assert_eq!(
            n.y_depop.unwrap(),
            [4.0, 16.0],
            "walk popularity replaces the sums"
        );
    }

    #[test]
    fn depop_only_moves_to_cosine_slots() {
        let mut p = Params {
            l3: 2.0,
            ..Params::default()
        };
        let mut n = Owned {
            x_depop: Some(vec![1.0]),
            y_depop: Some(vec![2.0]),
            ..Owned::default()
        };
        depop_as_cosine(&mut p, &mut n);
        assert_eq!((p.l2, p.l3), (2.0, 0.0));
        assert_eq!((n.x_cosine, n.y_cosine), (Some(vec![1.0]), Some(vec![2.0])));
        assert!(n.x_depop.is_none() && n.y_depop.is_none());
    }

    #[test]
    fn walk_normalizes_keeps_signs_and_returns_popularity() {
        let (mut a, mut b) = (m(), m());
        let pop = walk(
            &mut a,
            &mut b,
            2,
            Walk {
                alpha: 0.5,
                popularity: true,
            },
        )
        .unwrap();
        assert_eq!(pop.unwrap(), [1.0, 5.0]);
        let want = [(1.0f32 / 3.0).powf(0.5), -(2.0f32 / 3.0).powf(0.5), 1.0];
        // close, not equal: Miri adds random error to powf
        for d in [&a.data, &b.data] {
            assert!(
                d.iter().zip(&want).all(|(x, w)| (x - w).abs() <= 1e-6),
                "{d:?}"
            );
        }
        let (mut a, mut b) = (m(), m());
        assert!(
            walk(
                &mut a,
                &mut b,
                2,
                Walk {
                    alpha: 1.0,
                    popularity: false
                }
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(a.data, [1.0 / 3.0, -2.0 / 3.0, 1.0]);
    }

    #[test]
    fn permute_y_and_count() {
        let mut n = Owned {
            y_cosine: Some(vec![1.0, 2.0, 3.0]),
            y_depop: Some(vec![4.0, 5.0, 6.0]),
            ..Owned::default()
        };
        n.permute_y(&[2, 0, 1]);
        assert_eq!(n.y_cosine.as_deref(), Some(&[3.0, 1.0, 2.0][..]));
        assert_eq!(n.y_depop.as_deref(), Some(&[6.0, 4.0, 5.0][..]));
        assert_eq!(n.y_arrays(), 2);
    }
}
