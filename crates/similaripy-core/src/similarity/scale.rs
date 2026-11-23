//! Product form: a score `xy / (l X[row] Y[col])` is the plain dot product of
//! m1 rows scaled by 1 / (l X) and m2 columns by 1 / Y, so the kernel runs
//! DOT instead of COSINE.

use rayon::prelude::*;

use super::norms::Owned;
use crate::Error;
use crate::kernel::Params;
use crate::prep::{Csr, LEAF_NNZ, par_leaves_one};

/// One cosine term (or depop, after `depop_as_cosine`; an additive shrink sits
/// inside the norms), no power, no other term or shrink.
pub(crate) fn product_form(p: &Params) -> bool {
    p.a1 == 1.0
        && p.l1 == 0.0
        && (p.l2 != 0.0) != (p.l3 != 0.0)
        && p.stabilized_shrink == 0.0
        && p.bayesian_shrink == 0.0
}

/// Scales m1 rows by 1 / (l2 X) and m2 columns by 1 / Y in place (the cosine
/// slots); 0 where a norm is 0 (an empty row / column: the kernel scores 0
/// there too). A scaled value may underflow to 0.0: scored 0, never a result.
///
/// Errors on values out of float32 range for the score: a scale 0 or inf on a
/// row / column with entries (its norm underflows or overflows, e.g. data
/// x 1e-25 or x 1e19), or a scaled value that overflows. Every m1 row is
/// checked, targeted or not (scaled whole).
pub(crate) fn prescale(
    m1: &mut Csr<f32>,
    m2: &mut Csr<f32>,
    l2: f32,
    n: &Owned,
) -> Result<(), Error> {
    let (x, y) = (
        n.x_cosine.as_deref().unwrap(),
        n.y_cosine.as_deref().unwrap(),
    );
    let inv = |d: f32| if d != 0.0 { 1.0 / d } else { 0.0 };
    let sx: Vec<f32> = x.iter().map(|&x| inv(l2 * x)).collect();
    let sy: Vec<f32> = y.iter().map(|&y| inv(y)).collect();
    let fine = |s: f32| s.is_finite() && s != 0.0;
    let mut ok = true;
    for (r, w) in m1.indptr.windows(2).enumerate() {
        ok &= w[0] == w[1] || fine(sx[r]);
    }
    par_leaves_one(&m1.indptr, &mut m1.data, LEAF_NNZ, &|row0, ip, d| {
        let base = ip[0] as usize;
        for (i, w) in ip.windows(2).enumerate() {
            let s = sx[row0 + i];
            d[w[0] as usize - base..w[1] as usize - base]
                .iter_mut()
                .for_each(|v| *v *= s);
        }
    });
    m2.data
        .par_iter_mut()
        .zip(&m2.indices)
        .for_each(|(v, &c)| *v *= sy[c as usize]);
    ok &= m2.indices.iter().all(|&c| fine(sy[c as usize]));
    ok &= m1.data.iter().chain(&m2.data).all(|v| v.is_finite());
    if !ok {
        return Err(Error::InvalidInput(
            "input values out of float32 range for this similarity (a norm or a \
             scaled value underflows or overflows): rescale the input"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(indptr: Vec<i64>, indices: Vec<i32>, data: Vec<f32>) -> Csr<f32> {
        Csr {
            indptr,
            indices,
            data,
        }
    }

    fn norms(x: Vec<f32>, y: Vec<f32>) -> Owned {
        Owned {
            x_cosine: Some(x),
            y_cosine: Some(y),
            ..Owned::default()
        }
    }

    #[test]
    fn product_form_cases() {
        let p = |f: fn(&mut Params)| {
            let mut p = Params::default();
            f(&mut p);
            product_form(&p)
        };
        assert!(p(|p| p.l2 = 1.0));
        assert!(p(|p| p.l3 = 1.0));
        assert!(!p(|_| ()), "no term: plain dot already");
        assert!(!p(|p| (p.l2, p.l3) = (1.0, 1.0)));
        assert!(!p(|p| (p.l2, p.a1) = (1.0, 0.5)));
        assert!(!p(|p| (p.l2, p.l1) = (1.0, 1.0)));
        assert!(!p(|p| (p.l2, p.stabilized_shrink) = (1.0, 1.0)));
        assert!(!p(|p| (p.l2, p.bayesian_shrink) = (1.0, 1.0)));
    }

    #[test]
    fn scales_rows_and_columns() {
        // m1 = [[2, 4], [], [1]] (row 1 empty: norm 0, never checked)
        let mut m1 = csr(vec![0, 2, 2, 3], vec![0, 1, 0], vec![2.0, 4.0, 1.0]);
        // m2 = [[3, 6]]: columns 0, 1; column 2 empty (norm 0)
        let mut m2 = csr(vec![0, 2], vec![0, 1], vec![3.0, 6.0]);
        let n = norms(vec![2.0, 0.0, 4.0], vec![3.0, 2.0, 0.0]);
        prescale(&mut m1, &mut m2, 0.5, &n).unwrap();
        assert_eq!(m1.data, [2.0, 4.0, 0.5]); // 1 / (0.5 * 2) = 1, 1 / (0.5 * 4) = 0.5
        assert_eq!(m2.data, [1.0, 3.0]);
    }

    #[test]
    fn rejects_out_of_range() {
        let m = || csr(vec![0, 1], vec![0], vec![1.0]);
        let err = |x: f32, y: f32| {
            prescale(&mut m(), &mut m(), 1.0, &norms(vec![x], vec![y]))
                .unwrap_err()
                .to_string()
        };
        for (x, y) in [(1e-45, 1.0), (1.0, f32::INFINITY), (0.0, 1.0)] {
            assert!(err(x, y).contains("out of float32 range"), "{x} {y}");
        }
        // a scaled value that overflows: 1e30 / 1e-20
        let mut big = csr(vec![0, 1], vec![0], vec![1e30]);
        let e = prescale(&mut big, &mut m(), 1.0, &norms(vec![1e-20], vec![1.0])).unwrap_err();
        assert!(e.to_string().contains("out of float32 range"));
    }
}
