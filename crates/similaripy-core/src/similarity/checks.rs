//! The value rules (RC-2): options (re-checked for Rust callers: the Python
//! layer checks scalars first, with its own messages), finite values, and the
//! score rules on signed / weighted input. Messages as the Python layer's.

use super::{Options, Pop};
use crate::Error;
use crate::kernel::Params;
use crate::prep::Csr;

fn bad<T>(msg: String) -> Result<T, Error> {
    Err(Error::InvalidInput(msg))
}

/// Scalars finite and >= 0 (a1, walk alpha > 0, k >= 1), depop weights.
pub(crate) fn options(o: &Options, n_rows: usize, n_cols: usize) -> Result<(), Error> {
    let p = &o.params;
    if o.k < 1 {
        return bad("k must be >= 1".into());
    }
    let weights = [
        ("l1", p.l1),
        ("l2", p.l2),
        ("l3", p.l3),
        ("t1", p.t1),
        ("t2", p.t2),
        ("stabilized_shrink", p.stabilized_shrink),
        ("bayesian_shrink", p.bayesian_shrink),
        ("threshold", p.threshold),
        ("c1", o.c1),
        ("c2", o.c2),
        ("p1", o.p1),
        ("p2", o.p2),
        ("additive_shrink", o.additive_shrink),
    ];
    for (name, v) in weights {
        if !(v.is_finite() && v >= 0.0) {
            return bad(format!("{name} must be finite and >= 0, got {v}"));
        }
    }
    for (name, v) in [
        ("a1 (alpha)", Some(p.a1)),
        ("alpha", o.walk.map(|w| w.alpha)),
    ] {
        if let Some(v) = v
            && !(v.is_finite() && v > 0.0)
        {
            return bad(format!("{name} must be finite and > 0, got {v}"));
        }
    }
    pop("pop1", o.pop1, n_rows)?;
    // rp3beta's walk popularity replaces pop2
    let pop2 = if o.walk.is_some_and(|w| w.popularity) {
        Pop::None
    } else {
        o.pop2
    };
    pop("pop2", pop2, n_cols)
}

/// Custom weights: `n` of them, finite and >= 0 (a negative one would flip signs).
fn pop(name: &str, pop: Pop, n: usize) -> Result<(), Error> {
    let Pop::Weights(w) = pop else { return Ok(()) };
    if w.len() != n {
        return bad(format!(
            "{name} must be \"none\", \"sum\" or an array of length {n}, got shape ({},)",
            w.len()
        ));
    }
    if !w.iter().all(|x| (0.0..f32::INFINITY).contains(x)) {
        return bad(format!("{name} weights must be finite and >= 0"));
    }
    Ok(())
}

/// Finite in float32, checked before binary (which must not turn NaN / inf into 1.0).
pub(crate) fn finite(name: &str, data: &[f32]) -> Result<(), Error> {
    if !data.iter().all(|x| x.is_finite()) {
        return bad(format!("{name} values must be finite (in float32)"));
    }
    Ok(())
}

/// The rules on the values the kernel sees (after binary and the walk):
/// bayesian shrink and a1 != 1 rejected on signed input, the tversky
/// denominator rule, custom depop weights > 0 where used. Binary input is
/// never signed.
pub(crate) fn rules(
    m1: &Csr<f32>,
    m2: &Csr<f32>,
    n_cols: usize,
    p: &Params,
    binary: bool,
    pop1: Pop,
    pop2: Pop,
) -> Result<(), Error> {
    let signed = !binary && m1.data.iter().chain(&m2.data).any(|&x| x < 0.0);
    if signed && p.bayesian_shrink != 0.0 {
        return bad(
            "bayesian shrink (shrink_type='bayesian', s_plus bayesian_shrink) is not \
            supported on signed input (values < 0): xy / (xy + shrink) flips the sign \
            of negative scores"
                .into(),
        );
    }
    if signed && p.a1 != 1.0 {
        return bad(format!(
            "a1 (alpha) must be 1 on signed input (values < 0), got {}: \
             a power of a negative score is NaN or turns it positive",
            p.a1
        ));
    }
    // den = t1 |x|^2 + t2 |y|^2 + c xy (c = 1 - t1 - t2), |xy| <= |x| |y|: > 0 on
    // every pair iff c = 0 or 4 t1 t2 > c^2. The worst pair has xy = |x||y| for
    // c < 0 (any weighted input) and xy = -|x||y| for c > 0 (signed input only).
    // Binary, or every value in both matrices the same: den >= v^2 |X & Y| > 0.
    let (t1, t2) = (p.t1, p.t2);
    let c = 1.0 - t1 - t2;
    if p.l1 != 0.0
        && !binary
        && c != 0.0
        && (c < 0.0 || signed)
        && 4.0 * t1 * t2 <= c * c
        && !one_value(&m1.data, &m2.data)
    {
        return bad(format!(
            "t1, t2 (tversky alpha, beta) = ({t1}, {t2}) give a denominator <= 0 on some pairs{}: \
             need t1 + t2 = 1 or 4 t1 t2 > (1 - t1 - t2)^2",
            if c > 0.0 {
                " of signed input (values < 0)"
            } else {
                ""
            }
        ));
    }
    if p.l3 != 0.0 {
        if let Pop::Weights(w) = pop1 {
            let used = m1.indptr.windows(2).map(|r| r[1] > r[0]);
            used_weights("pop1", w, used)?;
        }
        if let Pop::Weights(w) = pop2 {
            let mut used = vec![false; n_cols];
            m2.indices.iter().for_each(|&c| used[c as usize] = true);
            used_weights("pop2", w, used.into_iter())?;
        }
    }
    Ok(())
}

/// Custom weights > 0 on the rows / columns with entries (only those reach a
/// denominator, where a 0 weight would score 0.0).
fn used_weights(name: &str, w: &[f32], used: impl Iterator<Item = bool>) -> Result<(), Error> {
    if w.iter().zip(used).any(|(&x, u)| u && x <= 0.0) {
        return bad(format!(
            "{name} weights must be > 0 on every row/column with entries"
        ));
    }
    Ok(())
}

/// Every stored value of both arrays equal to the same value (false if both empty).
fn one_value(a: &[f32], b: &[f32]) -> bool {
    let mut all = a.iter().chain(b);
    all.next().is_some_and(|&v| all.all(|&x| x == v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(rows: &[&[(i32, f32)]]) -> Csr<f32> {
        let mut m = Csr {
            indptr: vec![0],
            indices: vec![],
            data: vec![],
        };
        for r in rows {
            for &(c, v) in *r {
                m.indices.push(c);
                m.data.push(v);
            }
            m.indptr.push(m.indices.len() as i64);
        }
        m
    }

    fn err(r: Result<(), Error>) -> String {
        r.unwrap_err().to_string()
    }

    #[test]
    fn options_rejects_bad_scalars_and_weights() {
        let ok = Options::default();
        assert!(options(&ok, 2, 3).is_ok());
        let cases: Vec<(Options, &str)> = vec![
            (
                Options {
                    k: 0,
                    ..Default::default()
                },
                "k must be",
            ),
            (
                Options {
                    c1: f32::NAN,
                    ..Default::default()
                },
                "c1 must be finite and >= 0",
            ),
            (
                Options {
                    params: Params {
                        l2: -1.0,
                        ..Params::default()
                    },
                    ..Default::default()
                },
                "l2 must be",
            ),
            (
                Options {
                    params: Params {
                        a1: 0.0,
                        ..Params::default()
                    },
                    ..Default::default()
                },
                "a1 (alpha) must be finite and > 0",
            ),
            (
                Options {
                    walk: Some(super::super::Walk {
                        alpha: 0.0,
                        popularity: false,
                    }),
                    ..Default::default()
                },
                "alpha must be finite and > 0",
            ),
            (
                Options {
                    pop1: Pop::Weights(&[1.0]),
                    ..Default::default()
                },
                "array of length 2, got shape (1,)",
            ),
            (
                Options {
                    pop2: Pop::Weights(&[1.0, -1.0, 1.0]),
                    ..Default::default()
                },
                "pop2 weights must be finite and >= 0",
            ),
            (
                Options {
                    pop2: Pop::Weights(&[1.0, f32::INFINITY, 1.0]),
                    ..Default::default()
                },
                "finite and >= 0",
            ),
        ];
        for (o, msg) in cases {
            let e = err(options(&o, 2, 3));
            assert!(e.contains(msg), "{e:?} should contain {msg:?}");
        }
    }

    #[test]
    fn finite_rejects_nan_and_inf() {
        assert!(finite("matrix1", &[1.0, -2.0]).is_ok());
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(err(finite("matrix2", &[1.0, x])).contains("matrix2 values must be finite"));
        }
    }

    #[test]
    fn rules_on_signed_and_tversky() {
        let pos = csr(&[&[(0, 1.0), (1, 2.0)]]);
        let neg = csr(&[&[(0, 1.0), (1, -2.0)]]);
        let ones = csr(&[&[(0, 1.0), (1, 1.0)]]);
        let p = |f: fn(&mut Params)| {
            let mut p = Params::default();
            f(&mut p);
            p
        };
        let bayes = p(|p| p.bayesian_shrink = 1.0);
        let pow = p(|p| p.a1 = 0.5);
        let tv_low = p(|p| (p.l1, p.t1, p.t2) = (1.0, 0.2, 0.2)); // c = 0.6 > 0: signed only
        let tv_zero = p(|p| (p.l1, p.t1, p.t2) = (1.0, 0.0, 2.0)); // c = -1, 4 t1 t2 = 0
        let r = |m: &Csr<f32>, p: &Params, binary| rules(m, m, 2, p, binary, Pop::None, Pop::None);
        assert!(err(r(&neg, &bayes, false)).contains("not supported on signed input"));
        assert!(err(r(&neg, &pow, false)).contains("a1 (alpha) must be 1 on signed input"));
        assert!(err(r(&neg, &tv_low, false)).contains("pairs of signed input"));
        assert!(err(r(&pos, &tv_zero, false)).contains("denominator <= 0 on some pairs:"));
        for p in [&bayes, &pow, &tv_low] {
            assert!(r(&pos, p, false).is_ok());
            assert!(r(&neg, p, true).is_ok(), "binary is never signed");
        }
        assert!(r(&ones, &tv_zero, false).is_ok(), "one value: den > 0");
        assert!(r(&pos, &tv_zero, true).is_ok());
    }

    #[test]
    fn used_weights_only_where_entries() {
        // m1: row 1 empty; m2: column 2 empty
        let m1 = csr(&[&[(0, 1.0)], &[]]);
        let m2 = csr(&[&[(0, 1.0), (1, 1.0)]]);
        let l3 = Params {
            l3: 1.0,
            ..Params::default()
        };
        let r = |w1: &[f32], w2: &[f32], p: &Params| {
            rules(&m1, &m2, 3, p, false, Pop::Weights(w1), Pop::Weights(w2))
        };
        assert!(r(&[1.0, 0.0], &[1.0, 1.0, 0.0], &l3).is_ok());
        assert!(err(r(&[0.0, 1.0], &[1.0, 1.0, 1.0], &l3)).contains("pop1 weights must be > 0"));
        assert!(err(r(&[1.0, 1.0], &[1.0, 0.0, 1.0], &l3)).contains("pop2 weights must be > 0"));
        assert!(
            r(&[0.0, 0.0], &[0.0; 3], &Params::default()).is_ok(),
            "l3 = 0: unused"
        );
    }

    #[test]
    fn one_value_cases() {
        assert!(one_value(&[2.0, 2.0], &[2.0]));
        assert!(one_value(&[], &[3.0]));
        assert!(!one_value(&[2.0], &[3.0]));
        assert!(!one_value(&[], &[]));
    }
}
