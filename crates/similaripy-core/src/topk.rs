//! Top-K by score: keeps the k best under (score desc, col asc), so equal
//! scores resolve to the lower column whatever the push order (blocked,
//! single-block and any candidate order give the same rows).
//!
//! An unsorted buffer of up to 2k entries: `push` appends (one compare against
//! the floor, one store), and a full buffer is cut back to the k best by
//! `select_nth_unstable`, which raises the floor to the k-th best score. Per
//! accepted candidate that is O(1) amortized, against a heap's log k
//! unpredictable branches (4c D89: kernel −5..−18%, serving latency at cap 50
//! −11..−20% vs the heap on the M2; same results bit for bit). Between cuts the
//! floor lags (more pushes than a heap would take).
//!
//! Scores must not be NaN, and each column with a non-zero score is pushed at most
//! once per row. A 0.0 score is never a result: the drains drop it (a sum that
//! cancels on signed input, or an underflowed product, re-pushes its column with
//! 0.0, see `kernel::accumulate`). -0.0 (a negative score that underflows, signed
//! input) is harmless although the cut's `total_cmp` orders it below 0.0 and the
//! push's float compare does not: a +-0.0 entry never outranks a positive score
//! at a cut, and the drains drop both.

/// An entry as one integer in the top-K's order, better = smaller: the score
/// in `total_cmp` order, inverted (higher first), above the column (lower
/// first). One integer compare per step of the cut and the sorts, where the
/// (score, col) pair took a `total_cmp` and a column tie-break (D2: the cut and
/// the final sort are 25-35% of a cap-10 serving request).
#[inline(always)]
fn key(score: f32, col: i32) -> u64 {
    let t = score.to_bits() as i32;
    let t = t ^ (((t >> 31) as u32) >> 1) as i32; // total_cmp order as i32
    let hi = !((t as u32) ^ 0x8000_0000); // as u32, inverted: higher score first
    (u64::from(hi) << 32) | u64::from((col as u32) ^ 0x8000_0000)
}

/// `key`'s inverse: (score, col).
#[inline(always)]
fn unkey(key: u64) -> (f32, i32) {
    let t = (!((key >> 32) as u32) ^ 0x8000_0000) as i32;
    let bits = t ^ (((t >> 31) as u32) >> 1) as i32; // the map is its own inverse
    (
        f32::from_bits(bits as u32),
        ((key as u32) ^ 0x8000_0000) as i32,
    )
}

pub struct TopK {
    k: usize,
    /// Buffer length that triggers a cut: `cap_for(k)`.
    cap: usize,
    /// `key`s, unordered between cuts.
    buf: Vec<u64>,
    /// Lowest score push accepts: -inf until the first cut, then the k-th best.
    min: f32,
    cut: bool,
}

/// Cut when the buffer holds 2k entries (4c tonight9: 1.5k cuts too often, 4k
/// gains nothing); 1 for k = 0, which never pushes.
fn cap_for(k: usize) -> usize {
    k.saturating_mul(2).max(1)
}

/// k = 0 until `set_k` (serving's per-thread scratch); `min` starts at -inf
/// as in `new` (a derived 0.0 would reject negative scores before the first cut).
impl Default for TopK {
    fn default() -> Self {
        Self::new(0)
    }
}

impl TopK {
    pub fn new(k: usize) -> Self {
        let cap = cap_for(k);
        Self {
            k,
            cap,
            buf: Vec::with_capacity(cap),
            min: f32::NEG_INFINITY,
            cut: false,
        }
    }

    #[inline]
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(>=)` is the NaN-safe reject, below
    pub fn push(&mut self, col: i32, score: f32) {
        debug_assert!(!score.is_nan(), "TopK::push: NaN score");
        // Rejects NaN too; ties pass and the next cut decides them by column.
        if !(score >= self.min) || self.k == 0 {
            return;
        }
        self.buf.push(key(score, col));
        if self.buf.len() == self.cap {
            self.cut_to_k();
        }
    }

    /// Keeps the k best (unordered) and raises `min` to the k-th best score.
    /// Out of line (`#[cold]`): once per k accepted pushes, and inlined into the
    /// drain loops it would cost registers there (benchmarks/README.md,
    /// "Codegen-sensitive hot loops").
    #[cold]
    #[inline(never)]
    fn cut_to_k(&mut self) {
        self.buf.select_nth_unstable(self.k - 1);
        self.buf.truncate(self.k);
        self.min = unkey(self.buf[self.k - 1]).0;
        self.cut = true;
    }

    /// Sets k for the next row (serving reuses one `TopK` across requests with
    /// different k). The buffer must be empty, as after a drain.
    pub fn set_k(&mut self, k: usize) {
        debug_assert!(self.buf.is_empty(), "TopK::set_k on a non-empty buffer");
        self.k = k;
        self.cap = cap_for(k);
        self.buf.reserve(self.cap);
    }

    /// The lowest score `push` can still accept, for a caller that only pushes
    /// scores `>= threshold`: `threshold` until the first cut, then the k-th
    /// best at the last cut (`>= threshold`), `+inf` for k == 0. An equal
    /// score still goes to `push` (the column decides ties).
    #[inline]
    pub fn floor(&self, threshold: f32) -> f32 {
        if self.k == 0 {
            f32::INFINITY
        } else if self.cut {
            self.min
        } else {
            threshold
        }
    }

    /// Writes the kept (col, score) pairs with a non-zero score sorted by
    /// column, ids renumbered through `col_map` if given (a permutation),
    /// empties the buffer and returns the count. Slots past it are left
    /// untouched. Once per row: the sort is k log k, outside the scoring loop.
    pub fn drain_into(
        &mut self,
        col_map: Option<&[i32]>,
        cols: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        self.keep_k();
        // Re-keyed by (column, score bits): the integer sort orders by column
        // (unique once zeros are dropped).
        for e in &mut self.buf {
            let (score, col) = unkey(*e);
            let col = col_map.map_or(col, |map| map[col as usize]);
            *e = (u64::from((col as u32) ^ 0x8000_0000) << 32) | u64::from(score.to_bits());
        }
        self.buf.sort_unstable();
        for ((&e, c), v) in self.buf.iter().zip(cols).zip(values) {
            *c = (((e >> 32) as u32) ^ 0x8000_0000) as i32;
            *v = f32::from_bits(e as u32);
        }
        let n = self.buf.len();
        self.reset();
        n
    }

    /// Appends the kept (col, score) pairs with a non-zero score best first
    /// (score desc, col asc; `sorted = false`: any order, the same pairs) to
    /// `cols` / `values` and empties the buffer.
    pub fn drain_by_score(&mut self, cols: &mut Vec<i32>, values: &mut Vec<f32>, sorted: bool) {
        self.keep_k();
        if sorted {
            self.buf.sort_unstable();
        }
        cols.extend(self.buf.iter().map(|&e| unkey(e).1));
        values.extend(self.buf.iter().map(|&e| unkey(e).0));
        self.reset();
    }

    /// Cuts to the k best and drops +-0.0 entries (a cancelled sum or an f32
    /// underflow, never evidence, and may duplicate a column kept above it).
    fn keep_k(&mut self) {
        if self.buf.len() > self.k {
            self.cut_to_k();
        }
        self.buf.retain(|&e| unkey(e).0 != 0.0);
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.min = f32::NEG_INFINITY;
        self.cut = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    fn kept(t: &mut TopK, n: usize) -> Vec<(i32, f32)> {
        let (mut cols, mut vals) = (vec![-1; n + 1], vec![-1.0; n + 1]);
        let len = t.drain_into(None, &mut cols, &mut vals);
        assert_eq!((cols[len], vals[len]), (-1, -1.0)); // slots past the results untouched
        cols[..len]
            .iter()
            .copied()
            .zip(vals[..len].iter().copied())
            .collect()
    }

    #[test]
    fn drain_renumbers_sorts_and_reuses_the_buffer() {
        let mut t = TopK::new(3);
        for (c, s) in [(0, 1.0), (1, 3.0), (2, 2.0)] {
            t.push(c, s);
        }
        let (mut cols, mut vals) = ([0; 3], [0.0; 3]);
        // kernel columns 0, 1, 2 are original columns 7, 5, 6
        assert_eq!(t.drain_into(Some(&[7, 5, 6]), &mut cols, &mut vals), 3);
        assert_eq!((cols, vals), ([5, 6, 7], [3.0, 2.0, 1.0]));
        t.push(4, 1.0); // empty again, same capacity
        assert_eq!(kept(&mut t, 3), vec![(4, 1.0)]);
    }

    #[test]
    fn drain_by_score_orders_best_first_and_set_k_reuses() {
        let mut t = TopK::new(3);
        for (c, s) in [(4, 1.0), (1, 3.0), (2, 1.0), (0, 0.0)] {
            t.push(c, s);
        }
        let (mut cols, mut vals) = (vec![], vec![]);
        t.drain_by_score(&mut cols, &mut vals, true);
        assert_eq!((cols, vals), (vec![1, 2, 4], vec![3.0, 1.0, 1.0]));
        t.set_k(1);
        t.push(7, 2.0);
        t.push(3, 2.0);
        let (mut cols, mut vals) = (vec![], vec![]);
        t.drain_by_score(&mut cols, &mut vals, true);
        assert_eq!((cols, vals), (vec![3], vec![2.0]));
    }

    #[test]
    fn keeps_largest_and_lower_column_on_ties() {
        let mut t = TopK::new(2);
        for (c, s) in [(0, 1.0), (1, 3.0), (3, 2.0), (2, 2.0), (4, 0.5)] {
            t.push(c, s);
        }
        // col 2 ties col 3 at 2.0 and comes later: the cut keeps the lower column
        assert_eq!(kept(&mut t, 2), vec![(1, 3.0), (2, 2.0)]);
    }

    #[test]
    fn drops_zero_scores() {
        // An underflowed product re-pushes its column with 0.0 (kernel::accumulate).
        let mut t = TopK::new(5);
        for (c, s) in [(0, 1.0), (0, 0.0), (2, 0.0)] {
            t.push(c, s);
        }
        assert_eq!(kept(&mut t, 3), vec![(0, 1.0)]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "NaN score")]
    fn nan_push_asserts_in_debug() {
        TopK::new(1).push(0, f32::NAN);
    }

    #[test]
    #[cfg(not(debug_assertions))] // debug builds stop at the assert above
    fn nan_is_never_kept() {
        // Not produced by drain; guards reuse with other score sources.
        let mut t = TopK::new(1);
        t.push(5, 1.0);
        t.push(0, f32::NAN); // `!(score >= min)` rejects it before the buffer
        assert_eq!(kept(&mut t, 1), vec![(5, 1.0)]);
    }

    #[test]
    fn floor_is_threshold_until_the_first_cut_then_the_kth_best() {
        let mut t = TopK::new(2);
        assert_eq!(t.floor(0.5), 0.5);
        for (c, s) in [(0, 3.0), (1, 2.0), (2, 4.0)] {
            t.push(c, s);
            assert_eq!(t.floor(0.5), 0.5);
        }
        t.push(3, 1.0); // 2k = 4 entries: cut to {4.0, 3.0}
        assert_eq!(t.floor(0.5), 3.0);
        t.push(4, 2.5); // below the floor: rejected
        assert_eq!(kept(&mut t, 2), vec![(0, 3.0), (2, 4.0)]);
        assert_eq!(t.floor(0.5), 0.5); // drained: back to the threshold
        assert_eq!(TopK::new(0).floor(0.5), f32::INFINITY);
    }

    #[test]
    fn key_orders_as_score_desc_then_col_asc() {
        let scores = [
            f32::NEG_INFINITY,
            -2.5,
            -f32::MIN_POSITIVE,
            -1e-45,
            -0.0,
            0.0,
            1e-45,
            0.5,
            1.0,
            f32::MAX,
            f32::INFINITY,
        ];
        let cols = [i32::MIN, -1, 0, 1, 7, i32::MAX];
        let all: Vec<(f32, i32)> = scores
            .iter()
            .flat_map(|&s| cols.iter().map(move |&c| (s, c)))
            .collect();
        for &(sa, ca) in &all {
            assert_eq!(unkey(key(sa, ca)).0.to_bits(), sa.to_bits());
            assert_eq!(unkey(key(sa, ca)).1, ca);
            for &(sb, cb) in &all {
                let want = sb.total_cmp(&sa).then(ca.cmp(&cb)); // better first
                assert_eq!(
                    key(sa, ca).cmp(&key(sb, cb)),
                    want,
                    "({sa}, {ca}) vs ({sb}, {cb})"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 1000 }))]
        // Any non-NaN scores and columns: the key's integer order is the
        // top-K's order, and unkey inverts key bit for bit.
        #[test]
        fn key_matches_the_order_on_any_bits(a in any::<u32>(), b in any::<u32>(), ca in any::<i32>(), cb in any::<i32>()) {
            let (sa, sb) = (f32::from_bits(a), f32::from_bits(b));
            prop_assume!(!sa.is_nan() && !sb.is_nan());
            prop_assert_eq!(unkey(key(sa, ca)).1, ca);
            prop_assert_eq!(unkey(key(sa, ca)).0.to_bits(), a);
            prop_assert_eq!(key(sa, ca).cmp(&key(sb, cb)), sb.total_cmp(&sa).then(ca.cmp(&cb)));
        }

        // Any push order keeps the same entries: the brute-force k best under
        // (score desc, col asc). Few distinct scores, so ties are everywhere.
        #[test]
        fn result_independent_of_push_order(
            (items, k) in prop::collection::vec(1u8..4, 1..40).prop_flat_map(|s| {
                let items: Vec<(i32, f32)> =
                    s.iter().enumerate().map(|(c, &v)| (c as i32, f32::from(v))).collect();
                let n = items.len();
                (Just(items).prop_shuffle(), 1..=n + 2)
            })
        ) {
            let mut want = items.clone();
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            want.truncate(k);
            want.sort_by_key(|&(c, _)| c);
            let mut t = TopK::new(k);
            for &(c, s) in &items {
                t.push(c, s);
            }
            prop_assert_eq!(kept(&mut t, k), want);
        }

        // The callers' contract: one `TopK` from `Default` reused across rows
        // with `set_k` (serving), pushes gated by `val >= floor(threshold)` as
        // the drains do, 0.0 re-pushes of already-pushed columns, k = 0..n+2.
        // Each row keeps the brute-force k best non-zero entries.
        #[test]
        fn reused_and_floor_gated_rows_match_brute_force(
            rows in prop::collection::vec(
                prop::collection::vec((0u8..4, prop::bool::ANY), 0..30)
                    .prop_flat_map(|s| {
                        let n = s.len();
                        (Just(s).prop_shuffle(), 0..=n + 2)
                    }),
                1..6,
            )
        ) {
            let mut t = TopK::default();
            for (row, k) in rows {
                // column c: score v at index c; a `true` flag re-pushes c with 0.0 later
                let mut pushes: Vec<(i32, f32)> =
                    row.iter().enumerate().map(|(c, &(v, _))| (c as i32, f32::from(v))).collect();
                pushes.extend(row.iter().enumerate().filter(|(_, (_, z))| *z).map(|(c, _)| (c as i32, 0.0)));
                let mut want: Vec<(i32, f32)> = pushes.iter().copied().filter(|&(_, s)| s != 0.0).collect();
                want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                want.truncate(k);
                want.sort_by_key(|&(c, _)| c);
                t.set_k(k);
                for (c, s) in pushes {
                    if s >= t.floor(0.0) {
                        t.push(c, s);
                    }
                }
                prop_assert_eq!(kept(&mut t, k), want);
            }
        }
    }

    #[test]
    fn default_then_set_k_accepts_negative_scores() {
        // A derived Default (min = 0.0) rejected these before the first cut.
        let mut t = TopK::default();
        t.set_k(2);
        t.push(0, -1.0);
        t.push(1, -2.0);
        assert_eq!(kept(&mut t, 2), vec![(0, -1.0), (1, -2.0)]);
    }
}
