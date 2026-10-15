//! Top-K by score: keeps the k best under (score desc, col asc), so equal
//! scores resolve to the lower column whatever the push order (F4: blocked,
//! single-block and any candidate order give the same rows). A min-heap whose
//! root is the worst kept entry: lowest score, then highest column.
//!
//! Scores must not be NaN or -0.0 (the heap's `total_cmp` and the push's float compares
//! would disagree), and each column with a non-zero score is pushed at most once
//! per row. A 0.0 score is never a result: `drain_into` drops it (an underflowed
//! product re-pushes its column with 0.0, see `kernel::accumulate`).

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct Entry {
    score: f32,
    col: i32,
}

impl Eq for Entry {}

impl Ord for Entry {
    /// Greater = better: higher score, then lower column.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.col.cmp(&self.col))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct TopK {
    k: usize,
    heap: BinaryHeap<Reverse<Entry>>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    #[inline]
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(>=)` is the NaN-safe reject, below
    pub fn push(&mut self, col: i32, score: f32) {
        if self.heap.len() < self.k {
            self.heap.push(Reverse(Entry { score, col }));
            return;
        }
        let Some(&Reverse(root)) = self.heap.peek() else {
            return; // k == 0
        };
        // Almost every scored column is rejected here: one compare and one
        // branch. Ties take the second test; `!(>=)` also rejects NaN, which
        // `score < root` would let through to the tie test.
        if !(score >= root.score) {
            return;
        }
        if (score > root.score || col < root.col)
            && let Some(mut top) = self.heap.peek_mut()
        {
            *top = Reverse(Entry { score, col });
        }
    }

    /// Writes the kept (col, score) pairs with a non-zero score sorted by column, ids renumbered
    /// through `col_map` if given (a permutation), empties the heap and returns
    /// the count. Slots past it are left untouched. Once per row: the sort is
    /// k log k, outside the scoring loop.
    pub fn drain_into(
        &mut self,
        col_map: Option<&[i32]>,
        cols: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        // The heap's own buffer, sorted in place and handed back empty: no allocation.
        let mut kept = std::mem::take(&mut self.heap).into_vec();
        // Per kept entry, not per candidate: a 0.0 can only come from an f32
        // underflow under D6, and it is the duplicate of a column kept above it.
        kept.retain(|Reverse(e)| e.score != 0.0);
        if let Some(map) = col_map {
            for Reverse(e) in &mut kept {
                e.col = map[e.col as usize];
            }
        }
        kept.sort_unstable_by_key(|Reverse(e)| e.col);
        for ((Reverse(e), c), v) in kept.iter().zip(cols).zip(values) {
            *c = e.col;
            *v = e.score;
        }
        let n = kept.len();
        kept.clear();
        self.heap = BinaryHeap::from(kept);
        n
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
    fn drain_renumbers_sorts_and_reuses_the_heap() {
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
    fn keeps_largest_and_lower_column_on_ties() {
        let mut t = TopK::new(2);
        for (c, s) in [(0, 1.0), (1, 3.0), (3, 2.0), (2, 2.0), (4, 0.5)] {
            t.push(c, s);
        }
        // col 2 ties col 3 at 2.0, comes later, and replaces it: lower column wins
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
    fn nan_never_replaces_the_root() {
        // Not produced by drain; guards reuse with other score sources.
        let mut t = TopK::new(1);
        t.push(5, 1.0);
        t.push(0, f32::NAN); // lower column: a `<`-first reject would take the tie path
        assert_eq!(kept(&mut t, 1), vec![(5, 1.0)]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 1000 }))]
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
    }
}
