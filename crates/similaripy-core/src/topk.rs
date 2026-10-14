//! Top-K by score: keeps the k best under (score desc, col asc), so equal
//! scores resolve to the lower column whatever the push order (F4: blocked,
//! single-block and any candidate order give the same rows). A min-heap whose
//! root is the worst kept entry: lowest score, then highest column.
//!
//! Scores must not be NaN or -0.0 (the heap's `total_cmp` and the push's `>`/`==`
//! would disagree), and each column is pushed at most once per row.

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
    pub fn push(&mut self, col: i32, score: f32) {
        if self.heap.len() < self.k {
            self.heap.push(Reverse(Entry { score, col }));
        } else if let Some(mut top) = self.heap.peek_mut()
            && (score > top.0.score || (score == top.0.score && col < top.0.col))
        {
            *top = Reverse(Entry { score, col });
        }
    }

    /// Writes the kept (col, score) pairs in heap order and empties the heap.
    /// Slots past the number of results are left untouched.
    pub fn drain_into(&mut self, cols: &mut [i32], values: &mut [f32]) -> usize {
        let n = self.heap.len();
        for ((Reverse(e), c), v) in self
            .heap
            .drain()
            .zip(cols.iter_mut())
            .zip(values.iter_mut())
        {
            *c = e.col;
            *v = e.score;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    fn kept(t: &mut TopK, n: usize) -> Vec<(i32, f32)> {
        let (mut cols, mut vals) = (vec![-1; n + 1], vec![-1.0; n + 1]);
        let len = t.drain_into(&mut cols, &mut vals);
        assert_eq!((cols[len], vals[len]), (-1, -1.0)); // slots past the results untouched
        let mut got: Vec<_> = cols[..len]
            .iter()
            .copied()
            .zip(vals[..len].iter().copied())
            .collect();
        got.sort_by_key(|&(c, _)| c);
        got
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
