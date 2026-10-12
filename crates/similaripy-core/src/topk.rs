//! Top-K by score, same rule as the C++ `TopK`: a min-heap ordered by
//! (score, col); once full, a candidate replaces the root only if its
//! score is strictly greater.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct Entry {
    score: f32,
    col: i32,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(self.col.cmp(&other.col))
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
            && score > top.0.score
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

    #[test]
    fn keeps_largest_and_ignores_ties_once_full() {
        let mut t = TopK::new(2);
        for (c, s) in [(0, 1.0), (1, 3.0), (2, 2.0), (3, 2.0), (4, 0.5)] {
            t.push(c, s);
        }
        let (mut cols, mut vals) = ([-1; 3], [-1.0; 3]);
        assert_eq!(t.drain_into(&mut cols, &mut vals), 2);
        let mut got: Vec<_> = cols[..2]
            .iter()
            .copied()
            .zip(vals[..2].iter().copied())
            .collect();
        got.sort_by_key(|&(c, _)| c);
        // col 3 ties col 2 at 2.0 and does not replace it
        assert_eq!(got, vec![(1, 3.0), (2, 2.0)]);
        assert_eq!((cols[2], vals[2]), (-1, -1.0));
    }
}
