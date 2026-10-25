//! Id maps: the caller's int64 ids → internal ids (dense or hashed).

use super::model::Core;
use super::{EMPTY, invalid, words};
use crate::Error;
use crate::store::Buf;

#[inline(always)]
fn hash(key: i64, mult: u64, shift: u32) -> usize {
    ((key as u64).wrapping_mul(mult) >> shift) as usize
}

/// External int64 ids: open addressing, multiply-shift hash, linear probing,
/// load <= 0.5 (~2 ns per lookup, D64).
pub(super) struct ExtIds {
    /// External id per original column, dropped items too (`parts`).
    pub(super) by_col: Buf<i64>,
    /// External id per internal id.
    pub(super) out: Vec<i64>,
    /// (external id, internal id); internal `EMPTY` = free slot.
    slots: Vec<(i64, u32)>,
    /// Random odd multiplier per table: ids crafted to collide under a
    /// fixed one (a saved model from elsewhere) would make probing quadratic.
    mult: u64,
    shift: u32,
}

impl ExtIds {
    pub(super) fn new(by_col: Buf<i64>, new_to_old: &[i32]) -> Result<Self, Error> {
        let mut sorted = by_col.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return invalid("item_ids must be unique");
        }
        drop(sorted);
        let b: &[i64] = &by_col;
        let out: Vec<i64> = new_to_old.iter().map(|&c| b[c as usize]).collect();
        let cap = (2 * out.len()).next_power_of_two().max(2);
        let shift = 64 - cap.trailing_zeros();
        let mult = std::hash::BuildHasher::hash_one(&std::hash::RandomState::new(), 0u8) | 1;
        let mut slots = vec![(0i64, EMPTY); cap];
        for (internal, &key) in out.iter().enumerate() {
            let mut s = hash(key, mult, shift);
            while slots[s].1 != EMPTY {
                s = (s + 1) & (cap - 1);
            }
            slots[s] = (key, internal as u32);
        }
        Ok(Self {
            by_col,
            out,
            slots,
            mult,
            shift,
        })
    }

    #[inline]
    fn get(&self, id: i64) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut s = hash(id, self.mult, self.shift);
        loop {
            let (key, v) = self.slots[s];
            if v == EMPTY {
                return None;
            }
            if key == id {
                return Some(v);
            }
            s = (s + 1) & mask;
        }
    }
}

/// Original id → internal id (`EMPTY`: dropped); `new_to_old` must list
/// distinct ids < `n_cols`.
pub(super) fn dense_in(new_to_old: &[i32], n_cols: usize) -> Result<Vec<u32>, Error> {
    if n_cols > i32::MAX as usize {
        return invalid("more than 2^31 - 1 items");
    }
    let mut dense_in = vec![EMPTY; n_cols];
    for (new, &old) in new_to_old.iter().enumerate() {
        match usize::try_from(old).ok().and_then(|o| dense_in.get_mut(o)) {
            Some(d) if *d == EMPTY => *d = new as u32,
            _ => return invalid("new_to_old must list distinct item ids"),
        }
    }
    Ok(dense_in)
}

impl Core {
    /// The internal id of a caller's id (an external id, or S's column index
    /// without them); None for ids the model doesn't have.
    #[inline]
    pub(super) fn internal(&self, id: i64) -> Option<u32> {
        match &self.ext {
            Some(ext) => ext.get(id),
            None => match usize::try_from(id).ok().and_then(|c| self.dense_in.get(c)) {
                Some(&d) if d != EMPTY => Some(d),
                _ => None,
            },
        }
    }

    /// Appends (internal id, weight) per known id, in input order; ids the
    /// model doesn't have are skipped.
    pub(super) fn lookup_into(
        &self,
        ids: &[i64],
        weights: Option<&[f32]>,
        out: &mut Vec<(u32, f32)>,
    ) {
        for (i, &id) in ids.iter().enumerate() {
            if let Some(d) = self.internal(id) {
                out.push((d, weights.map_or(1.0, |w| w[i])));
            }
        }
    }

    /// A bitset over internal ids with the known `ids` set.
    pub(super) fn bits_of(&self, ids: &[i64]) -> Vec<u64> {
        let mut bits = vec![0u64; words(self.n_items())];
        for &id in ids {
            if let Some(i) = self.internal(id) {
                bits[i as usize / 64] |= 1 << (i % 64);
            }
        }
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ids crafted to share one slot under the old fixed multiplier
    // (k_i = i * C^-1 mod 2^64): a random multiplier spreads them.
    #[test]
    fn crafted_ids_dont_pile_up() {
        const C: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut inv = C; // Newton: inv = C^-1 mod 2^64
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(C.wrapping_mul(inv)));
        }
        assert_eq!(C.wrapping_mul(inv), 1);
        let n: usize = if cfg!(miri) { 64 } else { 4096 };
        let keys: Vec<i64> = (1..=n as u64).map(|i| i.wrapping_mul(inv) as i64).collect();
        let order: Vec<i32> = (0..n as i32).collect();
        let ids = ExtIds::new(keys.clone().into(), &order).unwrap();
        let mask = ids.slots.len() - 1;
        let max_probe = keys
            .iter()
            .map(|&k| {
                let home = hash(k, ids.mult, ids.shift);
                let at = ids.slots.iter().position(|s| s.0 == k).unwrap();
                (at.wrapping_sub(home)) & mask
            })
            .max()
            .unwrap();
        assert!(max_probe < 64, "max probe {max_probe}");
        assert!(
            keys.iter()
                .enumerate()
                .all(|(i, &k)| ids.get(k) == Some(i as u32))
        );
    }
}
