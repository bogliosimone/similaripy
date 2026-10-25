//! Serving: the top-K items for one history from a prepared item-item model S.
//!
//! The model is S made canonical and renumbered in one copy: items with
//! neither a row nor a column are dropped, the rest get internal ids in
//! popularity order (descending column count, ties to the lower column), rows
//! and columns alike, so a history item's internal id is both its row and its
//! column. Columns with entries come first, so the accumulator covers
//! `0..n_active`.
//!
//! A request: the caller's ids → internal ids (dense gather or hash; unknown
//! ids ignored), history sorted with duplicates merged, the slots of seen and
//! excluded items pre-set to `-inf` (never touched, never drained),
//! `scores = h · S` accumulated, then drained through the view's mask and the
//! request's include bits (tested only after the floor reject) into a top-K,
//! best first, internal → caller's ids. Batch = rayon over users calling the
//! same function, so a user gets the same bits from both. Choices measured in
//! step 4.1 (`reports/step4-explore.md` in the planning notes, D64); API D69.
//!
//! Files: `model` (build, the one check, save/load, views), `ids` (id maps),
//! `request` (one history, batch).

mod ids;
mod model;
mod request;

pub use model::{BlockSize, Core, Model, Parts};
pub use request::{Batch, BatchRequest, Lists, Request, recommend, recommend_batch};

use crate::Error;

/// No internal id (a dropped item) / an empty hash slot.
const EMPTY: u32 = u32::MAX;

fn invalid<T>(msg: impl std::fmt::Display) -> Result<T, Error> {
    Err(Error::InvalidInput(msg.to_string()))
}

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

/// # Safety
/// `i / 64 < bits.len()`.
#[inline(always)]
unsafe fn bit(bits: &[u64], i: usize) -> bool {
    // SAFETY: caller contract.
    unsafe { (bits.get_unchecked(i / 64) >> (i % 64)) & 1 != 0 }
}
