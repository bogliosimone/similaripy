//! SimilariPy core: top-K sparse similarity kernels in pure Rust.

pub mod csr;
mod kernel;
pub mod normalize;
pub mod prep;
pub mod serve;
mod topk;

pub use csr::{CsrPattern, CsrView};
pub use kernel::{Inputs, Norms, Output, Params, compute};

/// Accumulator block in float32 entries (~1 MB): `block_size='auto'` of the
/// similarity functions and the serving model.
pub const DEFAULT_BLOCK_SIZE: usize = 262_144;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidInput(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for Error {}
