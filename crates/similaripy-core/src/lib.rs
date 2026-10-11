//! SimilariPy core: top-K sparse similarity kernels in pure Rust.

pub mod csr;
mod kernel;
mod topk;

pub use csr::{CsrPattern, CsrView};
pub use kernel::{Inputs, Norms, Output, Params, Schedule, compute};

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
