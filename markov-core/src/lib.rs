pub mod chain;
pub mod error;
pub mod options;
pub mod sampling;
pub mod token;

#[cfg(test)]
pub mod test_support;

pub use chain::MarkovChain;
pub use error::MarkovError;
pub use options::{EosPolicy, GenerationOptions, MaxWords, MinWordsBeforeEos, Temperature};
pub use token::{BOS_ID, BOS_TOKEN, Count, EOS_ID, EOS_TOKEN, Prefix, TokenId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NgramOrder(u32);

impl NgramOrder {
    pub const DEFAULT: Self = Self(6);

    /// # Errors
    /// Returns `MarkovError::InvalidNgramOrder` if `value` is 0 or greater than `u32::MAX`.
    pub fn new(value: usize) -> Result<Self, MarkovError> {
        if value == 0 {
            return Err(MarkovError::InvalidNgramOrder(value));
        }

        let u32_val = u32::try_from(value).map_err(|_error| MarkovError::InvalidNgramOrder(value))?;
        Ok(Self(u32_val))
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// # Errors
    /// Returns `MarkovError::Boundary` if `u32` to `usize` conversion fails (should be impossible on 32/64-bit).
    pub fn as_usize(self) -> Result<usize, MarkovError> {
        usize::try_from(self.0).map_err(|_error| MarkovError::Boundary("u32 to usize conversion failed".into()))
    }
}
