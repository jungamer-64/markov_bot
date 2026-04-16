pub mod chain;
pub mod error;
pub mod options;
pub mod sampling;
pub mod token;

#[cfg(test)]
pub mod test_support;

pub use chain::MarkovChain;
pub use error::MarkovError;
pub use options::{EosPolicy, GenerationOptions};
pub use token::{BOS_ID, BOS_TOKEN, Count, EOS_ID, EOS_TOKEN, Prefix, TokenId};

pub const DEFAULT_NGRAM_ORDER: usize = 6;
pub const DEFAULT_GENERATION_TEMPERATURE: f64 = 1.0;

/// # Errors
/// Returns `MarkovError::Invalid` if `ngram_order` is 0 or greater than `u32::MAX`.
pub fn validate_ngram_order(ngram_order: usize, context: &str) -> Result<(), MarkovError> {
    if ngram_order == 0 {
        return Err(MarkovError::Invalid(format!("{context} must be >= 1")));
    }

    if u32::try_from(ngram_order).is_err() {
        return Err(MarkovError::Invalid(format!(
            "{context} must be <= {}",
            u32::MAX
        )));
    }

    Ok(())
}
