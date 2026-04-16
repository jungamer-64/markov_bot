use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum MarkovError {
    #[error("Ngram order must be >= 1 and <= u32::MAX, but got {0}")]
    InvalidNgramOrder(usize),

    #[error("Max words must be >= 1, but got {0}")]
    InvalidMaxWords(usize),

    #[error("Temperature must be > 0 and finite, but got {0}")]
    InvalidTemperature(f64),

    #[error("Min words before EOS ({min}) must be <= max words ({max})")]
    InvalidEosThreshold { min: usize, max: usize },

    #[error("Internal boundary error: {0}")]
    Boundary(String),
}

impl From<&str> for MarkovError {
    fn from(value: &str) -> Self {
        Self::Boundary(value.to_owned())
    }
}

impl From<String> for MarkovError {
    fn from(value: String) -> Self {
        Self::Boundary(value)
    }
}
