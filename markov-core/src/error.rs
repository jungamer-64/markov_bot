use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum MarkovError {
    #[error("Ngram order must be >= 1, but got {0}")]
    InvalidNgramOrder(usize),

    #[error("Max words must be >= 1, but got {0}")]
    InvalidMaxWords(usize),

    #[error("Temperature must be > 0 and finite, but got {0}")]
    InvalidTemperature(f64),

    #[error("Min words before EOS ({min}) must be <= max words ({max})")]
    InvalidEosThreshold { min: usize, max: usize },

    #[error("Internal boundary error: {0}")]
    Boundary(String),

    #[error("Token count exceeded u32::MAX")]
    TokenLimitExceeded,

    #[error("Training window is unexpectedly empty")]
    EmptyTrainingWindow,

    #[error("Failed to get start prefix range")]
    StartPrefixRangeError,

    #[error("Training prefix range is invalid")]
    InvalidTrainingPrefixRange,

    #[error("Training model index is out of bounds")]
    ModelIndexOutOfBounds,
}
