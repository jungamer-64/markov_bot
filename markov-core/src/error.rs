use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MarkovError {
    #[error("{0}")]
    Invalid(String),
}

impl From<&str> for MarkovError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

impl From<String> for MarkovError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}
