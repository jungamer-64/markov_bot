use crate::error::MarkovError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EosPolicy {
    Forbidden,
    Allowed,
}

impl From<bool> for EosPolicy {
    fn from(value: bool) -> Self {
        if value {
            Self::Allowed
        } else {
            Self::Forbidden
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    max_words: usize,
    temperature: f64,
    min_words_before_eos: usize,
}

impl GenerationOptions {
    /// # Errors
    /// Returns `MarkovError::Invalid` if the options are invalid.
    pub fn new(
        max_words: usize,
        temperature: f64,
        min_words_before_eos: usize,
    ) -> Result<Self, MarkovError> {
        if max_words == 0 {
            return Err("max_words must be > 0".into());
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("temperature must be > 0 and finite".into());
        }
        if min_words_before_eos > max_words {
            return Err("min_words_before_eos must be <= max_words".into());
        }

        Ok(Self {
            max_words,
            temperature,
            min_words_before_eos,
        })
    }

    #[must_use]
    pub const fn max_words(&self) -> usize {
        self.max_words
    }

    #[must_use]
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }

    #[must_use]
    pub const fn min_words_before_eos(&self) -> usize {
        self.min_words_before_eos
    }
}
