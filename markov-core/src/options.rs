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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Temperature(f64);

impl Temperature {
    pub const DEFAULT: Self = Self(1.0);

    /// # Errors
    /// Returns `MarkovError::InvalidTemperature` if the value is not > 0 or not finite.
    pub fn new(value: f64) -> Result<Self, MarkovError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(MarkovError::InvalidTemperature(value));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxWords(usize);

impl MaxWords {
    pub const DEFAULT: Self = Self(20);

    /// # Errors
    /// Returns `MarkovError::InvalidMaxWords` if the value is 0.
    pub fn new(value: usize) -> Result<Self, MarkovError> {
        if value == 0 {
            return Err(MarkovError::InvalidMaxWords(value));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MinWordsBeforeEos(usize);

impl MinWordsBeforeEos {
    pub const DEFAULT: Self = Self(0);

    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    max_words: MaxWords,
    temperature: Temperature,
    min_words_before_eos: MinWordsBeforeEos,
}

impl GenerationOptions {
    /// # Errors
    /// Returns `MarkovError::InvalidEosThreshold` if `min_words_before_eos` > `max_words`.
    pub fn new(
        max_words: MaxWords,
        temperature: Temperature,
        min_words_before_eos: MinWordsBeforeEos,
    ) -> Result<Self, MarkovError> {
        if min_words_before_eos.get() > max_words.get() {
            return Err(MarkovError::InvalidEosThreshold {
                min: min_words_before_eos.get(),
                max: max_words.get(),
            });
        }

        Ok(Self {
            max_words,
            temperature,
            min_words_before_eos,
        })
    }

    #[must_use]
    pub const fn max_words(&self) -> MaxWords {
        self.max_words
    }

    #[must_use]
    pub const fn temperature(&self) -> Temperature {
        self.temperature
    }

    #[must_use]
    pub const fn min_words_before_eos(&self) -> MinWordsBeforeEos {
        self.min_words_before_eos
    }
}
