use std::{env, path::PathBuf, time::Duration};

use markov_core::{MaxWords, MinWordsBeforeEos, NgramOrder, Temperature};
use markov_storage::StorageCompressionMode;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum ConfigError {
    #[error("Environment variable {0} is missing")]
    MissingEnvVar(String),

    #[error("Environment variable {0} is empty")]
    EmptyEnvVar(String),

    #[error("Failed to parse environment variable {0}: {1}")]
    ParseError(String, String),

    #[error("REPLY_MIN_WORDS_BEFORE_EOS ({min}) must be <= REPLY_MAX_WORDS ({max})")]
    InvalidEosThreshold { min: usize, max: usize },

    #[error("STORAGE_MIN_EDGE_COUNT must be >= 1")]
    InvalidMinEdgeCount,

    #[error("Markov core error: {0}")]
    Core(#[from] markov_core::MarkovError),

    #[error("Storage error: {0}")]
    Storage(#[from] markov_storage::StorageError),
}

#[derive(Clone, Debug)]
pub(crate) struct DiscordToken(String);

impl DiscordToken {
    #[must_use]
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReplyCooldown(Duration);

impl ReplyCooldown {
    #[must_use]
    pub(crate) const fn from_secs(secs: u64) -> Self {
        Self(Duration::from_secs(secs))
    }

    #[must_use]
    pub(crate) const fn get(self) -> Duration {
        self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BotConfig {
    discord_token: DiscordToken,
    data_path: PathBuf,
    ngram_order: NgramOrder,
    storage_min_edge_count: u64,
    storage_compression: StorageCompressionMode,
    max_words: MaxWords,
    temperature: Temperature,
    min_words_before_eos: MinWordsBeforeEos,
    reply_cooldown: ReplyCooldown,
}

impl BotConfig {
    pub(crate) const fn discord_token(&self) -> &DiscordToken {
        &self.discord_token
    }

    pub(crate) const fn data_path(&self) -> &PathBuf {
        &self.data_path
    }

    pub(crate) const fn ngram_order(&self) -> NgramOrder {
        self.ngram_order
    }

    pub(crate) const fn storage_min_edge_count(&self) -> u64 {
        self.storage_min_edge_count
    }

    pub(crate) const fn storage_compression(&self) -> StorageCompressionMode {
        self.storage_compression
    }

    pub(crate) const fn max_words(&self) -> MaxWords {
        self.max_words
    }

    pub(crate) const fn temperature(&self) -> Temperature {
        self.temperature
    }

    pub(crate) const fn min_words_before_eos(&self) -> MinWordsBeforeEos {
        self.min_words_before_eos
    }

    pub(crate) const fn reply_cooldown(&self) -> ReplyCooldown {
        self.reply_cooldown
    }

    pub(crate) fn from_env() -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok();
        Self::from_env_with(|key| env::var(key))
    }

    fn from_env_with<F>(mut get_var: F) -> Result<Self, ConfigError>
    where
        F: FnMut(&str) -> Result<String, env::VarError>,
    {
        let discord_token = DiscordToken(required_env_with(&mut get_var, "DISCORD_TOKEN")?);

        let data_path = get_var("MARKOV_DATA_PATH")
            .map_or_else(|_| PathBuf::from("data/markov_chain.mkv3"), PathBuf::from);

        let ngram_order_val = env_parse_or_default_with(
            &mut get_var,
            "MARKOV_NGRAM_ORDER",
            usize::try_from(NgramOrder::DEFAULT.get()).map_err(|_err| {
                ConfigError::Core(markov_core::MarkovError::Boundary(
                    "NgramOrder::DEFAULT exceeds usize range".into(),
                ))
            })?,
        )?;
        let ngram_order = NgramOrder::new(ngram_order_val)?;

        let storage_min_edge_count =
            env_parse_or_default_with(&mut get_var, "STORAGE_MIN_EDGE_COUNT", 1_u64)?;
        if storage_min_edge_count == 0 {
            return Err(ConfigError::InvalidMinEdgeCount);
        }

        let storage_compression = match get_var("STORAGE_COMPRESSION") {
            Ok(raw) => StorageCompressionMode::parse(raw.as_str())?,
            Err(env::VarError::NotPresent) => StorageCompressionMode::Auto,
            Err(error) => return Err(ConfigError::ParseError("STORAGE_COMPRESSION".to_owned(), error.to_string())),
        };

        let max_words_val = env_parse_or_default_with(
            &mut get_var,
            "REPLY_MAX_WORDS",
            MaxWords::DEFAULT.get(),
        )?;
        let max_words = MaxWords::new(max_words_val)?;

        let temperature_val = env_parse_or_default_with(
            &mut get_var,
            "REPLY_TEMPERATURE",
            Temperature::DEFAULT.get(),
        )?;
        let temperature = Temperature::new(temperature_val)?;

        let min_words_before_eos_val = env_parse_or_default_with(
            &mut get_var,
            "REPLY_MIN_WORDS_BEFORE_EOS",
            MinWordsBeforeEos::DEFAULT.get(),
        )?;
        let min_words_before_eos = MinWordsBeforeEos::new(min_words_before_eos_val);

        if min_words_before_eos.get() > max_words.get() {
            return Err(ConfigError::InvalidEosThreshold {
                min: min_words_before_eos.get(),
                max: max_words.get(),
            });
        }

        let reply_cooldown_secs =
            env_parse_or_default_with(&mut get_var, "REPLY_COOLDOWN_SECS", 5_u64)?;
        let reply_cooldown = ReplyCooldown::from_secs(reply_cooldown_secs);

        Ok(Self {
            discord_token,
            data_path,
            ngram_order,
            storage_min_edge_count,
            storage_compression,
            max_words,
            temperature,
            min_words_before_eos,
            reply_cooldown,
        })
    }
}

fn required_env_with<F>(get_var: &mut F, key: &str) -> Result<String, ConfigError>
where
    F: FnMut(&str) -> Result<String, env::VarError>,
{
    match get_var(key) {
        Ok(value) => {
            if value.trim().is_empty() {
                return Err(ConfigError::EmptyEnvVar(key.to_owned()));
            }
            Ok(value)
        }
        Err(env::VarError::NotPresent) => Err(ConfigError::MissingEnvVar(key.to_owned())),
        Err(error) => Err(ConfigError::ParseError(key.to_owned(), error.to_string())),
    }
}

fn env_parse_or_default_with<T, F>(get_var: &mut F, key: &str, default: T) -> Result<T, ConfigError>
where
    F: FnMut(&str) -> Result<String, env::VarError>,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match get_var(key) {
        Ok(raw) => raw.parse::<T>().map_err(|e| ConfigError::ParseError(key.to_owned(), e.to_string())),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(ConfigError::ParseError(key.to_owned(), error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::BotConfig;

    fn config_from_pairs(pairs: &[(&str, &str)]) -> Result<BotConfig, super::ConfigError> {
        let env = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect::<HashMap<_, _>>();
        BotConfig::from_env_with(|key| env.get(key).cloned().ok_or(std::env::VarError::NotPresent))
    }

    fn ensure(condition: bool, message: &str) -> Result<(), super::ConfigError> {
        if condition {
            Ok(())
        } else {
            Err(super::ConfigError::ParseError("test".to_owned(), message.to_owned()))
        }
    }

    fn ensure_eq<L, R>(left: &L, right: &R, message: &str) -> Result<(), super::ConfigError>
    where
        L: PartialEq<R> + std::fmt::Debug,
        R: std::fmt::Debug,
    {
        if left == right {
            Ok(())
        } else {
            Err(super::ConfigError::ParseError(
                "test".to_owned(),
                format!("{message}: expected {right:?}, got {left:?}"),
            ))
        }
    }

    #[test]
    fn defaults_ngram_order_to_six() -> Result<(), super::ConfigError> {
        let config = config_from_pairs(&[("DISCORD_TOKEN", "token")])?;
        ensure_eq(
            &config.ngram_order().get(),
            &6,
            "default ngram order should be 6",
        )
    }

    #[test]
    fn accepts_ngram_order_bounds() -> Result<(), super::ConfigError> {
        let lower = config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "1")])?;
        let upper = config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "6")])?;
        let seven = config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "7")])?;
        let sixteen =
            config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "16")])?;

        ensure_eq(
            &lower.ngram_order().get(),
            &1,
            "ngram order 1 should be accepted",
        )?;
        ensure_eq(
            &upper.ngram_order().get(),
            &6,
            "ngram order 6 should be accepted",
        )?;
        ensure_eq(
            &seven.ngram_order().get(),
            &7,
            "ngram order 7 should be accepted",
        )?;
        ensure_eq(
            &sixteen
                .ngram_order()
                .get(),
            &16,
            "ngram order 16 should be accepted",
        )?;
        Ok(())
    }

    #[test]
    fn rejects_zero_ngram_order() -> Result<(), super::ConfigError> {
        ensure(
            config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "0")]).is_err(),
            "ngram order 0 should be rejected",
        )
    }
}
