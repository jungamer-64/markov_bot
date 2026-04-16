use std::{env, error::Error, path::PathBuf};

use anyhow::{Error as AnyhowError, anyhow};
use markov_core::{MaxWords, MinWordsBeforeEos, NgramOrder, Temperature};
use markov_storage::StorageCompressionMode;

pub(crate) type DynError = AnyhowError;

#[derive(Clone, Debug)]
pub(crate) struct BotConfig {
    discord_token: String,
    data_path: PathBuf,
    ngram_order: NgramOrder,
    storage_min_edge_count: u64,
    storage_compression: StorageCompressionMode,
    max_words: MaxWords,
    temperature: Temperature,
    min_words_before_eos: MinWordsBeforeEos,
    reply_cooldown_secs: u64,
}

impl BotConfig {
    pub(crate) fn discord_token(&self) -> &str {
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

    pub(crate) const fn reply_cooldown_secs(&self) -> u64 {
        self.reply_cooldown_secs
    }

    pub(crate) fn from_env() -> Result<Self, DynError> {
        dotenvy::dotenv().ok();
        Self::from_env_with(|key| env::var(key))
    }

    fn from_env_with<F>(mut get_var: F) -> Result<Self, DynError>
    where
        F: FnMut(&str) -> Result<String, env::VarError>,
    {
        let discord_token = required_env_with(&mut get_var, "DISCORD_TOKEN")?;

        let data_path = get_var("MARKOV_DATA_PATH")
            .map_or_else(|_| PathBuf::from("data/markov_chain.mkv3"), PathBuf::from);

        let ngram_order = NgramOrder::new(env_parse_or_default_with(
            &mut get_var,
            "MARKOV_NGRAM_ORDER",
            NgramOrder::DEFAULT.as_usize().map_err(|e| AnyhowError::msg(e.to_string()))?,
        )?)
        .map_err(|error| AnyhowError::msg(error.to_string()))?;

        let storage_min_edge_count =
            env_parse_or_default_with(&mut get_var, "STORAGE_MIN_EDGE_COUNT", 1_u64)?;
        if storage_min_edge_count == 0 {
            return Err(anyhow!("STORAGE_MIN_EDGE_COUNT must be >= 1"));
        }

        let storage_compression = match get_var("STORAGE_COMPRESSION") {
            Ok(raw) => StorageCompressionMode::parse(raw.as_str())?,
            Err(env::VarError::NotPresent) => StorageCompressionMode::Auto,
            Err(error) => return Err(error.into()),
        };

        let max_words = MaxWords::new(env_parse_or_default_with(
            &mut get_var,
            "REPLY_MAX_WORDS",
            MaxWords::DEFAULT.get(),
        )?)
        .map_err(|error| AnyhowError::msg(error.to_string()))?;

        let temperature = Temperature::new(env_parse_or_default_with(
            &mut get_var,
            "REPLY_TEMPERATURE",
            Temperature::DEFAULT.get(),
        )?)
        .map_err(|error| AnyhowError::msg(error.to_string()))?;

        let min_words_before_eos = MinWordsBeforeEos::new(env_parse_or_default_with(
            &mut get_var,
            "REPLY_MIN_WORDS_BEFORE_EOS",
            MinWordsBeforeEos::DEFAULT.get(),
        )?);

        if min_words_before_eos.get() > max_words.get() {
            return Err(anyhow!(
                "REPLY_MIN_WORDS_BEFORE_EOS must be <= REPLY_MAX_WORDS"
            ));
        }

        let reply_cooldown_secs =
            env_parse_or_default_with(&mut get_var, "REPLY_COOLDOWN_SECS", 5_u64)?;

        Ok(Self {
            discord_token,
            data_path,
            ngram_order,
            storage_min_edge_count,
            storage_compression,
            max_words,
            temperature,
            min_words_before_eos,
            reply_cooldown_secs,
        })
    }
}

fn required_env_with<F>(get_var: &mut F, key: &str) -> Result<String, DynError>
where
    F: FnMut(&str) -> Result<String, env::VarError>,
{
    let value = get_var(key)?;
    if value.trim().is_empty() {
        return Err(anyhow!("{key} is empty"));
    }

    Ok(value)
}

fn env_parse_or_default_with<T, F>(get_var: &mut F, key: &str, default: T) -> Result<T, DynError>
where
    F: FnMut(&str) -> Result<String, env::VarError>,
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    match get_var(key) {
        Ok(raw) => Ok(raw.parse::<T>()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::BotConfig;

    fn ensure(condition: bool, message: &str) -> Result<(), super::DynError> {
        if condition {
            Ok(())
        } else {
            Err(anyhow::Error::msg(message.to_owned()))
        }
    }

    fn config_from_pairs(pairs: &[(&str, &str)]) -> Result<BotConfig, super::DynError> {
        let env = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect::<HashMap<_, _>>();
        BotConfig::from_env_with(|key| env.get(key).cloned().ok_or(std::env::VarError::NotPresent))
    }

    #[test]
    fn defaults_ngram_order_to_six() -> Result<(), super::DynError> {
        let config = config_from_pairs(&[("DISCORD_TOKEN", "token")])?;
        ensure(config.ngram_order().as_usize().map_err(|e| anyhow::Error::msg(e.to_string()))? == 6, "default ngram order should be 6")?;
        Ok(())
    }

    #[test]
    fn accepts_ngram_order_bounds() -> Result<(), super::DynError> {
        let lower = config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "1")])?;
        let upper = config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "6")])?;
        let seven = config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "7")])?;
        let sixteen =
            config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "16")])?;

        ensure(lower.ngram_order().as_usize().map_err(|e| anyhow::Error::msg(e.to_string()))? == 1, "ngram order 1 should be accepted")?;
        ensure(upper.ngram_order().as_usize().map_err(|e| anyhow::Error::msg(e.to_string()))? == 6, "ngram order 6 should be accepted")?;
        ensure(seven.ngram_order().as_usize().map_err(|e| anyhow::Error::msg(e.to_string()))? == 7, "ngram order 7 should be accepted")?;
        ensure(
            sixteen.ngram_order().as_usize().map_err(|e| anyhow::Error::msg(e.to_string()))? == 16,
            "ngram order 16 should be accepted",
        )?;
        Ok(())
    }

    #[test]
    fn rejects_zero_ngram_order() -> Result<(), super::DynError> {
        ensure(
            config_from_pairs(&[("DISCORD_TOKEN", "token"), ("MARKOV_NGRAM_ORDER", "0")]).is_err(),
            "ngram order 0 should be rejected",
        )?;
        Ok(())
    }
}
