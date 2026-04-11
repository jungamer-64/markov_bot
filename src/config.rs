use std::{env, error::Error, path::PathBuf};

use crate::storage::StorageCompressionMode;

pub(crate) type DynError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Debug)]
pub(super) struct BotConfig {
    pub(crate) discord_token: String,
    pub(crate) data_path: PathBuf,
    pub(crate) storage_min_edge_count: u64,
    pub(crate) storage_compression: StorageCompressionMode,
    pub(crate) max_words: usize,
    pub(crate) generation_temperature: f64,
    pub(crate) min_words_before_eos: usize,
    pub(crate) reply_cooldown_secs: u64,
}

impl BotConfig {
    pub(crate) fn from_env() -> Result<Self, DynError> {
        dotenvy::dotenv().ok();

        let discord_token = required_env("DISCORD_TOKEN")?;

        let data_path = env::var("MARKOV_DATA_PATH")
            .map_or_else(|_| PathBuf::from("data/markov_chain.mkv3"), PathBuf::from);

        let storage_min_edge_count = env_parse_or_default("STORAGE_MIN_EDGE_COUNT", 1_u64)?;
        if storage_min_edge_count == 0 {
            return Err("STORAGE_MIN_EDGE_COUNT must be >= 1".into());
        }

        let storage_compression = match env::var("STORAGE_COMPRESSION") {
            Ok(raw) => StorageCompressionMode::parse(raw.as_str())?,
            Err(env::VarError::NotPresent) => StorageCompressionMode::Auto,
            Err(error) => return Err(error.into()),
        };

        let max_words = env_parse_or_default("REPLY_MAX_WORDS", 20_usize)?;
        if max_words == 0 {
            return Err("REPLY_MAX_WORDS must be >= 1".into());
        }

        let generation_temperature = env_parse_or_default("REPLY_TEMPERATURE", 1.0_f64)?;
        if !generation_temperature.is_finite() || generation_temperature <= 0.0 {
            return Err("REPLY_TEMPERATURE must be a finite value > 0".into());
        }

        let min_words_before_eos = env_parse_or_default("REPLY_MIN_WORDS_BEFORE_EOS", 0_usize)?;
        if min_words_before_eos > max_words {
            return Err("REPLY_MIN_WORDS_BEFORE_EOS must be <= REPLY_MAX_WORDS".into());
        }

        let reply_cooldown_secs = env_parse_or_default("REPLY_COOLDOWN_SECS", 5_u64)?;

        Ok(Self {
            discord_token,
            data_path,
            storage_min_edge_count,
            storage_compression,
            max_words,
            generation_temperature,
            min_words_before_eos,
            reply_cooldown_secs,
        })
    }
}

fn required_env(key: &str) -> Result<String, DynError> {
    let value = env::var(key)?;
    if value.trim().is_empty() {
        return Err(format!("{key} is empty").into());
    }

    Ok(value)
}

fn env_parse_or_default<T>(key: &str, default: T) -> Result<T, DynError>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    match env::var(key) {
        Ok(raw) => Ok(raw.parse::<T>()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}
