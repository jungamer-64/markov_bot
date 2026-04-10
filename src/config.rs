use std::{env, error::Error, path::PathBuf};

pub type DynError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Debug)]
pub struct BotConfig {
    pub discord_token: String,
    pub data_path: PathBuf,
    pub max_words: usize,
    pub reply_cooldown_secs: u64,
}

impl BotConfig {
    pub fn from_env() -> Result<Self, DynError> {
        dotenvy::dotenv().ok();

        let discord_token = required_env("DISCORD_TOKEN")?;

        let data_path = env::var("MARKOV_DATA_PATH")
            .map_or_else(|_| PathBuf::from("data/markov_chain.mkv3"), PathBuf::from);

        let max_words = env_parse_or_default("REPLY_MAX_WORDS", 20_usize)?;
        if max_words == 0 {
            return Err("REPLY_MAX_WORDS must be >= 1".into());
        }

        let reply_cooldown_secs = env_parse_or_default("REPLY_COOLDOWN_SECS", 5_u64)?;

        Ok(Self {
            discord_token,
            data_path,
            max_words,
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
