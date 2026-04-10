use std::{io, path::Path};

use tokio::fs;

use crate::{config::DynError, markov::MarkovChain};

pub async fn load_chain(path: &Path) -> Result<MarkovChain, DynError> {
    match fs::read_to_string(path).await {
        Ok(content) => serde_json::from_str::<MarkovChain>(&content)
            .map_or_else(|_| Ok(MarkovChain::default()), Ok),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(MarkovChain::default()),
        Err(error) => Err(error.into()),
    }
}

pub async fn save_chain(path: &Path, chain: &MarkovChain) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let payload = serde_json::to_string_pretty(chain)?;
    fs::write(path, payload).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokio::fs;

    use crate::markov::MarkovChain;

    use super::{load_chain, save_chain};

    #[tokio::test]
    async fn load_returns_default_for_missing_file() {
        let file_path = temp_file_path("missing");
        let chain = load_chain(&file_path).await.expect("load should succeed");

        assert!(chain.starts.is_empty());
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        let file_path = temp_file_path("roundtrip");
        let mut chain = MarkovChain::default();
        chain.train_tokens(&[
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ]);

        save_chain(&file_path, &chain)
            .await
            .expect("save should succeed");
        let loaded = load_chain(&file_path).await.expect("load should succeed");

        assert!(!loaded.starts.is_empty());

        let _ = fs::remove_file(file_path).await;
    }

    fn temp_file_path(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();

        std::env::temp_dir().join(format!("markov_bot_{prefix}_{nanos}.json"))
    }
}
