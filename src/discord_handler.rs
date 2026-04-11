use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use rand::rng;
use tokio::sync::Mutex;
use twilight_http::Client as HttpClient;
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, UserMarker},
};

use crate::{
    config::{BotConfig, DynError},
    markov::{GenerationOptions, MarkovChain},
    storage::{load_chain, save_chain},
    tokenizer::Tokenizer,
};

const GENERATION_FALLBACK: &str = "まだ学習中です。もう少し話しかけてください。";

#[derive(Debug, Clone)]
struct RuntimeState {
    chain: MarkovChain,
    last_reply_at: Option<Instant>,
    target_channel_id: Option<Id<ChannelMarker>>,
}

#[derive(Debug)]
struct HandleOutcome {
    should_persist: bool,
    reply_text: Option<String>,
}

#[derive(Clone)]
pub(crate) struct DiscordHandler {
    config: BotConfig,
    current_user_id: Id<UserMarker>,
    tokenizer: Tokenizer,
    state: Arc<Mutex<RuntimeState>>,
}

impl DiscordHandler {
    pub(crate) async fn new(
        config: BotConfig,
        current_user_id: Id<UserMarker>,
    ) -> Result<Self, DynError> {
        let chain = load_chain(&config.data_path, config.ngram_order).await?;

        Ok(Self {
            config,
            current_user_id,
            tokenizer: Tokenizer::new(),
            state: Arc::new(Mutex::new(RuntimeState {
                chain,
                last_reply_at: None,
                target_channel_id: None,
            })),
        })
    }

    pub(crate) async fn set_target_channel(&self, channel_id: Id<ChannelMarker>) {
        let mut state = self.state.lock().await;
        state.target_channel_id = Some(channel_id);
    }

    pub(crate) async fn handle_message(
        &self,
        http: &HttpClient,
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
        content: &str,
    ) -> Result<(), DynError> {
        if self
            .should_skip_message(channel_id, author_id, author_is_bot)
            .await
        {
            return Ok(());
        }

        let tokens = self.tokenizer.tokenize(content);
        let cooldown = Duration::from_secs(self.config.reply_cooldown_secs);
        let outcome = self
            .update_state_for_message(tokens.as_slice(), cooldown)
            .await?;

        if outcome.should_persist {
            let snapshot = self.snapshot_chain().await;
            save_chain(
                &self.config.data_path,
                &snapshot,
                self.config.storage_min_edge_count,
                self.config.storage_compression,
            )
            .await?;
        }

        if let Some(text) = outcome.reply_text {
            let _ = http.create_message(channel_id).content(&text).await?;
        }

        Ok(())
    }

    async fn is_target_channel(&self, channel_id: Id<ChannelMarker>) -> bool {
        let state = self.state.lock().await;
        state.target_channel_id == Some(channel_id)
    }

    async fn should_skip_message(
        &self,
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
    ) -> bool {
        !self.is_target_channel(channel_id).await
            || should_ignore_author(author_is_bot, author_id, self.current_user_id)
    }

    async fn update_state_for_message(
        &self,
        tokens: &[String],
        cooldown: Duration,
    ) -> Result<HandleOutcome, DynError> {
        let mut state = self.state.lock().await;
        let should_persist = if tokens.is_empty() {
            false
        } else {
            state.chain.train_tokens(tokens)?;
            true
        };

        let reply_chain = if can_reply(state.last_reply_at, cooldown) {
            state.last_reply_at = Some(Instant::now());
            Some(state.chain.clone())
        } else {
            None
        };

        drop(state);

        let reply_text = reply_chain.map(|chain| self.build_reply_text(&chain));

        Ok(HandleOutcome {
            should_persist,
            reply_text,
        })
    }

    fn build_reply_text(&self, chain: &MarkovChain) -> String {
        let mut rng = rng();
        chain
            .generate_sentence_with_options(
                &mut rng,
                GenerationOptions::new(
                    self.config.max_words,
                    self.config.generation_temperature,
                    self.config.min_words_before_eos,
                ),
            )
            .unwrap_or_else(|| GENERATION_FALLBACK.to_owned())
    }

    async fn snapshot_chain(&self) -> MarkovChain {
        let state = self.state.lock().await;
        state.chain.clone()
    }
}

fn should_ignore_author(
    author_is_bot: bool,
    author_id: Id<UserMarker>,
    current_user_id: Id<UserMarker>,
) -> bool {
    author_is_bot || author_id == current_user_id
}

fn can_reply(last_reply_at: Option<Instant>, cooldown: Duration) -> bool {
    last_reply_at.is_none_or(|last| last.elapsed() >= cooldown)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use twilight_model::id::Id;

    use crate::{
        config::BotConfig,
        markov::MarkovChain,
        storage::{StorageCompressionMode, save_chain},
    };

    use super::DiscordHandler;

    fn run_async_test<F>(future: F) -> Result<(), crate::config::DynError>
    where
        F: std::future::Future<Output = Result<(), crate::config::DynError>>,
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(future)
    }

    #[test]
    fn new_loads_saved_chain_with_matching_ngram_order() -> Result<(), crate::config::DynError> {
        run_async_test(async {
            let temp_dir = tempdir()?;
            let data_path = temp_dir.path().join("chain.mkv3");
            let mut chain = MarkovChain::new(7)?;
            chain.train_tokens(&["a".to_owned(), "b".to_owned(), "c".to_owned()])?;
            save_chain(&data_path, &chain, 1, StorageCompressionMode::Auto).await?;

            let config = BotConfig {
                discord_token: "token".to_owned(),
                data_path,
                ngram_order: 7,
                storage_min_edge_count: 1,
                storage_compression: StorageCompressionMode::Auto,
                max_words: 20,
                generation_temperature: 1.0,
                min_words_before_eos: 0,
                reply_cooldown_secs: 5,
            };

            let result = DiscordHandler::new(config, Id::new(1)).await;
            crate::test_support::ensure(
                result.is_ok(),
                "DiscordHandler::new should load matching order=7 data",
            )?;
            Ok(())
        })
    }

    #[test]
    fn new_rejects_saved_ngram_order_mismatch() -> Result<(), crate::config::DynError> {
        run_async_test(async {
            let temp_dir = tempdir()?;
            let data_path = temp_dir.path().join("chain.mkv3");
            let mut chain = MarkovChain::new(6)?;
            chain.train_tokens(&["a".to_owned()])?;
            save_chain(&data_path, &chain, 1, StorageCompressionMode::Auto).await?;

            let config = BotConfig {
                discord_token: "token".to_owned(),
                data_path,
                ngram_order: 3,
                storage_min_edge_count: 1,
                storage_compression: StorageCompressionMode::Auto,
                max_words: 20,
                generation_temperature: 1.0,
                min_words_before_eos: 0,
                reply_cooldown_secs: 5,
            };

            let result = DiscordHandler::new(config, Id::new(1)).await;
            crate::test_support::ensure(
                result.is_err(),
                "DiscordHandler::new should fail when saved ngram order differs from config",
            )?;
            Ok(())
        })
    }
}
