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
pub struct DiscordHandler {
    config: BotConfig,
    tokenizer: Tokenizer,
    state: Arc<Mutex<RuntimeState>>,
}

impl DiscordHandler {
    pub async fn new(config: BotConfig) -> Result<Self, DynError> {
        let chain = load_chain(&config.data_path).await?;

        Ok(Self {
            config,
            tokenizer: Tokenizer::new(),
            state: Arc::new(Mutex::new(RuntimeState {
                chain,
                last_reply_at: None,
                target_channel_id: None,
            })),
        })
    }

    pub async fn set_target_channel(&self, channel_id: Id<ChannelMarker>) {
        let mut state = self.state.lock().await;
        state.target_channel_id = Some(channel_id);
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn handle_message(
        &self,
        http: &HttpClient,
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
        content: &str,
        current_user_id: Id<UserMarker>,
    ) -> Result<(), DynError> {
        if self
            .should_skip_message(channel_id, author_id, author_is_bot, current_user_id)
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
        current_user_id: Id<UserMarker>,
    ) -> bool {
        !self.is_target_channel(channel_id).await
            || should_ignore_author(author_is_bot, author_id, current_user_id)
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
