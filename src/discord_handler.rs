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
    markov::MarkovChain,
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
        let target_channel_id = {
            let state = self.state.lock().await;
            state.target_channel_id
        };

        if target_channel_id != Some(channel_id) {
            return Ok(());
        }
        if author_is_bot || author_id == current_user_id {
            return Ok(());
        }

        let tokens = self.tokenizer.tokenize(content);
        let cooldown = Duration::from_secs(self.config.reply_cooldown_secs);

        let mut should_persist = false;
        let mut reply_text: Option<String> = None;

        {
            let mut state = self.state.lock().await;

            if !tokens.is_empty() {
                state.chain.train_tokens(&tokens)?;
                should_persist = true;
            }

            let can_reply = state
                .last_reply_at
                .is_none_or(|last| last.elapsed() >= cooldown);

            if can_reply {
                let mut rng = rng();
                let generated = state
                    .chain
                    .generate_sentence(&mut rng, self.config.max_words);

                reply_text = Some(generated.unwrap_or_else(|| GENERATION_FALLBACK.to_owned()));
                state.last_reply_at = Some(Instant::now());
            }
        }

        if should_persist {
            let snapshot = {
                let state = self.state.lock().await;
                state.chain.clone()
            };
            save_chain(&self.config.data_path, &snapshot).await?;
        }

        if let Some(text) = reply_text {
            let _ = http.create_message(channel_id).content(&text).await?;
        }

        Ok(())
    }
}
