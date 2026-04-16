use std::{
    path::Path,
    time::{Duration, Instant},
};

use markov_core::{GenerationOptions, MarkovChain, NgramOrder};
use markov_storage::{StorageCompressionMode, decode_chain, encode_chain};
use rand::rng;
use thiserror::Error;
use tokio::fs;
use tokio::sync::{mpsc, oneshot};
use twilight_http::Client as HttpClient;
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, UserMarker},
};

use crate::{
    config::BotConfig,
    tokenizer::{Tokenizer, TokenizerError},
};

#[derive(Debug, Error)]
pub(crate) enum HandlerError {
    #[error("Storage error: {0}")]
    Storage(#[from] markov_storage::StorageError),

    #[error("Core error: {0}")]
    Core(#[from] markov_core::MarkovError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Discord API error: {0}")]
    Discord(#[from] twilight_http::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Actor error: {0}")]
    Actor(String),

    #[error("Invalid generation options: {0}")]
    InvalidOptions(String),
}

const GENERATION_FALLBACK: &str = "まだ学習中です。もう少し話しかけてください。";

#[derive(Debug)]
struct RuntimeState {
    chain: MarkovChain,
    last_reply_at: Option<Instant>,
    target_channel_id: Option<Id<ChannelMarker>>,
}

#[derive(Debug)]
enum HandlerCommand {
    SetTargetChannel(Id<ChannelMarker>),
    HandleMessage {
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
        tokens: Vec<String>,
        reply_tx: oneshot::Sender<Result<Option<String>, HandlerError>>,
    },
}

#[derive(Clone)]
pub(crate) struct DiscordHandler {
    tokenizer: Tokenizer,
    tx: mpsc::Sender<HandlerCommand>,
}

impl DiscordHandler {
    pub(crate) async fn new(
        config: BotConfig,
        current_user_id: Id<UserMarker>,
    ) -> Result<Self, HandlerError> {
        let chain = load_chain(config.data_path(), config.ngram_order()).await?;
        let (tx, rx) = mpsc::channel(100);

        let state = RuntimeState {
            chain,
            last_reply_at: None,
            target_channel_id: None,
        };

        let actor = HandlerActor::new(config, current_user_id, state, rx);
        tokio::spawn(actor.run());

        let tokenizer = match Tokenizer::new() {
            Ok(t) => t,
            Err(error) => {
                eprintln!("Warning: Failed to initialize Lindera (Japanese tokenizer), falling back to unicode-segmentation: {error}");
                Tokenizer::with_fallback()
            }
        };

        Ok(Self {
            tokenizer,
            tx,
        })
    }

    pub(crate) async fn set_target_channel(&self, channel_id: Id<ChannelMarker>) {
        let _ = self.tx.send(HandlerCommand::SetTargetChannel(channel_id)).await;
    }

    pub(crate) async fn handle_message(
        &self,
        http: &HttpClient,
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
        content: &str,
    ) -> Result<(), HandlerError> {
        let tokens = self.tokenizer.tokenize(content);
        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx.send(HandlerCommand::HandleMessage {
            channel_id,
            author_id,
            author_is_bot,
            tokens,
            reply_tx,
        }).await.map_err(|_error| HandlerError::Actor("handler actor is dead".to_owned()))?;

        let reply_text: Option<String> = reply_rx.await.map_err(|_error| HandlerError::Actor("reply channel closed".to_owned()))??;

        if let Some(text) = reply_text {
            let _ = http.create_message(channel_id).content(&text).await?;
        }

        Ok(())
    }
}

struct HandlerActor {
    config: BotConfig,
    current_user_id: Id<UserMarker>,
    state: RuntimeState,
    rx: mpsc::Receiver<HandlerCommand>,
}

impl HandlerActor {
    const fn new(
        config: BotConfig,
        current_user_id: Id<UserMarker>,
        state: RuntimeState,
        rx: mpsc::Receiver<HandlerCommand>,
    ) -> Self {
        Self {
            config,
            current_user_id,
            state,
            rx,
        }
    }

    async fn run(mut self) {
        while let Some(command) = self.rx.recv().await {
            match command {
                HandlerCommand::SetTargetChannel(id) => {
                    self.state.target_channel_id = Some(id);
                }
                HandlerCommand::HandleMessage {
                    channel_id,
                    author_id,
                    author_is_bot,
                    tokens,
                    reply_tx,
                } => {
                    let res = self.handle_message(channel_id, author_id, author_is_bot, tokens).await;
                    let _ = reply_tx.send(res);
                }
            }
        }
    }

    async fn handle_message(
        &mut self,
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
        tokens: Vec<String>,
    ) -> Result<Option<String>, HandlerError> {
        if !self.should_process(channel_id, author_id, author_is_bot) {
            return Ok(None);
        }

        let should_persist = if tokens.is_empty() {
            false
        } else {
            self.state.chain.train_tokens(&tokens)?;
            true
        };

        if should_persist {
            save_chain(
                self.config.data_path(),
                &self.state.chain,
                self.config.storage_min_edge_count(),
                self.config.storage_compression(),
            )
            .await?;
        }

        let cooldown = Duration::from_secs(self.config.reply_cooldown_secs());
        if can_reply(self.state.last_reply_at, cooldown) {
            self.state.last_reply_at = Some(Instant::now());
            Ok(Some(self.build_reply_text()?))
        } else {
            Ok(None)
        }
    }

    fn should_process(
        &self,
        channel_id: Id<ChannelMarker>,
        author_id: Id<UserMarker>,
        author_is_bot: bool,
    ) -> bool {
        self.state.target_channel_id == Some(channel_id)
            && !should_ignore_author(author_is_bot, author_id, self.current_user_id)
    }

    fn build_reply_text(&self) -> Result<String, HandlerError> {
        let mut rng = rng();
        let options = GenerationOptions::new(
            self.config.max_words(),
            self.config.temperature(),
            self.config.min_words_before_eos(),
        ).map_err(|e| HandlerError::InvalidOptions(e.to_string()))?;

        Ok(self.state
            .chain
            .generate_sentence_with_options(&mut rng, options)
            .unwrap_or_else(|| GENERATION_FALLBACK.to_owned()))
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

async fn load_chain(path: &Path, expected_ngram_order: NgramOrder) -> Result<MarkovChain, HandlerError> {
    if !path.exists() {
        return MarkovChain::new(expected_ngram_order).map_err(Into::into);
    }

    let bytes = fs::read(path).await?;
    decode_chain(bytes.as_slice(), expected_ngram_order).map_err(Into::into)
}

async fn save_chain(
    path: &Path,
    chain: &MarkovChain,
    min_edge_count: u64,
    compression_mode: StorageCompressionMode,
) -> Result<(), HandlerError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let payload = encode_chain(chain, markov_core::Count::new(min_edge_count), compression_mode)?;
    fs::write(path, payload).await?;
    Ok(())
}
