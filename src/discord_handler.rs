use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use markov_core::{Count, GenerationOptions, MarkovChain};
use markov_storage::{FileStore, SaveError};
use rand::rng;
use thiserror::Error;
use tokio::{sync::oneshot, task::JoinHandle};
use twilight_http::Client as HttpClient;
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, UserMarker},
};

use crate::{
    config::{BotConfig, StorageFailurePolicy},
    tokenizer::Tokenizer,
};

#[derive(Debug, Error)]
pub(crate) enum HandlerError {
    #[error("storage error: {0}")]
    Storage(#[from] markov_storage::StorageError),
    #[error("save error: {0}")]
    Save(#[from] SaveError),
    #[error("core error: {0}")]
    Core(#[from] markov_core::MarkovError),
    #[error("Discord API error: {0}")]
    Discord(#[from] twilight_http::Error),
    #[error("model worker is unavailable")]
    WorkerUnavailable,
    #[error("model command queue is full")]
    QueueFull,
    #[error("model worker did not return a reply: {0}")]
    Reply(#[from] oneshot::error::RecvError),
    #[error("model worker failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

const GENERATION_FALLBACK: &str = "まだ学習中です。もう少し話しかけてください。";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuthorRole {
    User,
    Bot,
}

#[derive(Debug)]
enum Command {
    SetTargetChannel {
        id: Id<ChannelMarker>,
        reply: oneshot::Sender<()>,
    },
    Message {
        channel: Id<ChannelMarker>,
        author: Id<UserMarker>,
        role: AuthorRole,
        tokens: Vec<String>,
        reply: oneshot::Sender<Option<String>>,
    },
}

/// Client endpoint only. The composition root owns and joins the worker task.
/// Dropping this endpoint closes admission; the worker drains accepted commands
/// and performs one final save of any dirty model before returning its outcome.
pub(crate) struct DiscordHandler {
    tokenizer: Tokenizer,
    tx: mpsc::SyncSender<Command>,
}

impl DiscordHandler {
    pub(crate) async fn new(
        config: BotConfig,
        current_user: Id<UserMarker>,
    ) -> Result<(Self, JoinHandle<Result<(), HandlerError>>), HandlerError> {
        let worker =
            tokio::task::spawn_blocking(move || Worker::open(&config, current_user)).await??;
        let tokenizer = match Tokenizer::new() {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                eprintln!("Tokenizer initialization failed; using Unicode segmentation: {error}");
                Tokenizer::with_fallback()
            }
        };
        let (tx, rx) = mpsc::sync_channel(100);
        let task = tokio::task::spawn_blocking(move || worker.run(&rx));
        Ok((Self { tokenizer, tx }, task))
    }

    fn send(&self, command: Command) -> Result<(), HandlerError> {
        self.tx.try_send(command).map_err(|error| match error {
            mpsc::TrySendError::Full(_) => HandlerError::QueueFull,
            mpsc::TrySendError::Disconnected(_) => HandlerError::WorkerUnavailable,
        })
    }

    pub(crate) async fn set_target_channel(
        &self,
        id: Id<ChannelMarker>,
    ) -> Result<(), HandlerError> {
        let (reply, response) = oneshot::channel();
        self.send(Command::SetTargetChannel { id, reply })?;
        response.await?;
        Ok(())
    }

    pub(crate) async fn handle_message(
        &self,
        http: &HttpClient,
        channel: Id<ChannelMarker>,
        author: Id<UserMarker>,
        role: AuthorRole,
        content: &str,
    ) -> Result<(), HandlerError> {
        let tokens = self.tokenizer.tokenize(content);
        let (reply, response) = oneshot::channel();
        self.send(Command::Message {
            channel,
            author,
            role,
            tokens,
            reply,
        })?;
        if let Some(text) = response.await? {
            http.create_message(channel).content(&text).await?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Persistence {
    Clean,
    /// The authoritative in-memory model is newer than the last acknowledged save.
    Dirty {
        retry_at: Instant,
    },
}

#[derive(Debug)]
struct SavePolicy {
    min_count: Count,
    compression: markov_storage::StorageCompressionMode,
    failure: StorageFailurePolicy,
    retry_interval: Duration,
}

struct Worker {
    save_policy: SavePolicy,
    generation: GenerationOptions,
    cooldown: Duration,
    current_user: Id<UserMarker>,
    chain: MarkovChain,
    store: FileStore,
    persistence: Persistence,
    channel: Option<Id<ChannelMarker>>,
    last_reply: Option<Instant>,
}

impl Worker {
    fn open(config: &BotConfig, current_user: Id<UserMarker>) -> Result<Self, HandlerError> {
        let store = FileStore::open(config.data_path(), config.storage_limits())?;
        let chain = store.load(config.ngram_order())?;
        Ok(Self {
            save_policy: SavePolicy {
                min_count: Count::new(config.storage_min_edge_count()),
                compression: config.storage_compression(),
                failure: config.storage_failure_policy(),
                retry_interval: config.storage_retry_interval(),
            },
            generation: GenerationOptions::new(
                config.max_words(),
                config.temperature(),
                config.min_words_before_eos(),
            )?,
            cooldown: config.reply_cooldown().get(),
            current_user,
            chain,
            store,
            persistence: Persistence::Clean,
            channel: None,
            last_reply: None,
        })
    }

    fn run(mut self, rx: &mpsc::Receiver<Command>) -> Result<(), HandlerError> {
        loop {
            // Check the deadline before receiving, so queued traffic cannot starve retry.
            if let Persistence::Dirty { retry_at } = self.persistence
                && Instant::now() >= retry_at
            {
                self.persist()?;
            }
            let command = match self.persistence {
                Persistence::Clean => rx
                    .recv()
                    .map_err(|_disconnected| mpsc::RecvTimeoutError::Disconnected),
                Persistence::Dirty { retry_at } => {
                    rx.recv_timeout(retry_at.saturating_duration_since(Instant::now()))
                }
            };
            match command {
                Ok(command) => self.process(command)?,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return self.finish(),
            }
        }
    }

    fn process(&mut self, command: Command) -> Result<(), HandlerError> {
        match command {
            Command::SetTargetChannel { id, reply } => {
                self.channel = Some(id);
                // The requester may cancel. Accepted state changes remain effective.
                let _ = reply.send(());
            }
            Command::Message {
                channel,
                author,
                role,
                tokens,
                reply,
            } => {
                let response = if self.channel == Some(channel)
                    && role == AuthorRole::User
                    && author != self.current_user
                {
                    self.learn_and_reply(&tokens)?
                } else {
                    None
                };
                // Cancellation does not roll back learning or an acknowledged save.
                let _ = reply.send(response);
            }
        }
        Ok(())
    }

    fn learn_and_reply(&mut self, tokens: &[String]) -> Result<Option<String>, HandlerError> {
        if !tokens.is_empty() {
            self.chain.train_tokens(tokens)?;
            if matches!(self.persistence, Persistence::Clean) {
                self.persistence = Persistence::Dirty {
                    retry_at: Instant::now(),
                };
                self.persist()?;
            }
        }
        if self
            .last_reply
            .is_some_and(|last| last.elapsed() < self.cooldown)
        {
            return Ok(None);
        }
        let text = self
            .chain
            .generate_sentence_with_options(&mut rng(), self.generation)
            .unwrap_or_else(|| GENERATION_FALLBACK.to_owned());
        self.last_reply = Some(Instant::now());
        Ok(Some(text))
    }

    fn persist(&mut self) -> Result<(), HandlerError> {
        match self.store.save(
            &self.chain,
            self.save_policy.min_count,
            self.save_policy.compression,
        ) {
            Ok(()) => {
                self.persistence = Persistence::Clean;
                Ok(())
            }
            Err(error)
                if self.save_policy.failure == StorageFailurePolicy::Retry
                    && error.is_retryable() =>
            {
                eprintln!(
                    "Model remains unsaved; retrying after {:?}: {error}",
                    self.save_policy.retry_interval
                );
                self.persistence = Persistence::Dirty {
                    retry_at: Instant::now() + self.save_policy.retry_interval,
                };
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    fn finish(&mut self) -> Result<(), HandlerError> {
        if matches!(self.persistence, Persistence::Dirty { .. }) {
            self.store.save(
                &self.chain,
                self.save_policy.min_count,
                self.save_policy.compression,
            )?;
            self.persistence = Persistence::Clean;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use markov_core::{MaxWords, MinWordsBeforeEos, NgramOrder, Temperature};
    use markov_storage::{StorageCompressionMode, StorageLimits};
    use std::{fs, path::Path, thread};

    fn ensure(condition: bool, message: &str) -> std::io::Result<()> {
        if condition {
            Ok(())
        } else {
            Err(std::io::Error::other(message.to_owned()))
        }
    }

    fn ensure_eq<L: PartialEq<R> + std::fmt::Debug + ?Sized, R: std::fmt::Debug + ?Sized>(
        left: &L,
        right: &R,
        message: &str,
    ) -> std::io::Result<()> {
        if left == right {
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "{message}: left={left:?}, right={right:?}"
            )))
        }
    }

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn worker(
        path: &Path,
        failure: StorageFailurePolicy,
        limits: StorageLimits,
    ) -> TestResult<Worker> {
        let store = FileStore::open(path, limits)?;
        let chain = store.load(NgramOrder::new(1)?)?;
        Ok(Worker {
            save_policy: SavePolicy {
                min_count: Count::new(1),
                compression: StorageCompressionMode::Uncompressed,
                failure,
                retry_interval: Duration::from_millis(20),
            },
            generation: GenerationOptions::new(
                MaxWords::new(3)?,
                Temperature::new(1.0)?,
                MinWordsBeforeEos::new(0),
            )?,
            cooldown: Duration::ZERO,
            current_user: Id::new(1),
            chain,
            store,
            persistence: Persistence::Clean,
            channel: Some(Id::new(2)),
            last_reply: None,
        })
    }

    #[test]
    fn exit_policy_propagates_save_failure_and_retry_does_not_repeat_learning() -> TestResult {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model");
        let mut owner = worker(&path, StorageFailurePolicy::Exit, StorageLimits::default())?;
        fs::create_dir(&path)?; // Replacement cannot overwrite a directory on any target OS.
        ensure(
            matches!(
                owner.learn_and_reply(&["first".into()]),
                Err(HandlerError::Save(_))
            ),
            "assert contract failed",
        )?;
        drop(owner);
        fs::remove_dir(&path)?;
        let mut owner = worker(&path, StorageFailurePolicy::Retry, StorageLimits::default())?;
        fs::create_dir(&path)?;
        owner.learn_and_reply(&["first".into()])?;
        owner.learn_and_reply(&["second".into()])?;
        ensure(
            matches!(owner.persistence, Persistence::Dirty { .. }),
            "assert contract failed",
        )?;
        fs::remove_dir(&path)?;
        owner.persist()?;
        let restored = owner.store.load(NgramOrder::new(1)?)?;
        ensure_eq(
            &(restored
                .starts()
                .values()
                .map(|count| count.get())
                .sum::<u64>()),
            &(2),
            "assert_eq contract failed",
        )?;
        ensure(
            matches!(owner.persistence, Persistence::Clean),
            "assert contract failed",
        )?;
        Ok(())
    }

    #[test]
    fn retry_policy_does_not_hide_preparation_errors() -> TestResult {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model");
        let mut owner = worker(
            &path,
            StorageFailurePolicy::Retry,
            StorageLimits::new(1, 100)?,
        )?;
        ensure(
            matches!(
                owner.learn_and_reply(&["first".into()]),
                Err(HandlerError::Save(SaveError::Preparation(_)))
            ),
            "assert contract failed",
        )?;
        Ok(())
    }

    #[test]
    fn idle_retry_and_shutdown_drain_are_owned() -> TestResult {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model");
        let mut owner = worker(&path, StorageFailurePolicy::Retry, StorageLimits::default())?;
        fs::create_dir(&path)?;
        owner.learn_and_reply(&["first".into()])?;
        fs::remove_dir(&path)?;
        let (tx, rx) = mpsc::sync_channel(10);
        let task = thread::spawn(move || owner.run(&rx));
        let deadline = Instant::now() + Duration::from_secs(5);
        let observed = loop {
            if path.is_file() {
                break true;
            }
            if Instant::now() >= deadline {
                break false;
            }
            thread::sleep(Duration::from_millis(5));
        };
        // Queue learning and close admission without waiting for its response.
        let (reply, _response) = oneshot::channel();
        let sent = tx.send(Command::Message {
            channel: Id::new(2),
            author: Id::new(3),
            role: AuthorRole::User,
            tokens: vec!["second".into()],
            reply,
        });
        drop(tx);
        task.join()
            .map_err(|_panic| std::io::Error::other("worker panicked"))??;
        sent?;
        ensure(
            observed,
            "dirty model must be retried without incoming messages",
        )?;
        let store = FileStore::open(&path, StorageLimits::default())?;
        ensure_eq(
            &(store
                .load(NgramOrder::new(1)?)?
                .starts()
                .values()
                .map(|count| count.get())
                .sum::<u64>()),
            &(2),
            "assert_eq contract failed",
        )?;
        Ok(())
    }

    #[test]
    fn shutdown_saves_dirty_model_once_and_propagates_failure() -> TestResult {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model");
        let mut owner = worker(&path, StorageFailurePolicy::Retry, StorageLimits::default())?;
        fs::create_dir(&path)?;
        owner.learn_and_reply(&["first".into()])?;
        ensure(
            owner.finish().is_err(),
            "retry policy cannot hide final save failure",
        )?;
        fs::remove_dir(&path)?;
        owner.finish()?;
        ensure(
            matches!(owner.persistence, Persistence::Clean),
            "assert contract failed",
        )?;
        ensure_eq(
            &(owner
                .store
                .load(NgramOrder::new(1)?)?
                .starts()
                .values()
                .map(|count| count.get())
                .sum::<u64>()),
            &(1),
            "assert_eq contract failed",
        )?;
        Ok(())
    }
}
