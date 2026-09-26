use std::collections::{BTreeSet, HashMap};

use markov_core::{BOS_TOKEN, Count, EOS_TOKEN, MarkovChain, NgramOrder};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod file;
mod limits;
mod read;
pub use file::{FileStore, Publication, SaveError, SaveStage, ensure_distinct_paths, read_file};
pub use limits::{LimitKind, StorageLimits};
mod types;
mod write;

#[cfg(test)]
mod tests;

const MAGIC: [u8; 8] = *b"MKV3BIN\0";
const VERSION: u32 = 8;
const FLAG_VOCAB_BLOB_RLE: u32 = 1 << 0;
const FLAG_VOCAB_BLOB_ZSTD: u32 = 1 << 1;
const SUPPORTED_FLAGS: u32 = FLAG_VOCAB_BLOB_RLE | FLAG_VOCAB_BLOB_ZSTD;
const TOKENIZER_VERSION: u32 = 1;
const NORMALIZATION_FLAGS: u32 = 0;
const CHECKSUM_PLACEHOLDER: u64 = 0;

const HEADER_SIZE: usize = 52;
const DESCRIPTOR_SIZE: usize = 24;
const CHECKSUM_SIZE: usize = std::mem::size_of::<u64>();
const CHECKSUM_OFFSET: usize = HEADER_SIZE - CHECKSUM_SIZE;
const SECTION_METADATA_COUNT: u64 = 3;
const START_SECTION_HEADER_SIZE: u64 = 4;
const MODEL_SECTION_HEADER_SIZE: u64 = 8;
const EDGE_RECORD_SIZE: u64 = 12;

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;
const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("storage writer is already active: {0}")]
    Busy(std::path::PathBuf),
    #[error("input and output refer to the same file")]
    SameFile,
    #[error("{kind:?} limit exceeded: {actual} > {limit}")]
    Limit {
        kind: LimitKind,
        actual: u64,
        limit: u64,
    },
    #[error("allocation failed: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    #[error("storage format error: {0}")]
    Format(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("markov core error: {0}")]
    Core(#[from] markov_core::MarkovError),
    #[error("checksum mismatch: expected {expected:016x}, got {actual:016x}")]
    Checksum { expected: u64, actual: u64 },
    #[error("magic mismatch: expected {expected:?}, got {actual:?}")]
    Magic { expected: [u8; 8], actual: [u8; 8] },
    #[error("unsupported version: {0}")]
    Version(u32),
    #[error("ngram order mismatch: expected {expected:?}, got {actual:?}")]
    NgramOrderMismatch {
        expected: NgramOrder,
        actual: NgramOrder,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageCompressionMode {
    Auto,
    Uncompressed,
    Rle,
    Zstd,
}

impl StorageCompressionMode {
    /// Parses a storage compression mode from a string.
    ///
    /// # Errors
    /// Returns `StorageError::Format` if the input string is not a supported compression mode.
    pub fn parse(raw: &str) -> Result<Self, StorageError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "none" | "off" | "uncompressed" => Ok(Self::Uncompressed),
            "rle" | "vocab_rle" | "vocab-blob-rle" => Ok(Self::Rle),
            "zstd" => Ok(Self::Zstd),
            _ => Err(StorageError::Format(format!(
                "unsupported STORAGE_COMPRESSION value: {raw} (expected: auto|none|rle|zstd)"
            ))),
        }
    }

    #[must_use]
    pub const fn as_env_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Uncompressed => "none",
            Self::Rle => "rle",
            Self::Zstd => "zstd",
        }
    }
}

/// The compression actually stored in a file; automatic selection is not a file property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageCompression {
    Uncompressed,
    Rle,
    Zstd,
}

impl From<StorageCompression> for StorageCompressionMode {
    fn from(value: StorageCompression) -> Self {
        match value {
            StorageCompression::Uncompressed => Self::Uncompressed,
            StorageCompression::Rle => Self::Rle,
            StorageCompression::Zstd => Self::Zstd,
        }
    }
}

/// Bounded v8 encoding and decoding. Filesystem publication is owned by `FileStore`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Codec {
    limits: StorageLimits,
}

impl Codec {
    #[must_use]
    pub const fn new(limits: StorageLimits) -> Self {
        Self { limits }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageSnapshot {
    pub schema_version: u32,
    pub source: SnapshotSource,
    pub tokens: Vec<String>,
    pub starts: Vec<SnapshotEntry>,
    pub models: Vec<SnapshotModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotSource {
    pub storage_version: u32,
    pub ngram_order: usize,
    pub compression: StorageCompression,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEntry {
    pub prefix: Vec<u32>,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotModel {
    pub order: usize,
    pub entries: Vec<SnapshotModelEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotModelEntry {
    pub prefix: Vec<u32>,
    pub edges: Vec<SnapshotEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEdge {
    pub next: u32,
    pub count: u64,
}

impl StorageSnapshot {
    #[must_use]
    pub const fn ngram_order(&self) -> usize {
        self.source.ngram_order
    }
}

impl Codec {
    /// Decodes a Markov chain from a byte slice.
    ///
    /// # Errors
    /// Returns `StorageError` if decoding fails.
    pub fn decode_chain(
        &self,
        bytes: &[u8],
        expected_ngram_order: NgramOrder,
    ) -> Result<MarkovChain, StorageError> {
        read::decode_chain(bytes, expected_ngram_order, self.limits)
    }

    /// Encodes a Markov chain into a byte vector.
    ///
    /// # Errors
    /// Returns `StorageError` if encoding or validation fails.
    pub fn encode_chain(
        &self,
        chain: &MarkovChain,
        min_edge_count: Count,
        compression_mode: StorageCompressionMode,
    ) -> Result<Vec<u8>, StorageError> {
        let sections = write::compile_chain(chain, min_edge_count, self.limits)?;
        let payload = write::encode_storage(&sections, compression_mode, self.limits)?;
        Ok(payload)
    }

    /// Decodes a storage snapshot from a byte slice.
    ///
    /// # Errors
    /// Returns `StorageError` if decoding fails.
    pub fn decode_snapshot(&self, bytes: &[u8]) -> Result<StorageSnapshot, StorageError> {
        read::decode_snapshot(bytes, self.limits)
    }

    /// Encodes a storage snapshot into a byte vector.
    ///
    /// # Errors
    /// Returns `StorageError` if encoding or validation fails.
    pub fn encode_snapshot(
        &self,
        snapshot: StorageSnapshot,
        compression_mode: StorageCompressionMode,
    ) -> Result<Vec<u8>, StorageError> {
        let chain = self.snapshot_to_chain(snapshot)?;
        self.encode_chain(&chain, Count::new(1), compression_mode)
    }

    /// Converts a storage snapshot to a Markov chain.
    ///
    /// # Errors
    /// Returns `StorageError` if the snapshot is invalid or conversion fails.
    pub fn snapshot_to_chain(
        &self,
        snapshot: StorageSnapshot,
    ) -> Result<MarkovChain, StorageError> {
        validate_snapshot(&snapshot)?;

        let order = NgramOrder::new(snapshot.ngram_order())?;
        self.limits.check_vocab_tokens(&snapshot.tokens)?;
        let mut models = (0..snapshot.ngram_order())
            .map(|_| HashMap::new())
            .collect::<Vec<_>>();
        for model in snapshot.models {
            let mut prefixes = HashMap::new();
            for entry in model.entries {
                let mut edges = HashMap::new();
                for edge in entry.edges {
                    edges.insert(
                        markov_core::TokenId::new(edge.next),
                        markov_core::Count::new(edge.count),
                    );
                }
                prefixes.insert(
                    markov_core::Prefix::new(
                        entry
                            .prefix
                            .into_iter()
                            .map(markov_core::TokenId::new)
                            .collect(),
                    ),
                    edges,
                );
            }
            let slot = models.get_mut(model.order - 1).ok_or_else(|| {
                StorageError::Format(format!(
                    "snapshot model order {} is out of range",
                    model.order
                ))
            })?;
            *slot = prefixes;
        }

        let starts = snapshot
            .starts
            .into_iter()
            .map(|entry| {
                (
                    markov_core::Prefix::new(
                        entry
                            .prefix
                            .into_iter()
                            .map(markov_core::TokenId::new)
                            .collect(),
                    ),
                    markov_core::Count::new(entry.count),
                )
            })
            .collect::<HashMap<_, _>>();

        let registry = markov_core::token::TokenRegistry::from_tokens(snapshot.tokens)?;

        Ok(MarkovChain::from_parts(order, registry, models, starts)?)
    }
}

/// Converts a Markov chain to a storage snapshot.
///
/// # Errors
/// Returns `StorageError` if the chain is invalid or conversion fails.
fn chain_to_snapshot(
    chain: &MarkovChain,
    compression_mode: StorageCompression,
) -> Result<StorageSnapshot, StorageError> {
    validate_special_tokens(chain.registry().tokens())?;

    if chain.models().len() != chain.order().as_usize()? {
        return Err(StorageError::Format(
            "model count does not match ngram order".to_owned(),
        ));
    }

    let mut starts = chain
        .starts()
        .iter()
        .map(|(prefix, count)| SnapshotEntry {
            prefix: prefix.as_slice().iter().map(|&id| id.get()).collect(),
            count: count.get(),
        })
        .collect::<Vec<_>>();
    starts.sort_unstable_by(|left, right| left.prefix.cmp(&right.prefix));

    let mut models = Vec::with_capacity(chain.models().len());
    for (index, model) in chain.models().iter().enumerate().rev() {
        let order = index + 1;
        let mut entries = model
            .iter()
            .map(|(prefix, edges)| {
                let mut snapshot_edges = edges
                    .iter()
                    .map(|(next, count)| SnapshotEdge {
                        next: next.get(),
                        count: count.get(),
                    })
                    .collect::<Vec<_>>();
                snapshot_edges.sort_unstable_by_key(|edge| edge.next);
                SnapshotModelEntry {
                    prefix: prefix.as_slice().iter().map(|&id| id.get()).collect(),
                    edges: snapshot_edges,
                }
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| left.prefix.cmp(&right.prefix));
        models.push(SnapshotModel { order, entries });
    }

    let snapshot = StorageSnapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        source: SnapshotSource {
            storage_version: VERSION,
            ngram_order: chain.order().as_usize()?,
            compression: compression_mode,
        },
        tokens: chain.registry().tokens().to_vec(),
        starts,
        models,
    };
    validate_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn validate_snapshot(snapshot: &StorageSnapshot) -> Result<(), StorageError> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
        return Err(StorageError::Format(format!(
            "unsupported snapshot schema version: {}",
            snapshot.schema_version
        )));
    }
    if snapshot.source.storage_version != VERSION {
        return Err(StorageError::Version(snapshot.source.storage_version));
    }
    NgramOrder::new(snapshot.source.ngram_order)?;
    // Check the cardinality before allocating anything based on untrusted order.
    if snapshot.models.len() != snapshot.source.ngram_order {
        return Err(StorageError::Format(
            "snapshot must contain exactly one model per order".into(),
        ));
    }
    let mut orders = BTreeSet::new();
    for model in &snapshot.models {
        if model.order == 0
            || model.order > snapshot.source.ngram_order
            || !orders.insert(model.order)
        {
            return Err(StorageError::Format(
                "duplicate or out-of-range snapshot model order".into(),
            ));
        }
        let mut prefixes = BTreeSet::new();
        for entry in &model.entries {
            if !prefixes.insert(&entry.prefix) {
                return Err(StorageError::Format(
                    "duplicate snapshot model prefix".into(),
                ));
            }
            let mut targets = BTreeSet::new();
            for edge in &entry.edges {
                if !targets.insert(edge.next) {
                    return Err(StorageError::Format(
                        "duplicate snapshot edge target".into(),
                    ));
                }
            }
        }
    }
    let mut starts = BTreeSet::new();
    for entry in &snapshot.starts {
        if !starts.insert(&entry.prefix) {
            return Err(StorageError::Format(
                "duplicate snapshot start prefix".into(),
            ));
        }
    }
    Ok(())
}

fn compression_mode_from_flags(flags: u32) -> Result<StorageCompression, StorageError> {
    match vocab_blob_compression_flags(flags)? {
        0 => Ok(StorageCompression::Uncompressed),
        FLAG_VOCAB_BLOB_RLE => Ok(StorageCompression::Rle),
        FLAG_VOCAB_BLOB_ZSTD => Ok(StorageCompression::Zstd),
        _ => Err(StorageError::Format(
            "unsupported vocab blob compression flags".to_owned(),
        )),
    }
}

fn validate_special_tokens(tokens: &[String]) -> Result<(), StorageError> {
    let Some(first) = tokens.first() else {
        return Err(StorageError::Format("vocabulary is empty".to_owned()));
    };
    if first != BOS_TOKEN {
        return Err(StorageError::Format("token id 0 must be <BOS>".to_owned()));
    }

    let Some(second) = tokens.get(1) else {
        return Err(StorageError::Format(
            "vocabulary is missing <EOS>".to_owned(),
        ));
    };
    if second != EOS_TOKEN {
        return Err(StorageError::Format("token id 1 must be <EOS>".to_owned()));
    }

    Ok(())
}

fn validate_token_id(token_id: u32, token_count: u32, context: &str) -> Result<(), StorageError> {
    if token_id >= token_count {
        return Err(StorageError::Format(format!(
            "{context}: token id {token_id} is out of range"
        )));
    }

    Ok(())
}

fn descriptor_count_for_ngram_order(ngram_order: usize) -> Result<u64, StorageError> {
    let ngram_order = u64_from_usize(ngram_order, "ngram order")?;
    SECTION_METADATA_COUNT
        .checked_add(ngram_order)
        .ok_or_else(|| StorageError::Format("section count overflow".to_owned()))
}

fn bytes_for_len(len: usize, element_size: u64, context: &str) -> Result<u64, StorageError> {
    let len = u64_from_usize(len, context)?;
    len.checked_mul(element_size)
        .ok_or_else(|| StorageError::Format(format!("{context} byte size overflow")))
}

fn align_to_eight(value: u64) -> Result<u64, StorageError> {
    value
        .checked_next_multiple_of(8)
        .ok_or_else(|| StorageError::Format("alignment overflow".into()))
}

fn checked_add(left: u64, right: u64, context: &str) -> Result<u64, StorageError> {
    left.checked_add(right)
        .ok_or_else(|| StorageError::Format(format!("{context} overflow")))
}

fn usize_from_u32(value: u32, context: &str) -> Result<usize, StorageError> {
    usize::try_from(value)
        .map_err(|_error| StorageError::Format(format!("{context} exceeds usize range")))
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize, StorageError> {
    usize::try_from(value)
        .map_err(|_error| StorageError::Format(format!("{context} exceeds usize range")))
}

fn u32_from_usize(value: usize, context: &str) -> Result<u32, StorageError> {
    u32::try_from(value)
        .map_err(|_error| StorageError::Format(format!("{context} exceeds u32 range")))
}

fn u64_from_usize(value: usize, context: &str) -> Result<u64, StorageError> {
    u64::try_from(value)
        .map_err(|_error| StorageError::Format(format!("{context} exceeds u64 range")))
}

fn aligned_metadata_end(section_count: u64) -> Result<u64, StorageError> {
    let header_size = u64_from_usize(HEADER_SIZE, "header size")?;
    let descriptor_size = u64_from_usize(DESCRIPTOR_SIZE, "section descriptor size")?;
    let descriptor_bytes = section_count.checked_mul(descriptor_size).ok_or_else(|| {
        StorageError::Format("section descriptor table byte size overflow".to_owned())
    })?;
    align_to_eight(checked_add(header_size, descriptor_bytes, "metadata size")?)
}

fn start_record_size(order: usize) -> Result<u64, StorageError> {
    let prefix_bytes = bytes_for_len(order, 4, "start record prefix")?;
    checked_add(prefix_bytes, 8, "start record size")
}

fn model_record_size(order: usize) -> Result<u64, StorageError> {
    let prefix_bytes = bytes_for_len(order, 4, "model record prefix")?;
    let with_edges = checked_add(prefix_bytes, 4, "model record edge_start size")?;
    let with_len = checked_add(with_edges, 4, "model record edge_len size")?;
    checked_add(with_len, 8, "model record total size")
}

fn compute_checksum(bytes: &[u8]) -> Result<u64, StorageError> {
    if bytes.len() < HEADER_SIZE {
        return Err(StorageError::Format(
            "cannot compute checksum: data is shorter than header".to_owned(),
        ));
    }

    let checksum_range = CHECKSUM_OFFSET..(CHECKSUM_OFFSET + CHECKSUM_SIZE);
    let mut hash = FNV1A64_OFFSET_BASIS;

    for (index, byte) in bytes.iter().enumerate() {
        let normalized = if checksum_range.contains(&index) {
            0_u8
        } else {
            *byte
        };

        hash ^= u64::from(normalized);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }

    Ok(hash)
}

fn vocab_blob_compression_flags(flags: u32) -> Result<u32, StorageError> {
    let compression_flags = flags & SUPPORTED_FLAGS;
    if compression_flags.count_ones() > 1 {
        return Err(StorageError::Format(
            "multiple vocab blob compression flags are set".to_owned(),
        ));
    }

    let unsupported = flags & !SUPPORTED_FLAGS;
    if unsupported != 0 {
        return Err(StorageError::Format(format!(
            "unsupported storage flags: 0x{unsupported:08x}"
        )));
    }

    Ok(compression_flags)
}

pub(crate) fn write_u64_at(
    bytes: &mut [u8],
    offset: usize,
    value: u64,
) -> Result<(), StorageError> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| StorageError::Format("write_u64_at: offset overflow".to_owned()))?;
    let slice = bytes
        .get_mut(offset..end)
        .ok_or_else(|| StorageError::Format("write_u64_at: offset out of bounds".to_owned()))?;

    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}
