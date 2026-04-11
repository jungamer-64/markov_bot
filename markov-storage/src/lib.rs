use std::collections::{BTreeSet, HashMap};

use markov_core::{BOS_TOKEN, Count, EOS_TOKEN, MarkovChain, validate_ngram_order};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod config {
    pub(crate) type DynError = super::StorageError;
}

mod markov {
    pub(crate) use markov_core::*;
}

mod read;
mod types;
mod write;

#[cfg(test)]
mod tests;

type DynError = StorageError;

const MAGIC: [u8; 8] = *b"MKV3BIN\0";
const VERSION: u32 = 8;
const FLAGS: u32 = 0;
const FLAG_VOCAB_BLOB_RLE: u32 = 1 << 0;
const FLAG_VOCAB_BLOB_ZSTD: u32 = 1 << 1;
const FLAG_VOCAB_BLOB_LZ4_FLEX: u32 = 1 << 2;
const SUPPORTED_FLAGS: u32 = FLAG_VOCAB_BLOB_RLE | FLAG_VOCAB_BLOB_ZSTD | FLAG_VOCAB_BLOB_LZ4_FLEX;
const TOKENIZER_VERSION: u32 = 1;
const NORMALIZATION_FLAGS: u32 = 0;
const CHECKSUM_PLACEHOLDER: u64 = 0;

const HEADER_SIZE: usize = 52;
const DESCRIPTOR_SIZE: usize = 24;
const CHECKSUM_SIZE: usize = std::mem::size_of::<u64>();
const CHECKSUM_OFFSET: usize = HEADER_SIZE - CHECKSUM_SIZE;
const SECTION_COUNT_BASE: u64 = 3;
const START_SECTION_HEADER_SIZE: u64 = 4;
const MODEL_SECTION_HEADER_SIZE: u64 = 8;
const EDGE_RECORD_SIZE: u64 = 12;

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;
const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("{0}")]
    Invalid(String),
    #[error("zstd decompression failed: {0}")]
    Zstd(#[from] std::io::Error),
    #[error("lz4 decompression failed: {0}")]
    Lz4(#[from] lz4_flex::block::DecompressError),
}

impl From<&str> for StorageError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

impl From<String> for StorageError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

impl From<markov_core::MarkovError> for StorageError {
    fn from(value: markov_core::MarkovError) -> Self {
        Self::Invalid(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageCompressionMode {
    Auto,
    Uncompressed,
    Rle,
    Zstd,
    Lz4Flex,
}

impl StorageCompressionMode {
    pub fn parse(raw: &str) -> Result<Self, DynError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "none" | "off" | "uncompressed" => Ok(Self::Uncompressed),
            "rle" | "vocab_rle" | "vocab-blob-rle" => Ok(Self::Rle),
            "zstd" => Ok(Self::Zstd),
            "lz4" | "lz4_flex" | "lz4-flex" => Ok(Self::Lz4Flex),
            _ => Err(format!(
                "unsupported STORAGE_COMPRESSION value: {raw} (expected: auto|none|rle|zstd|lz4_flex)"
            )
            .into()),
        }
    }

    pub const fn as_env_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Uncompressed => "none",
            Self::Rle => "rle",
            Self::Zstd => "zstd",
            Self::Lz4Flex => "lz4_flex",
        }
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
    pub compression: StorageCompressionMode,
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
    pub fn ngram_order(&self) -> usize {
        self.source.ngram_order
    }
}

pub fn decode_v8_chain(bytes: &[u8], expected_ngram_order: usize) -> Result<MarkovChain, DynError> {
    read::decode_chain(bytes, expected_ngram_order)
}

pub fn encode_v8_chain(
    chain: &MarkovChain,
    min_edge_count: Count,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    let sections = write::compile_chain(chain, min_edge_count)?;
    let payload = write::encode_storage(&sections, compression_mode)?;
    read::decode_chain(payload.as_slice(), chain.ngram_order)?;
    Ok(payload)
}

pub fn decode_v8_snapshot(bytes: &[u8]) -> Result<StorageSnapshot, DynError> {
    read::decode_snapshot(bytes)
}

pub fn encode_v8_snapshot(
    snapshot: &StorageSnapshot,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    let chain = snapshot_to_chain(snapshot)?;
    encode_v8_chain(&chain, 1, compression_mode)
}

pub fn snapshot_to_chain(snapshot: &StorageSnapshot) -> Result<MarkovChain, DynError> {
    validate_snapshot(snapshot)?;

    let token_to_id = build_token_index(snapshot.tokens.as_slice())?;
    let mut models = (0..snapshot.ngram_order())
        .map(|_| HashMap::new())
        .collect::<Vec<_>>();
    for model in &snapshot.models {
        let mut prefixes = HashMap::new();
        for entry in &model.entries {
            let mut edges = HashMap::new();
            for edge in &entry.edges {
                edges.insert(edge.next, edge.count);
            }
            prefixes.insert(entry.prefix.clone(), edges);
        }
        let slot = models
            .get_mut(model.order - 1)
            .ok_or_else(|| format!("snapshot model order {} is out of range", model.order))?;
        *slot = prefixes;
    }

    let starts = snapshot
        .starts
        .iter()
        .map(|entry| (entry.prefix.clone(), entry.count))
        .collect::<HashMap<_, _>>();

    Ok(MarkovChain {
        ngram_order: snapshot.ngram_order(),
        token_to_id,
        id_to_token: snapshot.tokens.clone(),
        models,
        starts,
    })
}

pub fn chain_to_snapshot(
    chain: &MarkovChain,
    compression_mode: StorageCompressionMode,
) -> Result<StorageSnapshot, DynError> {
    validate_ngram_order(chain.ngram_order, "chain ngram order")
        .map_err(|error| error.to_string())?;
    validate_special_tokens(chain.id_to_token.as_slice())?;
    validate_token_index(chain)?;

    if chain.models.len() != chain.ngram_order {
        return Err("model count does not match ngram order".into());
    }

    let mut starts = chain
        .starts
        .iter()
        .map(|(prefix, count)| SnapshotEntry {
            prefix: prefix.clone(),
            count: *count,
        })
        .collect::<Vec<_>>();
    starts.sort_unstable_by(|left, right| left.prefix.cmp(&right.prefix));

    let mut models = Vec::with_capacity(chain.models.len());
    for (index, model) in chain.models.iter().enumerate().rev() {
        let order = index + 1;
        let mut entries = model
            .iter()
            .map(|(prefix, edges)| {
                let mut snapshot_edges = edges
                    .iter()
                    .map(|(next, count)| SnapshotEdge {
                        next: *next,
                        count: *count,
                    })
                    .collect::<Vec<_>>();
                snapshot_edges.sort_unstable_by_key(|edge| edge.next);
                SnapshotModelEntry {
                    prefix: prefix.clone(),
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
            ngram_order: chain.ngram_order,
            compression: compression_mode,
        },
        tokens: chain.id_to_token.clone(),
        starts,
        models,
    };
    validate_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn validate_snapshot(snapshot: &StorageSnapshot) -> Result<(), DynError> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported snapshot schema version: {}",
            snapshot.schema_version
        )
        .into());
    }

    validate_ngram_order(snapshot.source.ngram_order, "snapshot ngram order")
        .map_err(|error| error.to_string())?;
    validate_special_tokens(snapshot.tokens.as_slice())?;
    let token_to_id = build_token_index(snapshot.tokens.as_slice())?;
    let token_count = u32_from_usize(snapshot.tokens.len(), "snapshot token count")?;

    let expected_orders = (1..=snapshot.source.ngram_order)
        .rev()
        .collect::<BTreeSet<_>>();
    let actual_orders = snapshot
        .models
        .iter()
        .map(|model| model.order)
        .collect::<BTreeSet<_>>();
    if actual_orders != expected_orders {
        return Err("snapshot models must cover every order from ngram_order down to 1".into());
    }

    let mut seen_starts = BTreeSet::new();
    for entry in &snapshot.starts {
        validate_snapshot_prefix(
            entry.prefix.as_slice(),
            snapshot.source.ngram_order,
            token_count,
            "snapshot start",
        )?;
        if entry.count == 0 {
            return Err("snapshot start count must be greater than zero".into());
        }
        if !seen_starts.insert(entry.prefix.clone()) {
            return Err("duplicate snapshot start prefix".into());
        }
    }

    if token_to_id.get(BOS_TOKEN) != Some(&0) || token_to_id.get(EOS_TOKEN) != Some(&1) {
        return Err("snapshot special token ids are invalid".into());
    }

    for model in &snapshot.models {
        let mut seen_prefixes = BTreeSet::new();
        for entry in &model.entries {
            validate_snapshot_prefix(
                entry.prefix.as_slice(),
                model.order,
                token_count,
                "snapshot model entry",
            )?;
            if !seen_prefixes.insert(entry.prefix.clone()) {
                return Err(
                    format!("duplicate snapshot model prefix for order {}", model.order).into(),
                );
            }
            if entry.edges.is_empty() {
                return Err(format!(
                    "snapshot model entry for order {} has no edges",
                    model.order
                )
                .into());
            }
            let mut seen_edges = BTreeSet::new();
            for edge in &entry.edges {
                validate_token_id(edge.next, token_count, "snapshot edge")?;
                if edge.count == 0 {
                    return Err("snapshot edge count must be greater than zero".into());
                }
                if !seen_edges.insert(edge.next) {
                    return Err("duplicate snapshot edge target".into());
                }
            }
        }
    }

    Ok(())
}

fn validate_snapshot_prefix(
    prefix: &[u32],
    expected_len: usize,
    token_count: u32,
    context: &str,
) -> Result<(), DynError> {
    if prefix.len() != expected_len {
        return Err(format!(
            "{context} prefix length mismatch: expected {expected_len}, got {}",
            prefix.len()
        )
        .into());
    }
    for token_id in prefix {
        validate_token_id(*token_id, token_count, context)?;
    }
    Ok(())
}

fn build_token_index(tokens: &[String]) -> Result<HashMap<String, u32>, DynError> {
    let mut index = HashMap::new();

    for (position, token) in tokens.iter().enumerate() {
        let token_id = u32_from_usize(position, "token id")?;

        if index.insert(token.clone(), token_id).is_some() {
            return Err(format!("duplicate token in vocab: {token}").into());
        }
    }

    Ok(index)
}

fn compression_mode_from_flags(flags: u32) -> Result<StorageCompressionMode, DynError> {
    match vocab_blob_compression_flags(flags)? {
        0 => Ok(StorageCompressionMode::Uncompressed),
        FLAG_VOCAB_BLOB_RLE => Ok(StorageCompressionMode::Rle),
        FLAG_VOCAB_BLOB_ZSTD => Ok(StorageCompressionMode::Zstd),
        FLAG_VOCAB_BLOB_LZ4_FLEX => Ok(StorageCompressionMode::Lz4Flex),
        _ => Err("unsupported vocab blob compression flags".into()),
    }
}

fn validate_special_tokens(tokens: &[String]) -> Result<(), DynError> {
    let Some(first) = tokens.first() else {
        return Err("vocabulary is empty".into());
    };
    if first != BOS_TOKEN {
        return Err("token id 0 must be <BOS>".into());
    }

    let Some(second) = tokens.get(1) else {
        return Err("vocabulary is missing <EOS>".into());
    };
    if second != EOS_TOKEN {
        return Err("token id 1 must be <EOS>".into());
    }

    Ok(())
}

fn validate_token_index(chain: &MarkovChain) -> Result<(), DynError> {
    if chain.token_to_id.len() != chain.id_to_token.len() {
        return Err("token index size mismatch".into());
    }

    for (index, token) in chain.id_to_token.iter().enumerate() {
        let expected_id = u32_from_usize(index, "token index")?;
        let actual_id = chain
            .token_to_id
            .get(token)
            .copied()
            .ok_or_else(|| format!("token_to_id is missing '{token}'"))?;
        if actual_id != expected_id {
            return Err(format!(
                "token_to_id mismatch for '{token}': expected {expected_id}, got {actual_id}"
            )
            .into());
        }
    }

    Ok(())
}

fn validate_token_id(token_id: u32, token_count: u32, context: &str) -> Result<(), DynError> {
    if token_id >= token_count {
        return Err(format!("{context}: token id {token_id} is out of range").into());
    }

    Ok(())
}

fn descriptor_count_for_ngram_order(ngram_order: usize) -> Result<u64, DynError> {
    let ngram_order = u64_from_usize(ngram_order, "ngram order")?;
    SECTION_COUNT_BASE
        .checked_add(ngram_order)
        .ok_or_else(|| "section count overflow".into())
}

fn bytes_for_len(len: usize, element_size: u64, context: &str) -> Result<u64, DynError> {
    let len = u64_from_usize(len, context)?;
    len.checked_mul(element_size)
        .ok_or_else(|| format!("{context} byte size overflow").into())
}

const fn align_to_eight(value: u64) -> u64 {
    value.next_multiple_of(8)
}

fn checked_add(left: u64, right: u64, context: &str) -> Result<u64, DynError> {
    left.checked_add(right)
        .ok_or_else(|| format!("{context} overflow").into())
}

fn usize_from_u32(value: u32, context: &str) -> Result<usize, DynError> {
    usize::try_from(value).map_err(|_error| format!("{context} exceeds usize range").into())
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize, DynError> {
    usize::try_from(value).map_err(|_error| format!("{context} exceeds usize range").into())
}

fn u32_from_usize(value: usize, context: &str) -> Result<u32, DynError> {
    u32::try_from(value).map_err(|_error| format!("{context} exceeds u32 range").into())
}

fn u64_from_usize(value: usize, context: &str) -> Result<u64, DynError> {
    u64::try_from(value).map_err(|_error| format!("{context} exceeds u64 range").into())
}

fn aligned_metadata_end(section_count: u64) -> Result<u64, DynError> {
    let header_size = u64_from_usize(HEADER_SIZE, "header size")?;
    let descriptor_size = u64_from_usize(DESCRIPTOR_SIZE, "section descriptor size")?;
    let descriptor_bytes = section_count
        .checked_mul(descriptor_size)
        .ok_or("section descriptor table byte size overflow")?;
    Ok(align_to_eight(checked_add(
        header_size,
        descriptor_bytes,
        "metadata size",
    )?))
}

fn start_record_size(order: usize) -> Result<u64, DynError> {
    let prefix_bytes = bytes_for_len(order, 4, "start record prefix")?;
    checked_add(prefix_bytes, 8, "start record size")
}

fn model_record_size(order: usize) -> Result<u64, DynError> {
    let prefix_bytes = bytes_for_len(order, 4, "model record prefix")?;
    let with_edges = checked_add(prefix_bytes, 4, "model record edge_start size")?;
    let with_len = checked_add(with_edges, 4, "model record edge_len size")?;
    checked_add(with_len, 8, "model record total size")
}

fn compute_checksum(bytes: &[u8]) -> Result<u64, DynError> {
    if bytes.len() < HEADER_SIZE {
        return Err("cannot compute checksum: data is shorter than header".into());
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

fn vocab_blob_compression_flags(flags: u32) -> Result<u32, DynError> {
    let compression_flags = flags & SUPPORTED_FLAGS;
    if compression_flags.count_ones() > 1 {
        return Err("multiple vocab blob compression flags are set".into());
    }

    let unsupported = flags & !SUPPORTED_FLAGS;
    if unsupported != 0 {
        return Err(format!("unsupported storage flags: 0x{unsupported:08x}").into());
    }

    Ok(compression_flags)
}
