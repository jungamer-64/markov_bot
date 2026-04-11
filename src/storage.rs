use std::{io, path::Path};

use tokio::fs;

use crate::{
    config::DynError,
    markov::{BOS_TOKEN, Count, EOS_TOKEN, MarkovChain, validate_ngram_order},
};

mod read;
mod types;
mod write;

#[cfg(test)]
mod tests;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StorageCompressionMode {
    Auto,
    Uncompressed,
    Rle,
    Zstd,
    Lz4Flex,
}

impl StorageCompressionMode {
    pub(crate) fn parse(raw: &str) -> Result<Self, DynError> {
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

    pub(crate) const fn as_env_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Uncompressed => "none",
            Self::Rle => "rle",
            Self::Zstd => "zstd",
            Self::Lz4Flex => "lz4_flex",
        }
    }
}

pub(crate) async fn load_chain(
    path: &Path,
    expected_ngram_order: usize,
) -> Result<MarkovChain, DynError> {
    validate_ngram_order(expected_ngram_order, "expected_ngram_order")?;

    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return MarkovChain::new(expected_ngram_order);
        }
        Err(error) => return Err(error.into()),
    };

    read::decode_chain(bytes.as_slice(), expected_ngram_order)
}

pub(crate) async fn save_chain(
    path: &Path,
    chain: &MarkovChain,
    min_edge_count: Count,
    compression_mode: StorageCompressionMode,
) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let sections = write::compile_chain(chain, min_edge_count)?;
    let payload = write::encode_storage(&sections, compression_mode)?;

    read::decode_chain(payload.as_slice(), chain.ngram_order)?;

    fs::write(path, payload).await?;

    Ok(())
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
