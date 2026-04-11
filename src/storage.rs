use std::{io, path::Path};

use tokio::fs;

use crate::{
    config::DynError,
    markov::{BOS_TOKEN, Count, EOS_TOKEN, MarkovChain, TokenId},
};

mod read;
mod types;
mod write;

#[cfg(test)]
mod tests;

use types::{
    EdgeRecord, FixedRecord, Header, Model1Sections, Model2Sections, Model3PrefixIndex,
    Model3Sections, Pair2Record, Pair3Record, Prefix1Record, Prefix2Record, Prefix3Record,
    SectionDescriptor, SectionEntry, SectionKind, SectionTable, StartRecord, StorageSections,
    VocabSections,
};

const MAGIC: [u8; 8] = *b"MKV3BIN\0";
const VERSION: u32 = 5;
const FLAGS: u32 = 0;
const FLAG_VOCAB_BLOB_RLE: u32 = 1 << 0;
const FLAG_VOCAB_BLOB_ZSTD: u32 = 1 << 1;
const FLAG_VOCAB_BLOB_LZ4_FLEX: u32 = 1 << 2;
const SUPPORTED_FLAGS: u32 = FLAG_VOCAB_BLOB_RLE | FLAG_VOCAB_BLOB_ZSTD | FLAG_VOCAB_BLOB_LZ4_FLEX;
const TOKENIZER_VERSION: u32 = 1;
const NORMALIZATION_FLAGS: u32 = 0;
const CHECKSUM_PLACEHOLDER: u64 = 0;

const HEADER_SIZE: usize = 44;
const DESCRIPTOR_SIZE: usize = 24;
const SECTION_COUNT: usize = 11;
const SECTION_COUNT_U32: u32 = 11;
const CHECKSUM_SIZE: usize = std::mem::size_of::<u64>();
const CHECKSUM_OFFSET: usize = HEADER_SIZE - CHECKSUM_SIZE;

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

const START_RECORD_SIZE: u64 = 12;
const PAIR3_RECORD_SIZE: u64 = 16;
const PAIR2_RECORD_SIZE: u64 = 12;
const PREFIX3_RECORD_SIZE: u64 = 20;
const EDGE_RECORD_SIZE: u64 = 12;
const PREFIX2_RECORD_SIZE: u64 = 24;
const PREFIX1_RECORD_SIZE: u64 = 20;

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

pub(crate) async fn load_chain(path: &Path) -> Result<MarkovChain, DynError> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(MarkovChain::default()),
        Err(error) => return Err(error.into()),
    };

    read::decode_chain(bytes.as_slice())
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

    read::decode_chain(payload.as_slice())?;

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

fn validate_token_id(token_id: u32, token_count: u32, context: &str) -> Result<(), DynError> {
    if token_id >= token_count {
        return Err(format!("{context}: token id {token_id} is out of range").into());
    }

    Ok(())
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

fn aligned_metadata_end(section_count: usize) -> Result<u64, DynError> {
    let header_size = u64_from_usize(HEADER_SIZE, "header size")?;
    let descriptor_size = u64_from_usize(DESCRIPTOR_SIZE, "section descriptor size")?;
    let descriptor_bytes =
        bytes_for_len(section_count, descriptor_size, "section descriptor table")?;
    Ok(align_to_eight(checked_add(
        header_size,
        descriptor_bytes,
        "metadata size",
    )?))
}

fn fixed_section_bytes<T: FixedRecord>(records: &[T], context: &str) -> Result<u64, DynError> {
    bytes_for_len(records.len(), T::SIZE, context)
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

    Ok(compression_flags)
}
