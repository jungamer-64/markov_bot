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
    CompiledStorage, EdgeRecord, Header, Model3Build, Pair3Record, ParsedStorage, Prefix1Record,
    Prefix2Record, Prefix3Record, SectionCounts, SectionRanges, SectionSizes, StartRecord,
};

const MAGIC: [u8; 8] = *b"MKV3BIN\0";
const VERSION: u32 = 1;
const FLAGS: u32 = 0;
const TOKENIZER_VERSION: u32 = 1;
const NORMALIZATION_FLAGS: u32 = 0;
const CHECKSUM: u64 = 0;

const HEADER_SIZE: usize = 156;

const START_RECORD_SIZE: u64 = 8;
const PAIR3_RECORD_SIZE: u64 = 16;
const PREFIX3_RECORD_SIZE: u64 = 16;
const EDGE_RECORD_SIZE: u64 = 8;
const PREFIX2_RECORD_SIZE: u64 = 20;
const PREFIX1_RECORD_SIZE: u64 = 16;

pub async fn load_chain(path: &Path) -> Result<MarkovChain, DynError> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(MarkovChain::default()),
        Err(error) => return Err(error.into()),
    };

    read::decode_chain(bytes.as_slice())
}

pub async fn save_chain(path: &Path, chain: &MarkovChain) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let compiled = write::compile_chain(chain)?;
    let payload = write::encode_storage(&compiled)?;

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
    usize::try_from(value).map_err(|_| format!("{context} exceeds usize range").into())
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize, DynError> {
    usize::try_from(value).map_err(|_| format!("{context} exceeds usize range").into())
}

fn u32_from_usize(value: usize, context: &str) -> Result<u32, DynError> {
    u32::try_from(value).map_err(|_| format!("{context} exceeds u32 range").into())
}

fn u64_from_usize(value: usize, context: &str) -> Result<u64, DynError> {
    u64::try_from(value).map_err(|_| format!("{context} exceeds u64 range").into())
}
