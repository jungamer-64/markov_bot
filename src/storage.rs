use std::{collections::HashMap, io, path::Path};

use tokio::fs;

use crate::{
    config::DynError,
    markov::{BOS_TOKEN, Count, EOS_TOKEN, MarkovChain, TokenId},
};

mod read;
mod write;

#[cfg(test)]
mod tests;

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

#[derive(Debug, Clone, Copy)]
struct Header {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    tokenizer_version: u32,
    normalization_flags: u32,
    token_count: u32,
    start_count: u32,
    model3_pair_count: u32,
    model3_prefix_count: u32,
    model3_edge_count: u32,
    model2_prefix_count: u32,
    model2_edge_count: u32,
    model1_prefix_count: u32,
    model1_edge_count: u32,
    vocab_offsets_offset: u64,
    vocab_blob_offset: u64,
    start_offset: u64,
    model3_pair_offset: u64,
    model3_prefix_offset: u64,
    model3_edge_offset: u64,
    model2_prefix_offset: u64,
    model2_edge_offset: u64,
    model1_prefix_offset: u64,
    model1_edge_offset: u64,
    file_size: u64,
    checksum: u64,
}

#[derive(Debug, Clone, Copy)]
struct StartRecord {
    prefix_id: u32,
    cumulative: u32,
}

#[derive(Debug, Clone, Copy)]
struct Pair3Record {
    w1: u32,
    w2: u32,
    prefix_start: u32,
    prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
struct Prefix3Record {
    w3: u32,
    edge_start: u32,
    edge_len: u32,
    total: u32,
}

#[derive(Debug, Clone, Copy)]
struct Prefix2Record {
    w1: u32,
    w2: u32,
    edge_start: u32,
    edge_len: u32,
    total: u32,
}

#[derive(Debug, Clone, Copy)]
struct Prefix1Record {
    w1: u32,
    edge_start: u32,
    edge_len: u32,
    total: u32,
}

#[derive(Debug, Clone, Copy)]
struct EdgeRecord {
    next: u32,
    cumulative: u32,
}

#[derive(Debug)]
struct CompiledStorage {
    vocab_offsets: Vec<u64>,
    vocab_blob: Vec<u8>,
    starts: Vec<StartRecord>,
    model3_pairs: Vec<Pair3Record>,
    model3_prefixes: Vec<Prefix3Record>,
    model3_edges: Vec<EdgeRecord>,
    model2_prefixes: Vec<Prefix2Record>,
    model2_edges: Vec<EdgeRecord>,
    model1_prefixes: Vec<Prefix1Record>,
    model1_edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone, Copy)]
struct SectionCounts {
    token: u32,
    start: u32,
    model3_pair: u32,
    model3_prefix: u32,
    model3_edge: u32,
    model2_prefix: u32,
    model2_edge: u32,
    model1_prefix: u32,
    model1_edge: u32,
}

#[derive(Debug, Clone, Copy)]
struct SectionSizes {
    vocab_offsets: u64,
    vocab_blob: u64,
    starts: u64,
    model3_pairs: u64,
    model3_prefixes: u64,
    model3_edges: u64,
    model2_prefixes: u64,
    model2_edges: u64,
    model1_prefixes: u64,
    model1_edges: u64,
}

#[derive(Debug, Clone)]
struct SectionRanges {
    vocab_offsets: std::ops::Range<usize>,
    vocab_blob_area: std::ops::Range<usize>,
    starts: std::ops::Range<usize>,
    model3_pairs: std::ops::Range<usize>,
    model3_prefixes: std::ops::Range<usize>,
    model3_edges: std::ops::Range<usize>,
    model2_prefixes: std::ops::Range<usize>,
    model2_edges: std::ops::Range<usize>,
    model1_prefixes: std::ops::Range<usize>,
    model1_edges: std::ops::Range<usize>,
}

#[derive(Debug)]
struct ParsedStorage {
    id_to_token: Vec<String>,
    starts: Vec<StartRecord>,
    model3_pairs: Vec<Pair3Record>,
    model3_prefixes: Vec<Prefix3Record>,
    model3_edges: Vec<EdgeRecord>,
    model2_prefixes: Vec<Prefix2Record>,
    model2_edges: Vec<EdgeRecord>,
    model1_prefixes: Vec<Prefix1Record>,
    model1_edges: Vec<EdgeRecord>,
}

type Model3Build = (
    Vec<Pair3Record>,
    Vec<Prefix3Record>,
    Vec<EdgeRecord>,
    HashMap<[TokenId; 3], u32>,
);

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
