use std::{collections::HashMap, io, path::Path, str};

use tokio::fs;

use crate::{
    config::DynError,
    markov::{BOS_TOKEN, Count, EOS_TOKEN, MarkovChain, TokenId},
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

    decode_chain(bytes.as_slice())
}

pub async fn save_chain(path: &Path, chain: &MarkovChain) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let compiled = compile_chain(chain)?;
    let payload = encode_storage(&compiled)?;

    fs::write(path, payload).await?;

    Ok(())
}

fn compile_chain(chain: &MarkovChain) -> Result<CompiledStorage, DynError> {
    validate_special_tokens(chain.id_to_token.as_slice())?;
    validate_token_index(chain)?;

    let token_count = u32_from_usize(chain.id_to_token.len(), "token count")?;
    let (vocab_offsets, vocab_blob) = build_vocab(chain.id_to_token.as_slice())?;

    let (model3_pairs, model3_prefixes, model3_edges, prefix_to_id) =
        build_model3(chain, token_count)?;
    let starts = build_starts(chain, &prefix_to_id)?;
    let (model2_prefixes, model2_edges) = build_model2(chain, token_count)?;
    let (model1_prefixes, model1_edges) = build_model1(chain, token_count)?;

    Ok(CompiledStorage {
        vocab_offsets,
        vocab_blob,
        starts,
        model3_pairs,
        model3_prefixes,
        model3_edges,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    })
}

fn encode_storage(compiled: &CompiledStorage) -> Result<Vec<u8>, DynError> {
    let header = build_header(compiled)?;
    let mut bytes = write_sections(compiled, header)?;

    let header_bytes = encode_header(header);
    bytes[..HEADER_SIZE].copy_from_slice(header_bytes.as_slice());

    Ok(bytes)
}

fn build_header(compiled: &CompiledStorage) -> Result<Header, DynError> {
    let counts = section_counts(compiled)?;
    let sizes = section_sizes(compiled)?;

    let mut offset = align_to_eight(u64_from_usize(HEADER_SIZE, "header size")?);

    let vocab_offsets_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.vocab_offsets,
        "vocab offsets end",
    )?);

    let vocab_blob_offset = offset;
    offset = align_to_eight(checked_add(offset, sizes.vocab_blob, "vocab blob end")?);

    let start_offset = offset;
    offset = align_to_eight(checked_add(offset, sizes.starts, "start records end")?);

    let model3_pair_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.model3_pairs,
        "model3 pair records end",
    )?);

    let model3_prefix_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.model3_prefixes,
        "model3 prefix records end",
    )?);

    let model3_edge_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.model3_edges,
        "model3 edge records end",
    )?);

    let model2_prefix_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.model2_prefixes,
        "model2 prefix records end",
    )?);

    let model2_edge_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.model2_edges,
        "model2 edge records end",
    )?);

    let model1_prefix_offset = offset;
    offset = align_to_eight(checked_add(
        offset,
        sizes.model1_prefixes,
        "model1 prefix records end",
    )?);

    let model1_edge_offset = offset;
    let file_size = checked_add(offset, sizes.model1_edges, "model1 edge records end")?;

    Ok(Header {
        magic: MAGIC,
        version: VERSION,
        flags: FLAGS,
        tokenizer_version: TOKENIZER_VERSION,
        normalization_flags: NORMALIZATION_FLAGS,
        token_count: counts.token,
        start_count: counts.start,
        model3_pair_count: counts.model3_pair,
        model3_prefix_count: counts.model3_prefix,
        model3_edge_count: counts.model3_edge,
        model2_prefix_count: counts.model2_prefix,
        model2_edge_count: counts.model2_edge,
        model1_prefix_count: counts.model1_prefix,
        model1_edge_count: counts.model1_edge,
        vocab_offsets_offset,
        vocab_blob_offset,
        start_offset,
        model3_pair_offset,
        model3_prefix_offset,
        model3_edge_offset,
        model2_prefix_offset,
        model2_edge_offset,
        model1_prefix_offset,
        model1_edge_offset,
        file_size,
        checksum: CHECKSUM,
    })
}

fn section_counts(compiled: &CompiledStorage) -> Result<SectionCounts, DynError> {
    let token_count = compiled
        .vocab_offsets
        .len()
        .checked_sub(1)
        .ok_or("vocab_offsets must contain at least one entry")?;

    Ok(SectionCounts {
        token: u32_from_usize(token_count, "token count")?,
        start: u32_from_usize(compiled.starts.len(), "start count")?,
        model3_pair: u32_from_usize(compiled.model3_pairs.len(), "model3 pair count")?,
        model3_prefix: u32_from_usize(compiled.model3_prefixes.len(), "model3 prefix count")?,
        model3_edge: u32_from_usize(compiled.model3_edges.len(), "model3 edge count")?,
        model2_prefix: u32_from_usize(compiled.model2_prefixes.len(), "model2 prefix count")?,
        model2_edge: u32_from_usize(compiled.model2_edges.len(), "model2 edge count")?,
        model1_prefix: u32_from_usize(compiled.model1_prefixes.len(), "model1 prefix count")?,
        model1_edge: u32_from_usize(compiled.model1_edges.len(), "model1 edge count")?,
    })
}

fn section_sizes(compiled: &CompiledStorage) -> Result<SectionSizes, DynError> {
    Ok(SectionSizes {
        vocab_offsets: bytes_for_len(compiled.vocab_offsets.len(), 8, "vocab offsets")?,
        vocab_blob: u64_from_usize(compiled.vocab_blob.len(), "vocab blob")?,
        starts: bytes_for_len(compiled.starts.len(), START_RECORD_SIZE, "start records")?,
        model3_pairs: bytes_for_len(
            compiled.model3_pairs.len(),
            PAIR3_RECORD_SIZE,
            "model3 pairs",
        )?,
        model3_prefixes: bytes_for_len(
            compiled.model3_prefixes.len(),
            PREFIX3_RECORD_SIZE,
            "model3 prefixes",
        )?,
        model3_edges: bytes_for_len(
            compiled.model3_edges.len(),
            EDGE_RECORD_SIZE,
            "model3 edges",
        )?,
        model2_prefixes: bytes_for_len(
            compiled.model2_prefixes.len(),
            PREFIX2_RECORD_SIZE,
            "model2 prefixes",
        )?,
        model2_edges: bytes_for_len(
            compiled.model2_edges.len(),
            EDGE_RECORD_SIZE,
            "model2 edges",
        )?,
        model1_prefixes: bytes_for_len(
            compiled.model1_prefixes.len(),
            PREFIX1_RECORD_SIZE,
            "model1 prefixes",
        )?,
        model1_edges: bytes_for_len(
            compiled.model1_edges.len(),
            EDGE_RECORD_SIZE,
            "model1 edges",
        )?,
    })
}

fn write_sections(compiled: &CompiledStorage, header: Header) -> Result<Vec<u8>, DynError> {
    let mut bytes = vec![0; HEADER_SIZE];

    write_at_offset(&mut bytes, header.vocab_offsets_offset, |target| {
        for offset in &compiled.vocab_offsets {
            write_u64(target, *offset);
        }
    })?;

    write_at_offset(&mut bytes, header.vocab_blob_offset, |target| {
        target.extend_from_slice(compiled.vocab_blob.as_slice());
    })?;

    write_at_offset(&mut bytes, header.start_offset, |target| {
        for record in &compiled.starts {
            write_u32(target, record.prefix_id);
            write_u32(target, record.cumulative);
        }
    })?;

    write_at_offset(&mut bytes, header.model3_pair_offset, |target| {
        for record in &compiled.model3_pairs {
            write_u32(target, record.w1);
            write_u32(target, record.w2);
            write_u32(target, record.prefix_start);
            write_u32(target, record.prefix_len);
        }
    })?;

    write_at_offset(&mut bytes, header.model3_prefix_offset, |target| {
        for record in &compiled.model3_prefixes {
            write_u32(target, record.w3);
            write_u32(target, record.edge_start);
            write_u32(target, record.edge_len);
            write_u32(target, record.total);
        }
    })?;

    write_at_offset(&mut bytes, header.model3_edge_offset, |target| {
        for record in &compiled.model3_edges {
            write_u32(target, record.next);
            write_u32(target, record.cumulative);
        }
    })?;

    write_at_offset(&mut bytes, header.model2_prefix_offset, |target| {
        for record in &compiled.model2_prefixes {
            write_u32(target, record.w1);
            write_u32(target, record.w2);
            write_u32(target, record.edge_start);
            write_u32(target, record.edge_len);
            write_u32(target, record.total);
        }
    })?;

    write_at_offset(&mut bytes, header.model2_edge_offset, |target| {
        for record in &compiled.model2_edges {
            write_u32(target, record.next);
            write_u32(target, record.cumulative);
        }
    })?;

    write_at_offset(&mut bytes, header.model1_prefix_offset, |target| {
        for record in &compiled.model1_prefixes {
            write_u32(target, record.w1);
            write_u32(target, record.edge_start);
            write_u32(target, record.edge_len);
            write_u32(target, record.total);
        }
    })?;

    write_at_offset(&mut bytes, header.model1_edge_offset, |target| {
        for record in &compiled.model1_edges {
            write_u32(target, record.next);
            write_u32(target, record.cumulative);
        }
    })?;

    let expected_file_size = usize_from_u64(header.file_size, "file size")?;
    if bytes.len() != expected_file_size {
        return Err(format!(
            "serialized size mismatch: expected {expected_file_size}, got {}",
            bytes.len()
        )
        .into());
    }

    Ok(bytes)
}

fn write_at_offset<F>(target: &mut Vec<u8>, offset: u64, writer: F) -> Result<(), DynError>
where
    F: FnOnce(&mut Vec<u8>),
{
    pad_to_offset(target, offset)?;
    writer(target);
    Ok(())
}

fn decode_chain(bytes: &[u8]) -> Result<MarkovChain, DynError> {
    let header = validate_header(bytes)?;
    let ranges = build_section_ranges(&header)?;
    let parsed = parse_storage(bytes, &header, &ranges)?;

    rebuild_chain(parsed)
}

fn validate_header(bytes: &[u8]) -> Result<Header, DynError> {
    let header = decode_header(bytes)?;

    if header.magic != MAGIC {
        return Err("invalid magic".into());
    }
    if header.version != VERSION {
        return Err(format!("unsupported version: {}", header.version).into());
    }
    if header.flags != FLAGS {
        return Err(format!("unsupported flags: {}", header.flags).into());
    }

    let file_size = u64_from_usize(bytes.len(), "file size")?;
    if header.file_size != file_size {
        return Err(format!(
            "file_size mismatch: header={}, actual={file_size}",
            header.file_size
        )
        .into());
    }

    let ordered_offsets = [
        header.vocab_offsets_offset,
        header.vocab_blob_offset,
        header.start_offset,
        header.model3_pair_offset,
        header.model3_prefix_offset,
        header.model3_edge_offset,
        header.model2_prefix_offset,
        header.model2_edge_offset,
        header.model1_prefix_offset,
        header.model1_edge_offset,
    ];

    if ordered_offsets
        .windows(2)
        .any(|window| window[0] > window[1])
    {
        return Err("section offsets are not ordered".into());
    }

    Ok(header)
}

fn build_section_ranges(header: &Header) -> Result<SectionRanges, DynError> {
    let vocab_offsets_len = usize_from_u32(
        header
            .token_count
            .checked_add(1)
            .ok_or("token_count overflow")?,
        "vocab offsets length",
    )?;

    let vocab_offsets = fixed_size_range(
        header.vocab_offsets_offset,
        vocab_offsets_len,
        8,
        header.file_size,
        "vocab offsets",
    )?;

    let starts = record_range(
        header.start_offset,
        header.start_count,
        START_RECORD_SIZE,
        header.file_size,
        "start records",
    )?;
    let model3_pairs = record_range(
        header.model3_pair_offset,
        header.model3_pair_count,
        PAIR3_RECORD_SIZE,
        header.file_size,
        "model3 pairs",
    )?;
    let model3_prefixes = record_range(
        header.model3_prefix_offset,
        header.model3_prefix_count,
        PREFIX3_RECORD_SIZE,
        header.file_size,
        "model3 prefixes",
    )?;
    let model3_edges = record_range(
        header.model3_edge_offset,
        header.model3_edge_count,
        EDGE_RECORD_SIZE,
        header.file_size,
        "model3 edges",
    )?;
    let model2_prefixes = record_range(
        header.model2_prefix_offset,
        header.model2_prefix_count,
        PREFIX2_RECORD_SIZE,
        header.file_size,
        "model2 prefixes",
    )?;
    let model2_edges = record_range(
        header.model2_edge_offset,
        header.model2_edge_count,
        EDGE_RECORD_SIZE,
        header.file_size,
        "model2 edges",
    )?;
    let model1_prefixes = record_range(
        header.model1_prefix_offset,
        header.model1_prefix_count,
        PREFIX1_RECORD_SIZE,
        header.file_size,
        "model1 prefixes",
    )?;
    let model1_edges = record_range(
        header.model1_edge_offset,
        header.model1_edge_count,
        EDGE_RECORD_SIZE,
        header.file_size,
        "model1 edges",
    )?;

    let vocab_blob_area_size = header
        .start_offset
        .checked_sub(header.vocab_blob_offset)
        .ok_or("vocab blob offset exceeds start offset")?;
    let vocab_blob_area = section_range(
        header.vocab_blob_offset,
        vocab_blob_area_size,
        header.file_size,
        "vocab blob area",
    )?;

    Ok(SectionRanges {
        vocab_offsets,
        vocab_blob_area,
        starts,
        model3_pairs,
        model3_prefixes,
        model3_edges,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    })
}

fn parse_storage(
    bytes: &[u8],
    header: &Header,
    ranges: &SectionRanges,
) -> Result<ParsedStorage, DynError> {
    let vocab_offsets_len = usize_from_u32(
        header
            .token_count
            .checked_add(1)
            .ok_or("token_count overflow")?,
        "vocab offsets length",
    )?;

    let vocab_offsets = parse_u64_values(
        bytes[ranges.vocab_offsets.clone()].as_ref(),
        vocab_offsets_len,
        "vocab offsets",
    )?;
    validate_vocab_offsets(vocab_offsets.as_slice())?;

    let vocab_blob_size = *vocab_offsets.last().ok_or("vocab offsets are empty")?;
    let vocab_blob_area_size = u64_from_usize(ranges.vocab_blob_area.len(), "vocab blob area")?;
    if vocab_blob_size > vocab_blob_area_size {
        return Err("vocab blob size exceeds allocated section".into());
    }

    let vocab_blob_end = ranges
        .vocab_blob_area
        .start
        .checked_add(usize_from_u64(vocab_blob_size, "vocab blob size")?)
        .ok_or("vocab blob range overflow")?;
    let vocab_blob = bytes
        .get(ranges.vocab_blob_area.start..vocab_blob_end)
        .ok_or("vocab blob range is invalid")?;

    let id_to_token = decode_vocab(vocab_offsets.as_slice(), vocab_blob)?;
    validate_special_tokens(id_to_token.as_slice())?;

    Ok(ParsedStorage {
        id_to_token,
        starts: parse_start_records(
            bytes[ranges.starts.clone()].as_ref(),
            usize_from_u32(header.start_count, "start count")?,
        )?,
        model3_pairs: parse_pair3_records(
            bytes[ranges.model3_pairs.clone()].as_ref(),
            usize_from_u32(header.model3_pair_count, "model3 pair count")?,
        )?,
        model3_prefixes: parse_prefix3_records(
            bytes[ranges.model3_prefixes.clone()].as_ref(),
            usize_from_u32(header.model3_prefix_count, "model3 prefix count")?,
        )?,
        model3_edges: parse_edge_records(
            bytes[ranges.model3_edges.clone()].as_ref(),
            usize_from_u32(header.model3_edge_count, "model3 edge count")?,
        )?,
        model2_prefixes: parse_prefix2_records(
            bytes[ranges.model2_prefixes.clone()].as_ref(),
            usize_from_u32(header.model2_prefix_count, "model2 prefix count")?,
        )?,
        model2_edges: parse_edge_records(
            bytes[ranges.model2_edges.clone()].as_ref(),
            usize_from_u32(header.model2_edge_count, "model2 edge count")?,
        )?,
        model1_prefixes: parse_prefix1_records(
            bytes[ranges.model1_prefixes.clone()].as_ref(),
            usize_from_u32(header.model1_prefix_count, "model1 prefix count")?,
        )?,
        model1_edges: parse_edge_records(
            bytes[ranges.model1_edges.clone()].as_ref(),
            usize_from_u32(header.model1_edge_count, "model1 edge count")?,
        )?,
    })
}

fn rebuild_chain(parsed: ParsedStorage) -> Result<MarkovChain, DynError> {
    let token_count = u32_from_usize(parsed.id_to_token.len(), "token count")?;

    let model3_keys = validate_and_build_model3_keys(
        parsed.model3_pairs.as_slice(),
        parsed.model3_prefixes.as_slice(),
        parsed.model3_edges.as_slice(),
        token_count,
    )?;
    validate_model2(
        parsed.model2_prefixes.as_slice(),
        parsed.model2_edges.as_slice(),
        token_count,
    )?;
    validate_model1(
        parsed.model1_prefixes.as_slice(),
        parsed.model1_edges.as_slice(),
        token_count,
    )?;
    validate_starts(parsed.starts.as_slice(), model3_keys.len())?;

    let token_to_id = build_token_index(parsed.id_to_token.as_slice())?;
    let starts = decode_starts(parsed.starts.as_slice(), model3_keys.as_slice())?;
    let model3 = decode_model3(
        model3_keys.as_slice(),
        parsed.model3_prefixes.as_slice(),
        parsed.model3_edges.as_slice(),
    )?;
    let model2 = decode_model2(
        parsed.model2_prefixes.as_slice(),
        parsed.model2_edges.as_slice(),
    )?;
    let model1 = decode_model1(
        parsed.model1_prefixes.as_slice(),
        parsed.model1_edges.as_slice(),
    )?;

    Ok(MarkovChain {
        token_to_id,
        id_to_token: parsed.id_to_token,
        model3,
        model2,
        model1,
        starts,
    })
}

fn record_range(
    offset: u64,
    count: u32,
    record_size: u64,
    file_size: u64,
    context: &str,
) -> Result<std::ops::Range<usize>, DynError> {
    let len = usize_from_u32(count, context)?;
    fixed_size_range(offset, len, record_size, file_size, context)
}

fn fixed_size_range(
    offset: u64,
    len: usize,
    element_size: u64,
    file_size: u64,
    context: &str,
) -> Result<std::ops::Range<usize>, DynError> {
    let size = bytes_for_len(len, element_size, context)?;
    section_range(offset, size, file_size, context)
}

fn validate_token_index(chain: &MarkovChain) -> Result<(), DynError> {
    for (index, token) in chain.id_to_token.iter().enumerate() {
        let token_id = u32_from_usize(index, "token index")?;

        let Some(stored_id) = chain.token_to_id.get(token).copied() else {
            return Err(format!("token '{token}' is missing in token_to_id").into());
        };

        if stored_id != token_id {
            return Err(format!("token '{token}' index mismatch").into());
        }
    }

    Ok(())
}

fn build_vocab(tokens: &[String]) -> Result<(Vec<u64>, Vec<u8>), DynError> {
    let mut offsets = Vec::with_capacity(tokens.len().saturating_add(1));
    let mut blob = Vec::new();
    let mut position = 0_u64;

    offsets.push(0);

    for token in tokens {
        let token_bytes = token.as_bytes();
        blob.extend_from_slice(token_bytes);

        let token_len = u64_from_usize(token_bytes.len(), "token byte length")?;
        position = checked_add(position, token_len, "vocab blob size")?;
        offsets.push(position);
    }

    Ok((offsets, blob))
}

fn build_model3(chain: &MarkovChain, token_count: u32) -> Result<Model3Build, DynError> {
    let mut entries = chain
        .model3
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();
    let mut prefix_to_id = HashMap::new();

    let mut index = 0_usize;
    while index < entries.len() {
        let [w1, w2, _] = entries[index].0;
        validate_token_id(w1, token_count, "model3 pair.w1")?;
        validate_token_id(w2, token_count, "model3 pair.w2")?;

        let prefix_start = u32_from_usize(prefix_records.len(), "model3 prefix start")?;
        let mut prefix_len = 0_u32;

        while index < entries.len() && entries[index].0[0] == w1 && entries[index].0[1] == w2 {
            let prefix = entries[index].0;
            let w3 = prefix[2];
            validate_token_id(w3, token_count, "model3 prefix.w3")?;

            let prefix_id = u32_from_usize(prefix_records.len(), "model3 prefix id")?;
            prefix_to_id.insert(prefix, prefix_id);

            let (edge_start, edge_len, total) = append_edges(
                entries[index].1,
                &mut edge_records,
                token_count,
                "model3 edges",
            )?;

            prefix_records.push(Prefix3Record {
                w3,
                edge_start,
                edge_len,
                total,
            });

            prefix_len = prefix_len
                .checked_add(1)
                .ok_or("model3 prefix length overflow")?;
            index += 1;
        }

        pair_records.push(Pair3Record {
            w1,
            w2,
            prefix_start,
            prefix_len,
        });
    }

    Ok((pair_records, prefix_records, edge_records, prefix_to_id))
}

fn build_model2(
    chain: &MarkovChain,
    token_count: u32,
) -> Result<(Vec<Prefix2Record>, Vec<EdgeRecord>), DynError> {
    let mut entries = chain
        .model2
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for (prefix, edges) in entries {
        validate_token_id(prefix[0], token_count, "model2 prefix.w1")?;
        validate_token_id(prefix[1], token_count, "model2 prefix.w2")?;

        let (edge_start, edge_len, total) =
            append_edges(edges, &mut edge_records, token_count, "model2 edges")?;

        prefix_records.push(Prefix2Record {
            w1: prefix[0],
            w2: prefix[1],
            edge_start,
            edge_len,
            total,
        });
    }

    Ok((prefix_records, edge_records))
}

fn build_model1(
    chain: &MarkovChain,
    token_count: u32,
) -> Result<(Vec<Prefix1Record>, Vec<EdgeRecord>), DynError> {
    let mut entries = chain
        .model1
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for (prefix, edges) in entries {
        validate_token_id(prefix, token_count, "model1 prefix.w1")?;

        let (edge_start, edge_len, total) =
            append_edges(edges, &mut edge_records, token_count, "model1 edges")?;

        prefix_records.push(Prefix1Record {
            w1: prefix,
            edge_start,
            edge_len,
            total,
        });
    }

    Ok((prefix_records, edge_records))
}

fn build_starts(
    chain: &MarkovChain,
    prefix_to_id: &HashMap<[TokenId; 3], u32>,
) -> Result<Vec<StartRecord>, DynError> {
    let mut entries = chain
        .starts
        .iter()
        .map(|(prefix, count)| (*prefix, *count))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut records = Vec::new();
    let mut cumulative = 0_u32;

    for (prefix, count) in entries {
        if count == 0 {
            continue;
        }

        let Some(prefix_id) = prefix_to_id.get(&prefix).copied() else {
            return Err("start prefix is missing from model3 prefixes".into());
        };

        let count = u32::try_from(count).map_err(|_| "start count exceeds u32 range")?;
        cumulative = cumulative
            .checked_add(count)
            .ok_or("start cumulative overflow")?;

        records.push(StartRecord {
            prefix_id,
            cumulative,
        });
    }

    Ok(records)
}

fn append_edges(
    source: &HashMap<TokenId, Count>,
    edges: &mut Vec<EdgeRecord>,
    token_count: u32,
    context: &str,
) -> Result<(u32, u32, u32), DynError> {
    let edge_start = u32_from_usize(edges.len(), "edge start")?;

    let mut sorted_edges = source
        .iter()
        .map(|(next, count)| (*next, *count))
        .collect::<Vec<_>>();
    sorted_edges.sort_unstable_by_key(|(next, _)| *next);

    let mut cumulative = 0_u32;

    for (next, count) in sorted_edges {
        if count == 0 {
            continue;
        }

        validate_token_id(next, token_count, context)?;

        let weight =
            u32::try_from(count).map_err(|_| format!("{context} count exceeds u32 range"))?;
        cumulative = cumulative
            .checked_add(weight)
            .ok_or_else(|| format!("{context} cumulative overflow"))?;

        edges.push(EdgeRecord { next, cumulative });
    }

    let edge_end = u32_from_usize(edges.len(), "edge end")?;
    let edge_len = edge_end
        .checked_sub(edge_start)
        .ok_or("edge length underflow")?;

    Ok((edge_start, edge_len, cumulative))
}

fn encode_header(header: Header) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_SIZE);

    bytes.extend_from_slice(header.magic.as_slice());
    write_u32(&mut bytes, header.version);
    write_u32(&mut bytes, header.flags);
    write_u32(&mut bytes, header.tokenizer_version);
    write_u32(&mut bytes, header.normalization_flags);
    write_u32(&mut bytes, header.token_count);
    write_u32(&mut bytes, header.start_count);
    write_u32(&mut bytes, header.model3_pair_count);
    write_u32(&mut bytes, header.model3_prefix_count);
    write_u32(&mut bytes, header.model3_edge_count);
    write_u32(&mut bytes, header.model2_prefix_count);
    write_u32(&mut bytes, header.model2_edge_count);
    write_u32(&mut bytes, header.model1_prefix_count);
    write_u32(&mut bytes, header.model1_edge_count);
    write_u64(&mut bytes, header.vocab_offsets_offset);
    write_u64(&mut bytes, header.vocab_blob_offset);
    write_u64(&mut bytes, header.start_offset);
    write_u64(&mut bytes, header.model3_pair_offset);
    write_u64(&mut bytes, header.model3_prefix_offset);
    write_u64(&mut bytes, header.model3_edge_offset);
    write_u64(&mut bytes, header.model2_prefix_offset);
    write_u64(&mut bytes, header.model2_edge_offset);
    write_u64(&mut bytes, header.model1_prefix_offset);
    write_u64(&mut bytes, header.model1_edge_offset);
    write_u64(&mut bytes, header.file_size);
    write_u64(&mut bytes, header.checksum);

    debug_assert_eq!(bytes.len(), HEADER_SIZE);
    bytes
}

fn decode_header(bytes: &[u8]) -> Result<Header, DynError> {
    if bytes.len() < HEADER_SIZE {
        return Err("header is too short".into());
    }

    let mut cursor = 0_usize;
    let magic = {
        let mut value = [0_u8; 8];
        value.copy_from_slice(read_exact(bytes, &mut cursor, 8)?);
        value
    };

    let version = read_u32_value(bytes, &mut cursor)?;
    let flags = read_u32_value(bytes, &mut cursor)?;
    let tokenizer_version = read_u32_value(bytes, &mut cursor)?;
    let normalization_flags = read_u32_value(bytes, &mut cursor)?;
    let token_count = read_u32_value(bytes, &mut cursor)?;
    let start_count = read_u32_value(bytes, &mut cursor)?;
    let model3_pair_count = read_u32_value(bytes, &mut cursor)?;
    let model3_prefix_count = read_u32_value(bytes, &mut cursor)?;
    let model3_edge_count = read_u32_value(bytes, &mut cursor)?;
    let model2_prefix_count = read_u32_value(bytes, &mut cursor)?;
    let model2_edge_count = read_u32_value(bytes, &mut cursor)?;
    let model1_prefix_count = read_u32_value(bytes, &mut cursor)?;
    let model1_edge_count = read_u32_value(bytes, &mut cursor)?;
    let vocab_offsets_offset = read_u64_value(bytes, &mut cursor)?;
    let vocab_blob_offset = read_u64_value(bytes, &mut cursor)?;
    let start_offset = read_u64_value(bytes, &mut cursor)?;
    let model3_pair_offset = read_u64_value(bytes, &mut cursor)?;
    let model3_prefix_offset = read_u64_value(bytes, &mut cursor)?;
    let model3_edge_offset = read_u64_value(bytes, &mut cursor)?;
    let model2_prefix_offset = read_u64_value(bytes, &mut cursor)?;
    let model2_edge_offset = read_u64_value(bytes, &mut cursor)?;
    let model1_prefix_offset = read_u64_value(bytes, &mut cursor)?;
    let model1_edge_offset = read_u64_value(bytes, &mut cursor)?;
    let file_size = read_u64_value(bytes, &mut cursor)?;
    let checksum = read_u64_value(bytes, &mut cursor)?;

    Ok(Header {
        magic,
        version,
        flags,
        tokenizer_version,
        normalization_flags,
        token_count,
        start_count,
        model3_pair_count,
        model3_prefix_count,
        model3_edge_count,
        model2_prefix_count,
        model2_edge_count,
        model1_prefix_count,
        model1_edge_count,
        vocab_offsets_offset,
        vocab_blob_offset,
        start_offset,
        model3_pair_offset,
        model3_prefix_offset,
        model3_edge_offset,
        model2_prefix_offset,
        model2_edge_offset,
        model1_prefix_offset,
        model1_edge_offset,
        file_size,
        checksum,
    })
}

fn parse_u64_values(bytes: &[u8], count: usize, context: &str) -> Result<Vec<u64>, DynError> {
    let expected = bytes_for_len(count, 8, context)?;
    let expected = usize_from_u64(expected, context)?;
    if bytes.len() != expected {
        return Err(format!(
            "{context} length mismatch: expected {expected}, got {}",
            bytes.len()
        )
        .into());
    }

    let mut values = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        values.push(read_u64_value(bytes, &mut cursor)?);
    }

    Ok(values)
}

fn parse_start_records(bytes: &[u8], count: usize) -> Result<Vec<StartRecord>, DynError> {
    let expected = bytes_for_len(count, START_RECORD_SIZE, "start records")?;
    let expected = usize_from_u64(expected, "start records")?;
    if bytes.len() != expected {
        return Err("start records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(StartRecord {
            prefix_id: read_u32_value(bytes, &mut cursor)?,
            cumulative: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

fn parse_pair3_records(bytes: &[u8], count: usize) -> Result<Vec<Pair3Record>, DynError> {
    let expected = bytes_for_len(count, PAIR3_RECORD_SIZE, "model3 pair records")?;
    let expected = usize_from_u64(expected, "model3 pair records")?;
    if bytes.len() != expected {
        return Err("model3 pair records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Pair3Record {
            w1: read_u32_value(bytes, &mut cursor)?,
            w2: read_u32_value(bytes, &mut cursor)?,
            prefix_start: read_u32_value(bytes, &mut cursor)?,
            prefix_len: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

fn parse_prefix3_records(bytes: &[u8], count: usize) -> Result<Vec<Prefix3Record>, DynError> {
    let expected = bytes_for_len(count, PREFIX3_RECORD_SIZE, "model3 prefix records")?;
    let expected = usize_from_u64(expected, "model3 prefix records")?;
    if bytes.len() != expected {
        return Err("model3 prefix records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Prefix3Record {
            w3: read_u32_value(bytes, &mut cursor)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

fn parse_prefix2_records(bytes: &[u8], count: usize) -> Result<Vec<Prefix2Record>, DynError> {
    let expected = bytes_for_len(count, PREFIX2_RECORD_SIZE, "model2 prefix records")?;
    let expected = usize_from_u64(expected, "model2 prefix records")?;
    if bytes.len() != expected {
        return Err("model2 prefix records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Prefix2Record {
            w1: read_u32_value(bytes, &mut cursor)?,
            w2: read_u32_value(bytes, &mut cursor)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

fn parse_prefix1_records(bytes: &[u8], count: usize) -> Result<Vec<Prefix1Record>, DynError> {
    let expected = bytes_for_len(count, PREFIX1_RECORD_SIZE, "model1 prefix records")?;
    let expected = usize_from_u64(expected, "model1 prefix records")?;
    if bytes.len() != expected {
        return Err("model1 prefix records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Prefix1Record {
            w1: read_u32_value(bytes, &mut cursor)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

fn parse_edge_records(bytes: &[u8], count: usize) -> Result<Vec<EdgeRecord>, DynError> {
    let expected = bytes_for_len(count, EDGE_RECORD_SIZE, "edge records")?;
    let expected = usize_from_u64(expected, "edge records")?;
    if bytes.len() != expected {
        return Err("edge records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(EdgeRecord {
            next: read_u32_value(bytes, &mut cursor)?,
            cumulative: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

fn decode_vocab(offsets: &[u64], blob: &[u8]) -> Result<Vec<String>, DynError> {
    if offsets.is_empty() {
        return Err("vocab offsets are empty".into());
    }

    let mut tokens = Vec::with_capacity(offsets.len().saturating_sub(1));
    for pair in offsets.windows(2) {
        let start = usize_from_u64(pair[0], "vocab token start")?;
        let end = usize_from_u64(pair[1], "vocab token end")?;
        let token_bytes = blob.get(start..end).ok_or("vocab token range is invalid")?;
        let token = str::from_utf8(token_bytes)
            .map_err(|_| "vocab token is not valid UTF-8")?
            .to_owned();
        tokens.push(token);
    }

    Ok(tokens)
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

fn validate_vocab_offsets(offsets: &[u64]) -> Result<(), DynError> {
    if offsets.first().copied() != Some(0) {
        return Err("vocab offsets must start with 0".into());
    }

    for pair in offsets.windows(2) {
        if pair[0] > pair[1] {
            return Err("vocab offsets must be non-decreasing".into());
        }
    }

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

fn validate_and_build_model3_keys(
    pairs: &[Pair3Record],
    prefixes: &[Prefix3Record],
    edges: &[EdgeRecord],
    token_count: u32,
) -> Result<Vec<[TokenId; 3]>, DynError> {
    let mut full_prefixes = vec![[0_u32; 3]; prefixes.len()];
    let mut assigned = vec![false; prefixes.len()];
    let mut previous_pair = None;

    for pair in pairs {
        validate_token_id(pair.w1, token_count, "model3 pair.w1")?;
        validate_token_id(pair.w2, token_count, "model3 pair.w2")?;

        let current_pair = (pair.w1, pair.w2);
        if let Some(previous) = previous_pair
            && current_pair <= previous
        {
            return Err("model3 pair records are not strictly sorted".into());
        }
        previous_pair = Some(current_pair);

        let prefix_start = usize_from_u32(pair.prefix_start, "model3 prefix start")?;
        let prefix_len = usize_from_u32(pair.prefix_len, "model3 prefix len")?;
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or("model3 prefix range overflow")?;

        if prefix_end > prefixes.len() {
            return Err("model3 pair prefix range is out of bounds".into());
        }

        let mut previous_w3 = None;
        for index in prefix_start..prefix_end {
            if assigned[index] {
                return Err("model3 pair prefix ranges overlap".into());
            }

            let prefix = prefixes[index];
            validate_token_id(prefix.w3, token_count, "model3 prefix.w3")?;

            if let Some(previous) = previous_w3
                && prefix.w3 <= previous
            {
                return Err("model3 prefix records are not sorted by w3".into());
            }
            previous_w3 = Some(prefix.w3);

            validate_prefix_edges(
                edges,
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model3 prefix",
            )?;

            full_prefixes[index] = [pair.w1, pair.w2, prefix.w3];
            assigned[index] = true;
        }
    }

    if assigned.iter().any(|is_assigned| !*is_assigned) {
        return Err("some model3 prefixes are not covered by pair records".into());
    }

    Ok(full_prefixes)
}

fn validate_model2(
    prefixes: &[Prefix2Record],
    edges: &[EdgeRecord],
    token_count: u32,
) -> Result<(), DynError> {
    let mut previous_key = None;

    for prefix in prefixes {
        validate_token_id(prefix.w1, token_count, "model2 prefix.w1")?;
        validate_token_id(prefix.w2, token_count, "model2 prefix.w2")?;

        let key = (prefix.w1, prefix.w2);
        if let Some(previous) = previous_key
            && key <= previous
        {
            return Err("model2 prefix records are not strictly sorted".into());
        }
        previous_key = Some(key);

        validate_prefix_edges(
            edges,
            prefix.edge_start,
            prefix.edge_len,
            prefix.total,
            token_count,
            "model2 prefix",
        )?;
    }

    Ok(())
}

fn validate_model1(
    prefixes: &[Prefix1Record],
    edges: &[EdgeRecord],
    token_count: u32,
) -> Result<(), DynError> {
    let mut previous_w1 = None;

    for prefix in prefixes {
        validate_token_id(prefix.w1, token_count, "model1 prefix.w1")?;

        if let Some(previous) = previous_w1
            && prefix.w1 <= previous
        {
            return Err("model1 prefix records are not strictly sorted".into());
        }
        previous_w1 = Some(prefix.w1);

        validate_prefix_edges(
            edges,
            prefix.edge_start,
            prefix.edge_len,
            prefix.total,
            token_count,
            "model1 prefix",
        )?;
    }

    Ok(())
}

fn validate_starts(starts: &[StartRecord], model3_prefix_count: usize) -> Result<(), DynError> {
    let mut previous_cumulative = 0_u32;

    for record in starts {
        let prefix_id = usize_from_u32(record.prefix_id, "start prefix_id")?;
        if prefix_id >= model3_prefix_count {
            return Err("start prefix_id is out of range".into());
        }

        if record.cumulative <= previous_cumulative {
            return Err("start cumulative must be strictly increasing".into());
        }

        previous_cumulative = record.cumulative;
    }

    Ok(())
}

fn validate_prefix_edges(
    edges: &[EdgeRecord],
    edge_start: u32,
    edge_len: u32,
    total: u32,
    token_count: u32,
    context: &str,
) -> Result<(), DynError> {
    let start = usize_from_u32(edge_start, "edge_start")?;
    let len = usize_from_u32(edge_len, "edge_len")?;
    let end = start.checked_add(len).ok_or("edge range overflow")?;

    if end > edges.len() {
        return Err(format!("{context} edge range is out of bounds").into());
    }

    if edge_len == 0 {
        if total != 0 {
            return Err(format!("{context} total must be zero when edge_len is zero").into());
        }

        return Ok(());
    }

    let edge_slice = &edges[start..end];
    let mut previous_next = None;
    let mut previous_cumulative = 0_u32;

    for edge in edge_slice {
        validate_token_id(edge.next, token_count, context)?;

        if let Some(previous) = previous_next
            && edge.next <= previous
        {
            return Err(format!("{context} edges are not sorted by next").into());
        }
        previous_next = Some(edge.next);

        if edge.cumulative <= previous_cumulative {
            return Err(format!("{context} cumulative must be strictly increasing").into());
        }
        previous_cumulative = edge.cumulative;
    }

    if previous_cumulative != total {
        return Err(format!("{context} total does not match last cumulative").into());
    }

    Ok(())
}

fn decode_starts(
    starts: &[StartRecord],
    model3_keys: &[[TokenId; 3]],
) -> Result<HashMap<[TokenId; 3], Count>, DynError> {
    let mut decoded = HashMap::new();
    let mut previous = 0_u32;

    for record in starts {
        let delta = record
            .cumulative
            .checked_sub(previous)
            .ok_or("start cumulative underflow")?;
        previous = record.cumulative;

        let prefix_index = usize_from_u32(record.prefix_id, "start prefix_id")?;
        let prefix = *model3_keys
            .get(prefix_index)
            .ok_or("start prefix_id is out of bounds")?;

        let entry = decoded.entry(prefix).or_insert(0_u64);
        *entry = (*entry).saturating_add(u64::from(delta));
    }

    Ok(decoded)
}

fn decode_model3(
    model3_keys: &[[TokenId; 3]],
    prefixes: &[Prefix3Record],
    edges: &[EdgeRecord],
) -> Result<HashMap<[TokenId; 3], HashMap<TokenId, Count>>, DynError> {
    let mut decoded = HashMap::new();

    for (index, prefix) in prefixes.iter().enumerate() {
        let key = *model3_keys
            .get(index)
            .ok_or("model3 prefix index is out of bounds")?;
        let edge_map = decode_edge_map(edges, prefix.edge_start, prefix.edge_len)?;
        decoded.insert(key, edge_map);
    }

    Ok(decoded)
}

fn decode_model2(
    prefixes: &[Prefix2Record],
    edges: &[EdgeRecord],
) -> Result<HashMap<[TokenId; 2], HashMap<TokenId, Count>>, DynError> {
    let mut decoded = HashMap::new();

    for prefix in prefixes {
        let key = [prefix.w1, prefix.w2];
        let edge_map = decode_edge_map(edges, prefix.edge_start, prefix.edge_len)?;
        decoded.insert(key, edge_map);
    }

    Ok(decoded)
}

fn decode_model1(
    prefixes: &[Prefix1Record],
    edges: &[EdgeRecord],
) -> Result<HashMap<TokenId, HashMap<TokenId, Count>>, DynError> {
    let mut decoded = HashMap::new();

    for prefix in prefixes {
        let edge_map = decode_edge_map(edges, prefix.edge_start, prefix.edge_len)?;
        decoded.insert(prefix.w1, edge_map);
    }

    Ok(decoded)
}

fn decode_edge_map(
    edges: &[EdgeRecord],
    edge_start: u32,
    edge_len: u32,
) -> Result<HashMap<TokenId, Count>, DynError> {
    let start = usize_from_u32(edge_start, "edge_start")?;
    let len = usize_from_u32(edge_len, "edge_len")?;
    let end = start.checked_add(len).ok_or("edge range overflow")?;
    let edge_slice = edges.get(start..end).ok_or("edge range is out of bounds")?;

    let mut map = HashMap::new();
    let mut previous = 0_u32;

    for edge in edge_slice {
        let delta = edge
            .cumulative
            .checked_sub(previous)
            .ok_or("edge cumulative underflow")?;
        previous = edge.cumulative;
        map.insert(edge.next, u64::from(delta));
    }

    Ok(map)
}

fn validate_token_id(token_id: u32, token_count: u32, context: &str) -> Result<(), DynError> {
    if token_id >= token_count {
        return Err(format!("{context}: token id {token_id} is out of range").into());
    }

    Ok(())
}

fn section_range(
    offset: u64,
    size: u64,
    file_size: u64,
    context: &str,
) -> Result<std::ops::Range<usize>, DynError> {
    let end = checked_add(offset, size, context)?;
    if end > file_size {
        return Err(format!("{context} exceeds file_size").into());
    }

    Ok(usize_from_u64(offset, context)?..usize_from_u64(end, context)?)
}

fn read_exact<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8], DynError> {
    let end = cursor.checked_add(count).ok_or("cursor overflow")?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or("unexpected EOF while reading")?;
    *cursor = end;
    Ok(slice)
}

fn read_u32_value(bytes: &[u8], cursor: &mut usize) -> Result<u32, DynError> {
    let raw = read_exact(bytes, cursor, 4)?;
    let mut array = [0_u8; 4];
    array.copy_from_slice(raw);
    Ok(u32::from_le_bytes(array))
}

fn read_u64_value(bytes: &[u8], cursor: &mut usize) -> Result<u64, DynError> {
    let raw = read_exact(bytes, cursor, 8)?;
    let mut array = [0_u8; 8];
    array.copy_from_slice(raw);
    Ok(u64::from_le_bytes(array))
}

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn pad_to_offset(target: &mut Vec<u8>, offset: u64) -> Result<(), DynError> {
    let offset = usize_from_u64(offset, "offset")?;

    if target.len() > offset {
        return Err(format!(
            "offset regression: current={}, target={offset}",
            target.len()
        )
        .into());
    }

    if target.len() < offset {
        target.resize(offset, 0);
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use rand::{SeedableRng, rngs::StdRng};
    use tokio::fs;

    use crate::markov::MarkovChain;

    use super::{load_chain, save_chain};

    #[tokio::test]
    async fn load_returns_default_for_missing_file() {
        let file_path = temp_file_path("missing");
        let chain = load_chain(&file_path).await.expect("load should succeed");

        assert!(chain.starts.is_empty());
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        let file_path = temp_file_path("roundtrip");
        let mut chain = MarkovChain::default();
        chain
            .train_tokens(&[
                "a".to_owned(),
                "b".to_owned(),
                "c".to_owned(),
                "d".to_owned(),
            ])
            .expect("training should succeed");
        chain
            .train_tokens(&["a".to_owned()])
            .expect("training should succeed");

        save_chain(&file_path, &chain)
            .await
            .expect("save should succeed");
        let loaded = load_chain(&file_path).await.expect("load should succeed");

        assert!(!loaded.starts.is_empty());

        let mut left_rng = StdRng::seed_from_u64(7);
        let mut right_rng = StdRng::seed_from_u64(7);
        let left = chain.generate_sentence(&mut left_rng, 10);
        let right = loaded.generate_sentence(&mut right_rng, 10);

        assert_eq!(left, right);

        let _ = fs::remove_file(file_path).await;
    }

    fn temp_file_path(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();

        std::env::temp_dir().join(format!("markov_bot_{prefix}_{nanos}.mkv3"))
    }
}
