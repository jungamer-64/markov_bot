use std::str;

use super::super::{
    DynError, EDGE_RECORD_SIZE, EdgeRecord, Header, PAIR3_RECORD_SIZE, PREFIX1_RECORD_SIZE,
    PREFIX2_RECORD_SIZE, PREFIX3_RECORD_SIZE, Pair3Record, ParsedStorage, Prefix1Record,
    Prefix2Record, Prefix3Record, START_RECORD_SIZE, SectionRanges, StartRecord, bytes_for_len,
    u64_from_usize, usize_from_u32, usize_from_u64, validate_special_tokens,
};
use super::{read_u32_value, read_u64_value};

pub(super) fn parse_storage(
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
