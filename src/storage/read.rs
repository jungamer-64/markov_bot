use std::{collections::HashMap, str};

use super::{
    CHECKSUM, Count, DynError, EDGE_RECORD_SIZE, EdgeRecord, FLAGS, HEADER_SIZE, Header, MAGIC,
    MarkovChain, NORMALIZATION_FLAGS, PAIR3_RECORD_SIZE, PREFIX1_RECORD_SIZE, PREFIX2_RECORD_SIZE,
    PREFIX3_RECORD_SIZE, Pair3Record, ParsedStorage, Prefix1Record, Prefix2Record, Prefix3Record,
    START_RECORD_SIZE, SectionRanges, StartRecord, TOKENIZER_VERSION, TokenId, VERSION,
    bytes_for_len, checked_add, u32_from_usize, u64_from_usize, usize_from_u32, usize_from_u64,
    validate_special_tokens, validate_token_id,
};

pub(super) fn decode_chain(bytes: &[u8]) -> Result<MarkovChain, DynError> {
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
    if header.tokenizer_version != TOKENIZER_VERSION {
        return Err(format!(
            "unsupported tokenizer_version: {}",
            header.tokenizer_version
        )
        .into());
    }
    if header.normalization_flags != NORMALIZATION_FLAGS {
        return Err(format!(
            "unsupported normalization_flags: {}",
            header.normalization_flags
        )
        .into());
    }
    if header.checksum != CHECKSUM {
        return Err(format!("unsupported checksum value: {}", header.checksum).into());
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

    let ranges = SectionRanges {
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
    };

    validate_section_layout(header, &ranges)?;

    Ok(ranges)
}

fn validate_section_layout(header: &Header, ranges: &SectionRanges) -> Result<(), DynError> {
    ensure_non_overlapping(
        "vocab offsets",
        &ranges.vocab_offsets,
        "vocab blob",
        ranges.vocab_blob_area.start,
    )?;
    ensure_non_overlapping(
        "vocab blob",
        &ranges.vocab_blob_area,
        "start records",
        ranges.starts.start,
    )?;
    ensure_non_overlapping(
        "start records",
        &ranges.starts,
        "model3 pairs",
        ranges.model3_pairs.start,
    )?;
    ensure_non_overlapping(
        "model3 pairs",
        &ranges.model3_pairs,
        "model3 prefixes",
        ranges.model3_prefixes.start,
    )?;
    ensure_non_overlapping(
        "model3 prefixes",
        &ranges.model3_prefixes,
        "model3 edges",
        ranges.model3_edges.start,
    )?;
    ensure_non_overlapping(
        "model3 edges",
        &ranges.model3_edges,
        "model2 prefixes",
        ranges.model2_prefixes.start,
    )?;
    ensure_non_overlapping(
        "model2 prefixes",
        &ranges.model2_prefixes,
        "model2 edges",
        ranges.model2_edges.start,
    )?;
    ensure_non_overlapping(
        "model2 edges",
        &ranges.model2_edges,
        "model1 prefixes",
        ranges.model1_prefixes.start,
    )?;
    ensure_non_overlapping(
        "model1 prefixes",
        &ranges.model1_prefixes,
        "model1 edges",
        ranges.model1_edges.start,
    )?;

    let file_size = usize_from_u64(header.file_size, "file_size")?;
    if ranges.model1_edges.end != file_size {
        return Err("model1 edges section must end at file_size".into());
    }

    Ok(())
}

fn ensure_non_overlapping(
    left_name: &str,
    left_range: &std::ops::Range<usize>,
    right_name: &str,
    right_start: usize,
) -> Result<(), DynError> {
    if left_range.end > right_start {
        return Err(format!("{left_name} section overlaps {right_name} section").into());
    }

    Ok(())
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
    let mut seen = vec![false; model3_prefix_count];

    for record in starts {
        let prefix_id = usize_from_u32(record.prefix_id, "start prefix_id")?;
        if prefix_id >= model3_prefix_count {
            return Err("start prefix_id is out of range".into());
        }
        if seen[prefix_id] {
            return Err("duplicate start prefix_id is not allowed".into());
        }
        seen[prefix_id] = true;

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
        *entry = (*entry)
            .checked_add(u64::from(delta))
            .ok_or("start count overflow while decoding")?;
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
