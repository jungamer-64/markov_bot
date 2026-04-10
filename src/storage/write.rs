use std::collections::HashMap;

use super::{
    CHECKSUM, CompiledStorage, Count, DynError, EDGE_RECORD_SIZE, EdgeRecord, FLAGS, HEADER_SIZE,
    Header, MAGIC, MarkovChain, Model3Build, NORMALIZATION_FLAGS, PAIR3_RECORD_SIZE,
    PREFIX1_RECORD_SIZE, PREFIX2_RECORD_SIZE, PREFIX3_RECORD_SIZE, Pair3Record, Prefix1Record,
    Prefix2Record, Prefix3Record, START_RECORD_SIZE, SectionCounts, SectionSizes, StartRecord,
    TOKENIZER_VERSION, TokenId, VERSION, align_to_eight, bytes_for_len, checked_add,
    u32_from_usize, u64_from_usize, usize_from_u64, validate_special_tokens, validate_token_id,
};

pub(super) fn compile_chain(chain: &MarkovChain) -> Result<CompiledStorage, DynError> {
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

pub(super) fn encode_storage(compiled: &CompiledStorage) -> Result<Vec<u8>, DynError> {
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
