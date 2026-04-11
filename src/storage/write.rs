use std::collections::HashMap;

use lz4_flex::block::compress as lz4_compress;

use super::{
    CHECKSUM_PLACEHOLDER, DESCRIPTOR_SIZE, DynError, EDGE_RECORD_SIZE, FLAG_VOCAB_BLOB_LZ4_FLEX,
    FLAG_VOCAB_BLOB_RLE, FLAG_VOCAB_BLOB_ZSTD, FLAGS, HEADER_SIZE, MAGIC,
    MODEL_SECTION_HEADER_SIZE, NORMALIZATION_FLAGS, START_SECTION_HEADER_SIZE,
    StorageCompressionMode, TOKENIZER_VERSION, VERSION, align_to_eight, aligned_metadata_end,
    bytes_for_len, checked_add, compute_checksum, descriptor_count_for_ngram_order,
    model_record_size, start_record_size, u32_from_usize, u64_from_usize, usize_from_u32,
    usize_from_u64, validate_special_tokens, validate_token_id, validate_token_index,
};
use crate::markov::{Count, MarkovChain, Prefix, TokenId, validate_ngram_order};

use super::types::{
    EdgeRecord, Header, ModelRecord, ModelSection, SectionDescriptor, SectionKind, StartRecord,
    StorageSections, VocabSections,
};

const LITERAL_CHUNK_MAX: usize = 128;
const REPEAT_CHUNK_MIN: usize = 3;
const REPEAT_CHUNK_MAX: usize = 130;

pub(super) fn compile_chain(
    chain: &MarkovChain,
    min_edge_count: Count,
) -> Result<StorageSections, DynError> {
    if min_edge_count == 0 {
        return Err("min_edge_count must be greater than zero".into());
    }

    validate_ngram_order(chain.ngram_order, "chain ngram order")?;
    validate_special_tokens(chain.id_to_token.as_slice())?;
    validate_token_index(chain)?;

    if chain.models.len() != chain.ngram_order {
        return Err("model count does not match ngram order".into());
    }

    let token_count = u32_from_usize(chain.id_to_token.len(), "token count")?;
    let vocab = build_vocab(chain.id_to_token.as_slice())?;
    let models = build_models(chain, token_count, min_edge_count)?;
    let starts = build_starts(chain, token_count, min_edge_count)?;

    Ok(StorageSections {
        ngram_order: chain.ngram_order,
        vocab,
        starts,
        models,
    })
}

pub(super) fn encode_storage(
    sections: &StorageSections,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    validate_ngram_order(sections.ngram_order, "storage ngram order")?;

    let encoded_vocab_blob = encode_vocab_blob(sections.vocab.blob.as_slice(), compression_mode)?;
    let payloads = build_section_payloads(sections, encoded_vocab_blob.bytes.as_slice())?;
    let ngram_order = u32_from_usize(sections.ngram_order, "storage ngram order")?;
    let (mut header, descriptors) =
        build_metadata(payloads.as_slice(), encoded_vocab_blob.flags, ngram_order)?;

    let metadata_without_checksum = encode_metadata(header, descriptors.as_slice());
    let mut bytes = write_section_payloads(
        payloads.as_slice(),
        descriptors.as_slice(),
        header.file_size,
    )?;
    let metadata_without_checksum_len = metadata_without_checksum.len();
    bytes
        .get_mut(..metadata_without_checksum_len)
        .ok_or("encoded metadata without checksum exceeds payload length")?
        .copy_from_slice(metadata_without_checksum.as_slice());

    header.checksum = compute_checksum(bytes.as_slice())?;

    let metadata = encode_metadata(header, descriptors.as_slice());
    let metadata_len = metadata.len();
    bytes
        .get_mut(..metadata_len)
        .ok_or("encoded metadata exceeds payload length")?
        .copy_from_slice(metadata.as_slice());

    Ok(bytes)
}

#[derive(Debug)]
struct EncodedVocabBlob {
    bytes: Vec<u8>,
    flags: u32,
}

#[derive(Debug)]
struct SectionPayload {
    kind: SectionKind,
    flags: u32,
    bytes: Vec<u8>,
}

fn build_vocab(tokens: &[String]) -> Result<VocabSections, DynError> {
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

    Ok(VocabSections { offsets, blob })
}

fn build_models(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Vec<ModelSection>, DynError> {
    let mut sections = Vec::with_capacity(chain.ngram_order);

    for order in 1..=chain.ngram_order {
        let model = chain
            .models
            .get(order - 1)
            .ok_or("model index is out of bounds")?;
        sections.push(build_model_section(
            order,
            model,
            token_count,
            min_edge_count,
        )?);
    }

    Ok(sections)
}

fn build_model_section(
    order: usize,
    source: &HashMap<Prefix, HashMap<TokenId, Count>>,
    token_count: u32,
    min_edge_count: Count,
) -> Result<ModelSection, DynError> {
    let mut entries = source.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

    let mut records = Vec::new();
    let mut edges = Vec::new();

    for (prefix, source_edges) in entries {
        if prefix.len() != order {
            return Err(format!(
                "model{order} prefix length mismatch: expected {order}, got {}",
                prefix.len()
            )
            .into());
        }

        validate_prefix(prefix.as_slice(), token_count, "model prefix")?;

        let retained = retained_edges(source_edges, token_count, min_edge_count, "model edge")?;
        if retained.is_empty() {
            continue;
        }

        let edge_start = u32_from_usize(edges.len(), "model edge start")?;
        let edge_len = u32_from_usize(retained.len(), "model edge length")?;
        let mut cumulative = 0_u64;

        for (next, count) in retained {
            cumulative = cumulative
                .checked_add(count)
                .ok_or("model edge cumulative overflow")?;
            edges.push(EdgeRecord { next, cumulative });
        }

        records.push(ModelRecord {
            prefix: prefix.clone(),
            edge_start,
            edge_len,
            total: cumulative,
        });
    }

    Ok(ModelSection {
        order,
        records,
        edges,
    })
}

fn build_starts(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Vec<StartRecord>, DynError> {
    let top_model = chain.models.last().ok_or("top-level model is missing")?;
    let mut entries = chain.starts.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

    let mut records = Vec::new();
    let mut cumulative = 0_u64;

    for (prefix, count) in entries {
        if prefix.len() != chain.ngram_order {
            return Err(format!(
                "start prefix length mismatch: expected {}, got {}",
                chain.ngram_order,
                prefix.len()
            )
            .into());
        }
        if *count == 0 {
            continue;
        }

        validate_prefix(prefix.as_slice(), token_count, "start prefix")?;

        let survives = top_model
            .get(prefix.as_slice())
            .is_some_and(|edges| has_retained_edge(edges, min_edge_count));
        if !survives {
            continue;
        }

        cumulative = cumulative
            .checked_add(*count)
            .ok_or("start cumulative overflow")?;
        records.push(StartRecord {
            prefix: prefix.clone(),
            cumulative,
        });
    }

    Ok(records)
}

fn retained_edges(
    source_edges: &HashMap<TokenId, Count>,
    token_count: u32,
    min_edge_count: Count,
    context: &str,
) -> Result<Vec<(TokenId, Count)>, DynError> {
    let mut retained = source_edges
        .iter()
        .filter_map(|(next, count)| (*count >= min_edge_count).then_some((*next, *count)))
        .collect::<Vec<_>>();
    retained.sort_unstable_by_key(|(next, _)| *next);

    for (next, count) in &retained {
        validate_token_id(*next, token_count, context)?;
        if *count == 0 {
            return Err(format!("{context}: retained edge count must be > 0").into());
        }
    }

    Ok(retained)
}

fn has_retained_edge(edges: &HashMap<TokenId, Count>, min_edge_count: Count) -> bool {
    edges.values().any(|count| *count >= min_edge_count)
}

fn validate_prefix(prefix: &[TokenId], token_count: u32, context: &str) -> Result<(), DynError> {
    for token_id in prefix {
        validate_token_id(*token_id, token_count, context)?;
    }

    Ok(())
}

fn build_section_payloads(
    sections: &StorageSections,
    encoded_vocab_blob: &[u8],
) -> Result<Vec<SectionPayload>, DynError> {
    if sections.models.len() != sections.ngram_order {
        return Err("storage model count does not match ngram order".into());
    }

    let mut payloads = Vec::with_capacity(sections.ngram_order.saturating_add(3));
    payloads.push(SectionPayload {
        kind: SectionKind::VocabOffsets,
        flags: 0,
        bytes: encode_u64_values(sections.vocab.offsets.as_slice())?,
    });
    payloads.push(SectionPayload {
        kind: SectionKind::VocabBlob,
        flags: 0,
        bytes: encoded_vocab_blob.to_vec(),
    });
    payloads.push(SectionPayload {
        kind: SectionKind::Starts,
        flags: 0,
        bytes: encode_starts_section(sections.starts.as_slice(), sections.ngram_order)?,
    });

    for section in sections.models.iter().rev() {
        let flags = u32_from_usize(section.order, "model section order")?;
        payloads.push(SectionPayload {
            kind: SectionKind::Model,
            flags,
            bytes: encode_model_section(section)?,
        });
    }

    Ok(payloads)
}

fn encode_u64_values(values: &[u64]) -> Result<Vec<u8>, DynError> {
    let capacity = usize_from_u64(
        bytes_for_len(values.len(), 8, "vocab offsets section")?,
        "vocab offsets section",
    )?;
    let mut bytes = Vec::with_capacity(capacity);
    for value in values {
        write_u64(&mut bytes, *value);
    }
    Ok(bytes)
}

fn encode_starts_section(records: &[StartRecord], ngram_order: usize) -> Result<Vec<u8>, DynError> {
    let records_len = u32_from_usize(records.len(), "start record count")?;
    let body_bytes = bytes_for_len(
        records.len(),
        start_record_size(ngram_order)?,
        "start section",
    )?;
    let total_bytes = checked_add(START_SECTION_HEADER_SIZE, body_bytes, "start section size")?;
    let capacity = usize_from_u64(total_bytes, "start section size")?;
    let mut bytes = Vec::with_capacity(capacity);
    write_u32(&mut bytes, records_len);

    for record in records {
        if record.prefix.len() != ngram_order {
            return Err(format!(
                "start record prefix length mismatch: expected {ngram_order}, got {}",
                record.prefix.len()
            )
            .into());
        }
        write_prefix(&mut bytes, record.prefix.as_slice());
        write_u64(&mut bytes, record.cumulative);
    }

    Ok(bytes)
}

fn encode_model_section(section: &ModelSection) -> Result<Vec<u8>, DynError> {
    let records_len = u32_from_usize(section.records.len(), "model record count")?;
    let edges_len = u32_from_usize(section.edges.len(), "model edge count")?;
    let record_bytes = bytes_for_len(
        section.records.len(),
        model_record_size(section.order)?,
        "model section records",
    )?;
    let edge_bytes = bytes_for_len(section.edges.len(), EDGE_RECORD_SIZE, "model section edges")?;
    let total_bytes = checked_add(
        MODEL_SECTION_HEADER_SIZE,
        checked_add(record_bytes, edge_bytes, "model section body")?,
        "model section size",
    )?;
    let capacity = usize_from_u64(total_bytes, "model section size")?;
    let mut bytes = Vec::with_capacity(capacity);
    write_u32(&mut bytes, records_len);
    write_u32(&mut bytes, edges_len);

    for record in &section.records {
        if record.prefix.len() != section.order {
            return Err(format!(
                "model record prefix length mismatch: expected {}, got {}",
                section.order,
                record.prefix.len()
            )
            .into());
        }
        write_prefix(&mut bytes, record.prefix.as_slice());
        write_u32(&mut bytes, record.edge_start);
        write_u32(&mut bytes, record.edge_len);
        write_u64(&mut bytes, record.total);
    }

    for edge in &section.edges {
        write_u32(&mut bytes, edge.next);
        write_u64(&mut bytes, edge.cumulative);
    }

    Ok(bytes)
}

fn write_prefix(target: &mut Vec<u8>, prefix: &[TokenId]) {
    for token_id in prefix {
        write_u32(target, *token_id);
    }
}

fn build_metadata(
    payloads: &[SectionPayload],
    flags: u32,
    ngram_order: u32,
) -> Result<(Header, Vec<SectionDescriptor>), DynError> {
    let expected_section_count =
        descriptor_count_for_ngram_order(usize_from_u32(ngram_order, "ngram order")?)?;
    let actual_section_count = u64_from_usize(payloads.len(), "section payload count")?;
    if actual_section_count != expected_section_count {
        return Err(format!(
            "section payload count mismatch: expected {expected_section_count}, got {actual_section_count}"
        )
        .into());
    }

    let mut offset = aligned_metadata_end(actual_section_count)?;
    let mut file_size = offset;
    let mut descriptors = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let size = u64_from_usize(payload.bytes.len(), payload.kind.label())?;
        descriptors.push(SectionDescriptor {
            kind: payload.kind.as_u32(),
            flags: payload.flags,
            offset,
            size,
        });

        file_size = checked_add(offset, size, payload.kind.label())?;
        offset = align_to_eight(file_size);
    }

    Ok((
        Header {
            magic: MAGIC,
            version: VERSION,
            flags,
            tokenizer_version: TOKENIZER_VERSION,
            normalization_flags: NORMALIZATION_FLAGS,
            ngram_order,
            section_count: actual_section_count,
            file_size,
            checksum: CHECKSUM_PLACEHOLDER,
        },
        descriptors,
    ))
}

fn encode_metadata(header: Header, descriptors: &[SectionDescriptor]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_SIZE + descriptors.len() * DESCRIPTOR_SIZE);

    bytes.extend_from_slice(header.magic.as_slice());
    write_u32(&mut bytes, header.version);
    write_u32(&mut bytes, header.flags);
    write_u32(&mut bytes, header.tokenizer_version);
    write_u32(&mut bytes, header.normalization_flags);
    write_u32(&mut bytes, header.ngram_order);
    write_u64(&mut bytes, header.section_count);
    write_u64(&mut bytes, header.file_size);
    write_u64(&mut bytes, header.checksum);

    for descriptor in descriptors {
        write_u32(&mut bytes, descriptor.kind);
        write_u32(&mut bytes, descriptor.flags);
        write_u64(&mut bytes, descriptor.offset);
        write_u64(&mut bytes, descriptor.size);
    }

    debug_assert_eq!(
        bytes.len(),
        HEADER_SIZE + descriptors.len() * DESCRIPTOR_SIZE
    );
    bytes
}

fn write_section_payloads(
    payloads: &[SectionPayload],
    descriptors: &[SectionDescriptor],
    file_size: u64,
) -> Result<Vec<u8>, DynError> {
    if payloads.len() != descriptors.len() {
        return Err("section payload count does not match descriptor count".into());
    }

    let metadata_end = usize_from_u64(
        aligned_metadata_end(u64_from_usize(descriptors.len(), "descriptor count")?)?,
        "metadata size",
    )?;
    let mut bytes = vec![0; metadata_end];

    for (payload, descriptor) in payloads.iter().zip(descriptors.iter()) {
        if descriptor.kind != payload.kind.as_u32() || descriptor.flags != payload.flags {
            return Err("section descriptor order does not match payload order".into());
        }

        write_at_offset(&mut bytes, descriptor.offset, payload.bytes.as_slice())?;
    }

    let expected_file_size = usize_from_u64(file_size, "file size")?;
    if bytes.len() != expected_file_size {
        return Err(format!(
            "serialized size mismatch: expected {expected_file_size}, got {}",
            bytes.len()
        )
        .into());
    }

    Ok(bytes)
}

fn write_at_offset(target: &mut Vec<u8>, offset: u64, bytes: &[u8]) -> Result<(), DynError> {
    pad_to_offset(target, offset)?;
    target.extend_from_slice(bytes);
    Ok(())
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

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn encode_vocab_blob(
    vocab_blob: &[u8],
    compression_mode: StorageCompressionMode,
) -> Result<EncodedVocabBlob, DynError> {
    match compression_mode {
        StorageCompressionMode::Auto => encode_auto(vocab_blob),
        StorageCompressionMode::Uncompressed => Ok(EncodedVocabBlob {
            bytes: vocab_blob.to_vec(),
            flags: FLAGS,
        }),
        StorageCompressionMode::Rle => Ok(EncodedVocabBlob {
            bytes: encode_rle(vocab_blob)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_RLE,
        }),
        StorageCompressionMode::Zstd => Ok(EncodedVocabBlob {
            bytes: zstd::bulk::compress(vocab_blob, 0)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_ZSTD,
        }),
        StorageCompressionMode::Lz4Flex => Ok(EncodedVocabBlob {
            bytes: lz4_compress(vocab_blob),
            flags: FLAGS | FLAG_VOCAB_BLOB_LZ4_FLEX,
        }),
    }
}

fn encode_auto(vocab_blob: &[u8]) -> Result<EncodedVocabBlob, DynError> {
    if vocab_blob.is_empty() {
        return Ok(EncodedVocabBlob {
            bytes: Vec::new(),
            flags: FLAGS,
        });
    }

    let mut best = EncodedVocabBlob {
        bytes: vocab_blob.to_vec(),
        flags: FLAGS,
    };

    for candidate in [
        EncodedVocabBlob {
            bytes: encode_rle(vocab_blob)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_RLE,
        },
        EncodedVocabBlob {
            bytes: zstd::bulk::compress(vocab_blob, 0)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_ZSTD,
        },
        EncodedVocabBlob {
            bytes: lz4_compress(vocab_blob),
            flags: FLAGS | FLAG_VOCAB_BLOB_LZ4_FLEX,
        },
    ] {
        if candidate.bytes.len() < best.bytes.len() {
            best = candidate;
        }
    }

    Ok(best)
}

fn encode_rle(input: &[u8]) -> Result<Vec<u8>, DynError> {
    let mut encoded = Vec::with_capacity(input.len());
    let mut cursor = 0_usize;

    while cursor < input.len() {
        let repeat_len = repeat_run_len(input, cursor);
        if repeat_len >= REPEAT_CHUNK_MIN {
            let value = *input.get(cursor).ok_or("rle cursor is out of bounds")?;
            push_repeat_chunk(&mut encoded, value, repeat_len)?;
            cursor += repeat_len;
            continue;
        }

        let literal_len = literal_run_len(input, cursor);
        let literal_end = cursor
            .checked_add(literal_len)
            .ok_or("rle literal length overflow")?;
        let literal = input
            .get(cursor..literal_end)
            .ok_or("rle literal range is out of bounds")?;
        push_literal_chunk(&mut encoded, literal)?;
        cursor += literal_len;
    }

    Ok(encoded)
}

fn repeat_run_len(input: &[u8], start: usize) -> usize {
    let Some(&value) = input.get(start) else {
        return 0;
    };
    let mut len = 1_usize;

    while start + len < input.len()
        && input.get(start + len).copied() == Some(value)
        && len < REPEAT_CHUNK_MAX
    {
        len += 1;
    }

    len
}

fn literal_run_len(input: &[u8], start: usize) -> usize {
    let mut len = 1_usize;

    while start + len < input.len() && len < LITERAL_CHUNK_MAX {
        if repeat_run_len(input, start + len) >= REPEAT_CHUNK_MIN {
            break;
        }
        len += 1;
    }

    len
}

fn push_repeat_chunk(encoded: &mut Vec<u8>, value: u8, repeat_len: usize) -> Result<(), DynError> {
    let repeat_control = repeat_len - REPEAT_CHUNK_MIN;
    let control =
        u8::try_from(repeat_control).map_err(|_error| "repeat chunk length is bounded")?;
    let control = control.saturating_add(128);
    encoded.push(control);
    encoded.push(value);
    Ok(())
}

fn push_literal_chunk(encoded: &mut Vec<u8>, bytes: &[u8]) -> Result<(), DynError> {
    let literal_control = bytes.len() - 1;
    let control =
        u8::try_from(literal_control).map_err(|_error| "literal chunk length is bounded")?;
    encoded.push(control);
    encoded.extend_from_slice(bytes);
    Ok(())
}
