use std::collections::HashMap;

use super::{
    DynError, MODEL_SECTION_HEADER_SIZE, START_SECTION_HEADER_SIZE, StorageCompressionMode,
    align_to_eight, bytes_for_len, checked_add, compute_checksum, model_record_size,
    start_record_size, u32_from_usize, u64_from_usize, validate_special_tokens,
};
use crate::markov::{Count, MarkovChain, Prefix, TokenId};

use super::types::{
    EdgeRecord, ModelRecord, ModelSection, SectionDescriptor, SectionKind, StartRecord,
    StorageSections, VocabSections,
};

const VOCAB_COMPRESSION_THRESHOLD: usize = 128;
const REPEAT_CONTROL_THRESHOLD: u8 = 128;
const REPEAT_CHUNK_MIN: usize = 3;
const REPEAT_CHUNK_MAX: usize = 130;

pub(super) fn compile_chain(
    chain: &MarkovChain,
    min_edge_count: Count,
) -> Result<StorageSections, DynError> {
    validate_special_tokens(chain.id_to_token())?;
    if min_edge_count.get() == 0 {
        return Err("min_edge_count must be >= 1".into());
    }

    let starts = compile_starts(chain)?;
    let mut models = Vec::with_capacity(chain.order().as_usize()?);
    for order_val in (1..=chain.order().as_usize()?).rev() {
        let model = chain
            .models()
            .get(order_val - 1)
            .ok_or_else(|| format!("chain model section for order {order_val} is missing"))?;
        models.push(compile_model(model, order_val, min_edge_count)?);
    }

    Ok(StorageSections {
        ngram_order: chain.order(),
        vocab: VocabSections {
            offsets: build_vocab_offsets(chain.id_to_token())?,
            blob: chain.id_to_token().join("").into_bytes(),
        },
        starts,
        models,
    })
}

pub(super) fn encode_storage(
    sections: &StorageSections,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    let mut bytes = Vec::new();

    let (vocab_blob, flags) = encode_vocab_blob(&sections.vocab.blob, compression_mode)?;
    let section_count = u64_from_usize(sections.models.len() + 3, "section count")?;
    let file_size = calculate_file_size(sections, vocab_blob.len(), section_count)?;

    write_header(&mut bytes, sections, flags, section_count, file_size)?;
    let descriptors = write_descriptors(&mut bytes, sections, vocab_blob.len())?;
    pad_to_eight(&mut bytes)?;

    write_sections(&mut bytes, sections, &descriptors, vocab_blob.as_slice())?;

    let checksum = compute_checksum(bytes.as_slice())?;
    super::write_u64_at(bytes.as_mut_slice(), super::CHECKSUM_OFFSET, checksum)?;

    Ok(bytes)
}

fn build_vocab_offsets(tokens: &[String]) -> Result<Vec<u64>, DynError> {
    let mut offsets = Vec::with_capacity(tokens.len() + 1);
    let mut current = 0_u64;
    offsets.push(current);

    for token in tokens {
        let len = u64_from_usize(token.len(), "token length")?;
        current = checked_add(current, len, "vocab offset")?;
        offsets.push(current);
    }

    Ok(offsets)
}

fn calculate_file_size(
    sections: &StorageSections,
    vocab_blob_len: usize,
    section_count: u64,
) -> Result<u64, DynError> {
    let mut current = super::aligned_metadata_end(section_count)?;

    let mut add_section = |size: u64| {
        current = align_to_eight(current);
        current = checked_add(current, size, "file size")?;
        Ok::<(), DynError>(())
    };

    add_section(bytes_for_len(
        sections.vocab.offsets.len(),
        8,
        "vocab offsets section",
    )?)?;
    add_section(u64_from_usize(vocab_blob_len, "vocab blob section")?)?;
    add_section(calculate_starts_section_size(
        sections.starts.as_slice(),
        sections.ngram_order.as_usize()?,
    )?)?;

    for model in &sections.models {
        add_section(calculate_model_section_size(model)?)?;
    }

    Ok(current)
}

fn calculate_starts_section_size(records: &[StartRecord], order: usize) -> Result<u64, DynError> {
    let records_len = bytes_for_len(records.len(), start_record_size(order)?, "starts section")?;
    checked_add(
        START_SECTION_HEADER_SIZE,
        records_len,
        "starts section size",
    )
}

fn calculate_model_section_size(model: &ModelSection) -> Result<u64, DynError> {
    let records_len = bytes_for_len(
        model.records.len(),
        model_record_size(model.order)?,
        "model section records",
    )?;
    let edges_len = bytes_for_len(model.edges.len(), super::EDGE_RECORD_SIZE, "model section edges")?;
    checked_add(
        MODEL_SECTION_HEADER_SIZE,
        checked_add(records_len, edges_len, "model section records and edges")?,
        "model section size",
    )
}

fn compile_model(
    model_data: &HashMap<Prefix, HashMap<TokenId, Count>>,
    order: usize,
    min_edge_count: Count,
) -> Result<ModelSection, DynError> {
    let mut records = Vec::with_capacity(model_data.len());
    let mut edges = Vec::with_capacity(model_data.len());
    let mut prefixes = model_data.keys().collect::<Vec<_>>();
    prefixes.sort_unstable();

    for prefix in prefixes {
        if prefix.len() != order {
            return Err(format!(
                "model{order} prefix length mismatch: expected {order}, got {}",
                prefix.len()
            )
            .into());
        }

        let candidates = model_data
            .get(prefix)
            .ok_or_else(|| format!("model{order} is missing prefix {prefix:?}"))?;
        let mut targets = candidates.keys().collect::<Vec<_>>();
        targets.sort_unstable();

        let edge_start = u32_from_usize(edges.len(), "model edge start")?;
        let mut cumulative = 0_u64;

        for next in targets {
            let count = candidates
                .get(next)
                .ok_or_else(|| format!("model{order} is missing next token {next:?}"))?;
            if count.get() < min_edge_count.get() {
                continue;
            }

            cumulative = cumulative
                .checked_add(count.get())
                .ok_or("model edge cumulative count overflow")?;
            edges.push(EdgeRecord { next: *next, cumulative: Count::new(cumulative) });
        }

        let edge_len = u32_from_usize(edges.len(), "model edge len")? - edge_start;
        if edge_len > 0 {
            records.push(ModelRecord {
                prefix: prefix.clone(),
                edge_start,
                edge_len,
                total: Count::new(cumulative),
            });
        }
    }

    Ok(ModelSection {
        order,
        records,
        edges,
    })
}

fn compile_starts(chain: &MarkovChain) -> Result<Vec<StartRecord>, DynError> {
    let mut records = Vec::with_capacity(chain.starts().len());
    let mut prefixes = chain.starts().keys().collect::<Vec<_>>();
    prefixes.sort_unstable();

    let mut cumulative = 0_u64;
    for prefix in prefixes {
        if prefix.len() != chain.order().as_usize()? {
            return Err(format!(
                "start record prefix length mismatch: expected {}, got {}",
                chain.order().as_usize()?,
                prefix.len()
            )
            .into());
        }

        let count = chain
            .starts()
            .get(prefix)
            .ok_or_else(|| format!("chain starts is missing prefix {prefix:?}"))?;
        if count.get() == 0 {
            continue;
        }

        cumulative = cumulative
            .checked_add(count.get())
            .ok_or("start record cumulative count overflow")?;
        records.push(StartRecord {
            prefix: prefix.clone(),
            cumulative: Count::new(cumulative),
        });
    }

    Ok(records)
}

fn write_header(
    target: &mut Vec<u8>,
    sections: &StorageSections,
    flags: u32,
    section_count: u64,
    file_size: u64,
) -> Result<(), DynError> {
    target.extend_from_slice(super::MAGIC.as_slice());
    write_u32(target, super::VERSION);
    write_u32(target, flags);
    write_u32(target, super::TOKENIZER_VERSION);
    write_u32(target, super::NORMALIZATION_FLAGS);
    write_u32(target, u32_from_usize(sections.ngram_order.as_usize()?, "ngram order")?);
    write_u64(target, section_count);
    write_u64(target, file_size);
    write_u64(target, super::CHECKSUM_PLACEHOLDER);
    Ok(())
}

fn write_descriptors(
    target: &mut Vec<u8>,
    sections: &StorageSections,
    vocab_blob_len: usize,
) -> Result<Vec<SectionDescriptor>, DynError> {
    let section_count = u64_from_usize(sections.models.len() + 3, "section count")?;
    let mut descriptors = Vec::with_capacity(usize::try_from(section_count).unwrap_or(0));
    let mut current_offset = super::aligned_metadata_end(section_count)?;

    let mut add_descriptor = |kind: SectionKind, flags: u32, size: u64| {
        current_offset = align_to_eight(current_offset);
        let descriptor = SectionDescriptor {
            kind: kind.as_u32(),
            flags,
            offset: current_offset,
            size,
        };
        descriptors.push(descriptor);
        write_u32(target, descriptor.kind);
        write_u32(target, descriptor.flags);
        write_u64(target, descriptor.offset);
        write_u64(target, descriptor.size);
        current_offset += size;
        Ok::<(), DynError>(())
    };

    add_descriptor(
        SectionKind::VocabOffsets,
        0,
        bytes_for_len(sections.vocab.offsets.len(), 8, "vocab offsets")?,
    )?;
    add_descriptor(
        SectionKind::VocabBlob,
        0,
        u64_from_usize(vocab_blob_len, "vocab blob")?,
    )?;
    add_descriptor(
        SectionKind::Starts,
        0,
        calculate_starts_section_size(sections.starts.as_slice(), sections.ngram_order.as_usize()?)?,
    )?;

    for model in &sections.models {
        add_descriptor(
            SectionKind::Model,
            u32_from_usize(model.order, "model order")?,
            calculate_model_section_size(model)?,
        )?;
    }

    Ok(descriptors)
}

fn write_sections(
    target: &mut Vec<u8>,
    sections: &StorageSections,
    descriptors: &[SectionDescriptor],
    vocab_blob: &[u8],
) -> Result<(), DynError> {
    write_u64_section(
        target,
        sections.vocab.offsets.as_slice(),
        descriptors.first().ok_or("missing vocab offsets descriptor")?,
    )?;
    write_blob_section(
        target,
        vocab_blob,
        descriptors.get(1).ok_or("missing vocab blob descriptor")?,
    )?;
    write_starts_section(
        target,
        sections.starts.as_slice(),
        sections.ngram_order.as_usize()?,
        descriptors.get(2).ok_or("missing starts descriptor")?,
    )?;

    for (index, model) in sections.models.iter().enumerate() {
        write_model_section(
            target,
            model,
            descriptors
                .get(index + 3)
                .ok_or("missing model descriptor")?,
        )?;
    }

    Ok(())
}

fn write_u64_section(
    target: &mut Vec<u8>,
    values: &[u64],
    descriptor: &SectionDescriptor,
) -> Result<(), DynError> {
    pad_to_offset(target, descriptor.offset)?;
    for value in values {
        write_u64(target, *value);
    }
    Ok(())
}

fn write_blob_section(
    target: &mut Vec<u8>,
    blob: &[u8],
    descriptor: &SectionDescriptor,
) -> Result<(), DynError> {
    pad_to_offset(target, descriptor.offset)?;
    target.extend_from_slice(blob);
    Ok(())
}

fn write_starts_section(
    target: &mut Vec<u8>,
    records: &[StartRecord],
    ngram_order: usize,
    descriptor: &SectionDescriptor,
) -> Result<(), DynError> {
    pad_to_offset(target, descriptor.offset)?;

    write_u32(target, u32_from_usize(records.len(), "start records")?);
    for record in records {
        if record.prefix.len() != ngram_order {
            return Err(format!(
                "start record prefix length mismatch: expected {ngram_order}, got {}",
                record.prefix.len()
            )
            .into());
        }

        for token_id in record.prefix.as_slice() {
            write_u32(target, token_id.get());
        }
        write_u64(target, record.cumulative.get());
    }

    Ok(())
}

fn write_model_section(
    target: &mut Vec<u8>,
    section: &ModelSection,
    descriptor: &SectionDescriptor,
) -> Result<(), DynError> {
    pad_to_offset(target, descriptor.offset)?;

    write_u32(target, u32_from_usize(section.records.len(), "model records")?);
    write_u32(target, u32_from_usize(section.edges.len(), "model edges")?);

    for record in &section.records {
        if record.prefix.len() != section.order {
            return Err(format!(
                "model{} record prefix length mismatch: expected {}, got {}",
                section.order,
                section.order,
                record.prefix.len()
            )
            .into());
        }

        for token_id in record.prefix.as_slice() {
            write_u32(target, token_id.get());
        }
        write_u32(target, record.edge_start);
        write_u32(target, record.edge_len);
        write_u64(target, record.total.get());
    }

    for edge in &section.edges {
        write_u32(target, edge.next.get());
        write_u64(target, edge.cumulative.get());
    }

    Ok(())
}

fn pad_to_offset(target: &mut Vec<u8>, offset: u64) -> Result<(), DynError> {
    let offset = usize::try_from(offset).map_err(|_error| "offset exceeds usize range")?;
    if target.len() > offset {
        return Err("target already exceeds requested offset".into());
    }
    target.resize(offset, 0);
    Ok(())
}

fn pad_to_eight(target: &mut Vec<u8>) -> Result<(), DynError> {
    let current = u64_from_usize(target.len(), "buffer size")?;
    let padded = align_to_eight(current);
    pad_to_offset(target, padded)
}

fn encode_vocab_blob(
    blob: &[u8],
    mode: StorageCompressionMode,
) -> Result<(Vec<u8>, u32), DynError> {
    if blob.len() < VOCAB_COMPRESSION_THRESHOLD && mode == StorageCompressionMode::Auto {
        return Ok((blob.to_vec(), 0));
    }

    match mode {
        StorageCompressionMode::Uncompressed => Ok((blob.to_vec(), 0)),
        StorageCompressionMode::Rle => {
            Ok((encode_vocab_blob_rle(blob)?, super::FLAG_VOCAB_BLOB_RLE))
        }
        StorageCompressionMode::Zstd | StorageCompressionMode::Auto => {
            let encoded = zstd::bulk::compress(blob, 3)?;
            Ok((encoded, super::FLAG_VOCAB_BLOB_ZSTD))
        }
    }
}

fn encode_vocab_blob_rle(blob: &[u8]) -> Result<Vec<u8>, DynError> {
    let mut encoded = Vec::with_capacity(blob.len());
    let mut cursor = 0_usize;

    while cursor < blob.len() {
        let repeat_len = count_repeats(blob.get(cursor..).ok_or("RLE: blob range is invalid")?);
        if repeat_len >= REPEAT_CHUNK_MIN {
            let chunk_len = repeat_len.min(REPEAT_CHUNK_MAX);
            let control = REPEAT_CONTROL_THRESHOLD
                .checked_add(u8::try_from(chunk_len - REPEAT_CHUNK_MIN).map_err(|_error| "RLE: chunk_len overflow")?)
                .ok_or("RLE: control byte overflow")?;
            encoded.push(control);
            encoded.push(
                *blob
                    .get(cursor)
                    .ok_or("RLE: repeat byte range is invalid")?,
            );
            cursor += chunk_len;
        } else {
            let literal_len = count_literals(blob.get(cursor..).ok_or("RLE: literal blob range is invalid")?);
            let chunk_len = literal_len.min(usize::from(REPEAT_CONTROL_THRESHOLD));
            encoded.push(u8::try_from(chunk_len - 1).map_err(|_error| "RLE: literal chunk_len underflow")?);
            encoded.extend_from_slice(
                blob.get(cursor..cursor + chunk_len)
                    .ok_or("RLE: literal chunk range is invalid")?,
            );
            cursor += chunk_len;
        }
    }

    Ok(encoded)
}

fn count_repeats(data: &[u8]) -> usize {
    let Some(first) = data.first() else {
        return 0;
    };
    data.iter().take_while(|&&b| b == *first).count()
}

fn count_literals(data: &[u8]) -> usize {
    let mut count = 0;
    while count < data.len() {
        if count_repeats(data.get(count..).unwrap_or(&[])) >= REPEAT_CHUNK_MIN {
            break;
        }
        count += 1;
    }
    count
}

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}
