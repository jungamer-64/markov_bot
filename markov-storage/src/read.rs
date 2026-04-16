use std::{collections::HashMap, str};

use crate::StorageError;
use super::{
    DynError, EDGE_RECORD_SIZE, FLAG_VOCAB_BLOB_RLE,
    FLAG_VOCAB_BLOB_ZSTD, HEADER_SIZE, MAGIC, MODEL_SECTION_HEADER_SIZE, NORMALIZATION_FLAGS,
    START_SECTION_HEADER_SIZE, TOKENIZER_VERSION, VERSION, aligned_metadata_end, chain_to_snapshot,
    checked_add, compression_mode_from_flags, compute_checksum,
    model_record_size, start_record_size, u64_from_usize, usize_from_u32, usize_from_u64,
    validate_special_tokens, validate_token_id, vocab_blob_compression_flags,
};
use crate::markov::{Count, MarkovChain, NgramOrder, Prefix, TokenId};

use super::types::{
    EdgeRecord, Header, ModelRecord, ModelSection, SectionDescriptor, SectionEntry, SectionKind,
    SectionTable, StartRecord, StorageSections, VocabSections,
};

type RebuiltModels = Vec<HashMap<Prefix, HashMap<TokenId, Count>>>;

const REPEAT_CONTROL_THRESHOLD: u8 = 128;
const REPEAT_CHUNK_MIN: usize = 3;
const REPEAT_CHUNK_MAX: usize = 130;
const MAX_RLE_EXPANSION_PER_ENCODED_BYTE: usize = REPEAT_CHUNK_MAX / 2;

pub(super) fn decode_chain(
    bytes: &[u8],
    expected_ngram_order: NgramOrder,
) -> Result<MarkovChain, DynError> {
    let header = validate_header(bytes)?;
    let actual_ngram_order = NgramOrder::new(usize::try_from(header.ngram_order)
        .map_err(|_error| StorageError::Format("header ngram_order exceeds usize range".into()))?)?;
    if actual_ngram_order != expected_ngram_order {
        return Err(StorageError::NgramOrderMismatch {
            expected: expected_ngram_order,
            actual: actual_ngram_order,
        });
    }

    let expected_section_count = header.expected_section_count()?;
    if header.section_count != expected_section_count {
        return Err(StorageError::Format(format!(
            "section count mismatch: expected {expected_section_count}, got {}",
            header.section_count
        )));
    }

    let stored_file_size = u64_from_usize(bytes.len(), "file size")?;
    if header.file_size != stored_file_size {
        return Err(StorageError::Format(format!(
            "file size mismatch: header={}, actual={stored_file_size}",
            header.file_size
        )));
    }

    let actual_checksum = compute_checksum(bytes)?;
    if header.checksum != actual_checksum {
        return Err(StorageError::Checksum {
            expected: header.checksum,
            actual: actual_checksum,
        });
    }

    let table = build_section_table(bytes, &header)?;
    let sections = parse_storage(bytes, &header, &table, actual_ngram_order)?;

    rebuild_chain(&sections)
}

pub(super) fn decode_snapshot(bytes: &[u8]) -> Result<super::StorageSnapshot, DynError> {
    let header = validate_header(bytes)?;
    let actual_ngram_order = NgramOrder::new(usize::try_from(header.ngram_order)
        .map_err(|_error| StorageError::Format("header ngram_order exceeds usize range".into()))?)?;

    let expected_section_count = header.expected_section_count()?;
    if header.section_count != expected_section_count {
        return Err(StorageError::Format(format!(
            "section count mismatch: expected {expected_section_count}, got {}",
            header.section_count
        )));
    }

    let stored_file_size = u64_from_usize(bytes.len(), "file size")?;
    if header.file_size != stored_file_size {
        return Err(StorageError::Format(format!(
            "file size mismatch: header={}, actual={stored_file_size}",
            header.file_size
        )));
    }

    let actual_checksum = compute_checksum(bytes)?;
    if header.checksum != actual_checksum {
        return Err(StorageError::Checksum {
            expected: header.checksum,
            actual: actual_checksum,
        });
    }

    let table = build_section_table(bytes, &header)?;
    let sections = parse_storage(bytes, &header, &table, actual_ngram_order)?;

    let chain = rebuild_chain(&sections)?;
    let compression_mode = compression_mode_from_flags(header.flags)?;

    chain_to_snapshot(&chain, compression_mode)
}

fn validate_header(bytes: &[u8]) -> Result<Header, DynError> {
    if bytes.len() < HEADER_SIZE {
        return Err(StorageError::Format(
            "storage file is shorter than the header".into(),
        ));
    }

    let mut cursor = 0_usize;
    let magic = read_exact(bytes, &mut cursor, 8)?;
    let mut magic_bytes = [0_u8; 8];
    magic_bytes.copy_from_slice(magic);
    if magic_bytes != MAGIC {
        return Err(StorageError::Magic {
            expected: MAGIC,
            actual: magic_bytes,
        });
    }

    let version = read_u32_value(bytes, &mut cursor)?;
    if version != VERSION {
        return Err(StorageError::Version(version));
    }

    let flags = read_u32_value(bytes, &mut cursor)?;
    vocab_blob_compression_flags(flags)?;

    let tokenizer_version = read_u32_value(bytes, &mut cursor)?;
    if tokenizer_version != TOKENIZER_VERSION {
        return Err(StorageError::Format(format!(
            "unsupported tokenizer version: {tokenizer_version}"
        )));
    }

    let normalization_flags = read_u32_value(bytes, &mut cursor)?;
    if normalization_flags != NORMALIZATION_FLAGS {
        return Err(StorageError::Format(format!(
            "unsupported normalization flags: {normalization_flags}"
        )));
    }

    let ngram_order = read_u32_value(bytes, &mut cursor)?;
    let section_count = read_u64_value(bytes, &mut cursor)?;
    let file_size = read_u64_value(bytes, &mut cursor)?;
    let checksum = read_u64_value(bytes, &mut cursor)?;

    Ok(Header {
        _magic: magic_bytes,
        _version: version,
        flags,
        _tokenizer_version: tokenizer_version,
        _normalization_flags: normalization_flags,
        ngram_order,
        section_count,
        file_size,
        checksum,
    })
}

fn build_section_table(bytes: &[u8], header: &Header) -> Result<SectionTable, DynError> {
    let descriptor_count = usize_from_u64(header.section_count, "section count")?;
    let metadata_end = usize_from_u64(
        aligned_metadata_end(header.section_count)?,
        "aligned metadata end",
    )?;
    let mut entries = Vec::with_capacity(descriptor_count);
    let mut cursor = HEADER_SIZE;
    let mut last_end = metadata_end;

    for index in 0..descriptor_count {
        let descriptor = SectionDescriptor {
            kind: read_u32_value(bytes, &mut cursor)?,
            flags: read_u32_value(bytes, &mut cursor)?,
            offset: read_u64_value(bytes, &mut cursor)?,
            size: read_u64_value(bytes, &mut cursor)?,
        };
        validate_descriptor(&descriptor, index, descriptor_count, header.ngram_order)?;

        let range = descriptor_range(bytes, &descriptor)?;
        if range.start < metadata_end {
            return Err(format!(
                "{} section starts before aligned metadata end",
                section_label(&descriptor)
            )
            .into());
        }
        if range.start < last_end {
            return Err(format!(
                "{} section overlaps previous section",
                section_label(&descriptor)
            )
            .into());
        }
        last_end = range.end;

        entries.push(SectionEntry { descriptor, range });
    }

    Ok(SectionTable { entries })
}

fn validate_descriptor(
    descriptor: &SectionDescriptor,
    index: usize,
    descriptor_count: usize,
    ngram_order: u32,
) -> Result<(), DynError> {
    let kind = SectionKind::from_u32(descriptor.kind)
        .ok_or_else(|| format!("unknown section kind: {}", descriptor.kind))?;

    let expected_order = if index >= 3 {
        let model_index = index - 3;
        let ngram_order = usize_from_u32(ngram_order, "header ngram order")?;
        if model_index >= ngram_order {
            return Err("descriptor table has too many model sections".into());
        }
        Some(ngram_order - model_index)
    } else {
        None
    };

    match index {
        0 if kind == SectionKind::VocabOffsets && descriptor.flags == 0 => {}
        1 if kind == SectionKind::VocabBlob && descriptor.flags == 0 => {}
        2 if kind == SectionKind::Starts && descriptor.flags == 0 => {}
        3.. => {
            let expected_order = expected_order.ok_or("missing expected model order")?;
            if kind != SectionKind::Model {
                return Err(format!(
                    "section order mismatch at index {index}: expected model section"
                )
                .into());
            }

            let actual_order = usize_from_u32(descriptor.flags, "model section order")?;
            if actual_order != expected_order {
                return Err(format!(
                    "model section order mismatch at index {index}: expected {expected_order}, got {actual_order}"
                )
                .into());
            }
        }
        _ => {
            return Err(format!(
                "section order mismatch at index {index}: got {}",
                kind.label()
            )
            .into());
        }
    }

    if descriptor_count < 3 {
        return Err("section table is too short".into());
    }

    Ok(())
}

fn descriptor_range(
    bytes: &[u8],
    descriptor: &SectionDescriptor,
) -> Result<std::ops::Range<usize>, DynError> {
    let start = usize_from_u64(descriptor.offset, "section offset")?;
    let size = usize_from_u64(descriptor.size, "section size")?;
    let end = start.checked_add(size).ok_or("section range overflow")?;
    if end > bytes.len() {
        return Err(format!("{} section exceeds file bounds", section_label(descriptor)).into());
    }

    Ok(start..end)
}

fn section_label(descriptor: &SectionDescriptor) -> String {
    match descriptor.kind() {
        Some(SectionKind::Model) => format!("model(order={})", descriptor.flags),
        Some(kind) => kind.label().to_owned(),
        None => format!("unknown({})", descriptor.kind),
    }
}

fn parse_storage(
    bytes: &[u8],
    header: &Header,
    table: &SectionTable,
    ngram_order: NgramOrder,
) -> Result<StorageSections, DynError> {
    let vocab_offsets_entry = table.unique_entry(SectionKind::VocabOffsets)?;
    let vocab_offsets = parse_u64_section(section_bytes(bytes, vocab_offsets_entry)?)?;
    validate_vocab_offsets(vocab_offsets.as_slice())?;

    let vocab_blob_entry = table.unique_entry(SectionKind::VocabBlob)?;
    let expected_blob_size = vocab_offsets
        .last()
        .copied()
        .ok_or("vocab offsets are empty")?;
    let expected_blob_size = usize_from_u64(expected_blob_size, "vocab blob size")?;
    let vocab_blob = decode_vocab_blob(
        section_bytes(bytes, vocab_blob_entry)?,
        expected_blob_size,
        header.flags,
    )?;

    let starts_entry = table.unique_entry(SectionKind::Starts)?;
    let starts = parse_starts_section(section_bytes(bytes, starts_entry)?, ngram_order.as_usize()?)?;

    let mut models = Vec::with_capacity(ngram_order.as_usize()?);
    for entry in table.model_entries() {
        let order = usize_from_u32(entry.descriptor.flags, "model section order")?;
        models.push(parse_model_section(section_bytes(bytes, entry)?, order)?);
    }

    Ok(StorageSections {
        ngram_order,
        vocab: VocabSections {
            offsets: vocab_offsets,
            blob: vocab_blob,
        },
        starts,
        models,
    })
}

fn section_bytes<'a>(bytes: &'a [u8], entry: &SectionEntry) -> Result<&'a [u8], DynError> {
    bytes
        .get(entry.range.clone())
        .ok_or_else(|| "section range is invalid".into())
}

fn parse_u64_section(bytes: &[u8]) -> Result<Vec<u64>, DynError> {
    if !bytes.len().is_multiple_of(8) {
        return Err("u64 section size is not a multiple of 8".into());
    }

    let mut values = Vec::with_capacity(bytes.len() / 8);
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        values.push(read_u64_value(bytes, &mut cursor)?);
    }

    Ok(values)
}

fn parse_starts_section(bytes: &[u8], ngram_order: usize) -> Result<Vec<StartRecord>, DynError> {
    let mut cursor = 0_usize;
    let record_count = usize_from_u32(read_u32_value(bytes, &mut cursor)?, "start record count")?;
    let expected_records_bytes = bytes_for_count(
        record_count,
        start_record_size(ngram_order)?,
        "start section",
    )?;
    let expected_size = checked_add(
        START_SECTION_HEADER_SIZE,
        expected_records_bytes,
        "start section size",
    )?;
    let actual_size = u64_from_usize(bytes.len(), "start section size")?;
    if actual_size != expected_size {
        return Err(format!(
            "start section size mismatch: expected {expected_size}, got {actual_size}"
        )
        .into());
    }

    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        records.push(StartRecord {
            prefix: read_prefix(bytes, &mut cursor, ngram_order)?,
            cumulative: Count::new(read_u64_value(bytes, &mut cursor)?),
        });
    }

    Ok(records)
}

fn parse_model_section(bytes: &[u8], order: usize) -> Result<ModelSection, DynError> {
    let mut cursor = 0_usize;
    let record_count = usize_from_u32(read_u32_value(bytes, &mut cursor)?, "model record count")?;
    let edge_count = usize_from_u32(read_u32_value(bytes, &mut cursor)?, "model edge count")?;

    let expected_record_bytes = bytes_for_count(
        record_count,
        model_record_size(order)?,
        "model section records",
    )?;
    let expected_edge_bytes = bytes_for_count(edge_count, EDGE_RECORD_SIZE, "model section edges")?;
    let expected_size = checked_add(
        MODEL_SECTION_HEADER_SIZE,
        checked_add(
            expected_record_bytes,
            expected_edge_bytes,
            "model section body",
        )?,
        "model section size",
    )?;
    let actual_size = u64_from_usize(bytes.len(), "model section size")?;
    if actual_size != expected_size {
        return Err(format!(
            "model section size mismatch: expected {expected_size}, got {actual_size}"
        )
        .into());
    }

    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        records.push(ModelRecord {
            prefix: read_prefix(bytes, &mut cursor, order)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: Count::new(read_u64_value(bytes, &mut cursor)?),
        });
    }

    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        edges.push(EdgeRecord {
            next: TokenId::new(read_u32_value(bytes, &mut cursor)?),
            cumulative: Count::new(read_u64_value(bytes, &mut cursor)?),
        });
    }

    Ok(ModelSection {
        order,
        records,
        edges,
    })
}

fn read_prefix(bytes: &[u8], cursor: &mut usize, order: usize) -> Result<Prefix, DynError> {
    let mut prefix = Vec::with_capacity(order);
    for _ in 0..order {
        prefix.push(TokenId::new(read_u32_value(bytes, cursor)?));
    }
    Ok(Prefix::new(prefix))
}

fn rebuild_chain(sections: &StorageSections) -> Result<MarkovChain, DynError> {
    if sections.models.len() != sections.ngram_order.as_usize()? {
        return Err("storage model section count does not match ngram order".into());
    }

    let id_to_token = decode_vocab(
        sections.vocab.offsets.as_slice(),
        sections.vocab.blob.as_slice(),
    )?;
    validate_special_tokens(id_to_token.as_slice())?;

    let token_to_id = id_to_token
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let token_id =
                TokenId::new(u32::try_from(index).map_err(|_error| "token count exceeds u32 range")?);
            Ok((token.clone(), token_id))
        })
        .collect::<Result<HashMap<_, _>, DynError>>()?;

    let token_count =
        u32::try_from(id_to_token.len()).map_err(|_error| "token count exceeds u32 range")?;

    let starts = decode_starts(
        sections.starts.as_slice(),
        sections.ngram_order.as_usize()?,
        token_count,
    )?;
    let models = decode_models(
        sections.models.as_slice(),
        sections.ngram_order.as_usize()?,
        token_count,
    )?;

    MarkovChain::from_parts(
        sections.ngram_order,
        token_to_id,
        id_to_token,
        models,
        starts,
    ).map_err(|error| error.to_string().into())
}

fn decode_starts(
    records: &[StartRecord],
    ngram_order: usize,
    token_count: u32,
) -> Result<HashMap<Prefix, Count>, DynError> {
    let mut starts = HashMap::with_capacity(records.len());
    let mut previous_prefix: Option<&[TokenId]> = None;
    let mut previous_cumulative = Count::ZERO;

    for record in records {
        if record.prefix.len() != ngram_order {
            return Err(format!(
                "start record prefix length mismatch: expected {ngram_order}, got {}",
                record.prefix.len()
            )
            .into());
        }
        validate_prefix(record.prefix.as_slice(), token_count, "start record")?;

        if let Some(prev) = previous_prefix
            && prev >= record.prefix.as_slice()
        {
            return Err("start records must be sorted by unique prefix".into());
        }
        if record.cumulative.get() <= previous_cumulative.get() {
            return Err("start records must have strictly increasing cumulative counts".into());
        }

        let count = Count::new(record.cumulative.get() - previous_cumulative.get());
        starts.insert(record.prefix.clone(), count);
        previous_prefix = Some(record.prefix.as_slice());
        previous_cumulative = record.cumulative;
    }

    Ok(starts)
}

fn decode_models(
    sections: &[ModelSection],
    ngram_order: usize,
    token_count: u32,
) -> Result<RebuiltModels, DynError> {
    let mut models = Vec::with_capacity(ngram_order);

    for expected_order in 1..=ngram_order {
        let section = sections
            .get(ngram_order - expected_order)
            .ok_or("model section is missing")?;
        if section.order != expected_order {
            return Err(format!(
                "model section order mismatch: expected {expected_order}, got {}",
                section.order
            )
            .into());
        }

        models.push(decode_model_section(section, token_count)?);
    }

    Ok(models)
}

fn decode_model_section(
    section: &ModelSection,
    token_count: u32,
) -> Result<HashMap<Prefix, HashMap<TokenId, Count>>, DynError> {
    let mut model = HashMap::with_capacity(section.records.len());
    let mut expected_edge_start = 0_usize;
    let mut previous_prefix: Option<&[TokenId]> = None;

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
        validate_prefix(record.prefix.as_slice(), token_count, "model record")?;

        if let Some(prev) = previous_prefix
            && prev >= record.prefix.as_slice()
        {
            return Err(format!(
                "model{} records must be sorted by unique prefix",
                section.order
            )
            .into());
        }

        let edge_start = usize_from_u32(record.edge_start, "model edge start")?;
        if edge_start != expected_edge_start {
            return Err(format!(
                "model{} edge_start mismatch: expected {expected_edge_start}, got {edge_start}",
                section.order
            )
            .into());
        }
        let edge_len = usize_from_u32(record.edge_len, "model edge len")?;
        let edge_end = edge_start
            .checked_add(edge_len)
            .ok_or("model edge range overflow")?;
        let edge_slice = section
            .edges
            .get(edge_start..edge_end)
            .ok_or_else(|| format!("model{} edge range is invalid", section.order))?;

        let mut edges = HashMap::with_capacity(edge_slice.len());
        let mut previous_cumulative = Count::ZERO;
        let mut previous_next: Option<TokenId> = None;

        for edge in edge_slice {
            validate_token_id(edge.next.get(), token_count, "model edge")?;
            if let Some(prev_next) = previous_next
                && prev_next >= edge.next
            {
                return Err(format!(
                    "model{} edges must be sorted by unique token id",
                    section.order
                )
                .into());
            }
            if edge.cumulative.get() <= previous_cumulative.get() {
                return Err(format!(
                    "model{} edges must have strictly increasing cumulative counts",
                    section.order
                )
                .into());
            }

            edges.insert(edge.next, Count::new(edge.cumulative.get() - previous_cumulative.get()));
            previous_cumulative = edge.cumulative;
            previous_next = Some(edge.next);
        }

        if previous_cumulative.get() != record.total.get() {
            return Err(format!(
                "model{} total mismatch: expected {}, got {}",
                section.order, previous_cumulative.get(), record.total.get()
            )
            .into());
        }

        model.insert(record.prefix.clone(), edges);
        expected_edge_start = edge_end;
        previous_prefix = Some(record.prefix.as_slice());
    }

    if expected_edge_start != section.edges.len() {
        return Err(format!("model{} edges contain trailing data", section.order).into());
    }

    Ok(model)
}

fn decode_vocab(offsets: &[u64], blob: &[u8]) -> Result<Vec<String>, DynError> {
    if offsets.is_empty() {
        return Err("vocab offsets are empty".into());
    }

    let mut tokens = Vec::with_capacity(offsets.len().saturating_sub(1));
    for pair in offsets.windows(2) {
        let [start_offset, end_offset] = <&[u64; 2]>::try_from(pair)
            .map_err(|_error| "vocab offset pair must contain two values")?;
        let start = usize_from_u64(*start_offset, "vocab token start")?;
        let end = usize_from_u64(*end_offset, "vocab token end")?;
        let token_bytes = blob.get(start..end).ok_or("vocab token range is invalid")?;
        let token = str::from_utf8(token_bytes)
            .map_err(|_error| "vocab token is not valid UTF-8")?
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
        let [start_offset, end_offset] = <&[u64; 2]>::try_from(pair)
            .map_err(|_error| "vocab offset pair must contain two values")?;
        if start_offset > end_offset {
            return Err("vocab offsets must be non-decreasing".into());
        }
    }

    Ok(())
}

fn validate_prefix(prefix: &[TokenId], token_count: u32, context: &str) -> Result<(), DynError> {
    for token_id in prefix {
        validate_token_id(token_id.get(), token_count, context)?;
    }

    Ok(())
}

fn decode_vocab_blob(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
    flags: u32,
) -> Result<Vec<u8>, DynError> {
    let compression_flags = vocab_blob_compression_flags(flags)?;

    if compression_flags == FLAG_VOCAB_BLOB_RLE {
        decode_vocab_blob_rle(vocab_blob_bytes, expected_size)
    } else if compression_flags == FLAG_VOCAB_BLOB_ZSTD {
        decode_vocab_blob_zstd(vocab_blob_bytes, expected_size)
    } else if compression_flags == 0 {
        decode_vocab_blob_plain(vocab_blob_bytes, expected_size)
    } else {
        Err("unsupported vocab blob compression flags".into())
    }
}

fn decode_vocab_blob_plain(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, DynError> {
    if vocab_blob_bytes.len() != expected_size {
        return Err(format!(
            "vocab blob size mismatch: expected {expected_size}, got {}",
            vocab_blob_bytes.len()
        )
        .into());
    }

    Ok(vocab_blob_bytes.to_vec())
}

fn decode_vocab_blob_rle(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, DynError> {
    validate_rle_expected_size(vocab_blob_bytes.len(), expected_size)?;

    let mut decoded = Vec::with_capacity(expected_size);
    let mut cursor = 0_usize;

    while decoded.len() < expected_size {
        let control = *vocab_blob_bytes
            .get(cursor)
            .ok_or("compressed vocab blob is truncated")?;
        cursor += 1;

        if control < REPEAT_CONTROL_THRESHOLD {
            decode_literal_chunk(
                vocab_blob_bytes,
                control,
                &mut cursor,
                &mut decoded,
                expected_size,
            )?;
        } else {
            decode_repeat_chunk(
                vocab_blob_bytes,
                control,
                &mut cursor,
                &mut decoded,
                expected_size,
            )?;
        }
    }

    if cursor != vocab_blob_bytes.len() {
        return Err("compressed vocab blob has trailing bytes".into());
    }

    Ok(decoded)
}

fn decode_vocab_blob_zstd(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, DynError> {
    let decoded = zstd::bulk::decompress(vocab_blob_bytes, expected_size)?;
    if decoded.len() != expected_size {
        return Err("zstd vocab blob size does not match expected decoded size".into());
    }

    Ok(decoded)
}

fn validate_rle_expected_size(encoded_size: usize, expected_size: usize) -> Result<(), DynError> {
    let max_decoded_size = encoded_size
        .checked_mul(MAX_RLE_EXPANSION_PER_ENCODED_BYTE)
        .ok_or("compressed vocab blob expansion bound overflow")?;
    if expected_size > max_decoded_size {
        return Err("compressed vocab blob decoded size exceeds supported expansion bound".into());
    }

    Ok(())
}

fn decode_literal_chunk(
    source: &[u8],
    control: u8,
    cursor: &mut usize,
    decoded: &mut Vec<u8>,
    expected_size: usize,
) -> Result<(), DynError> {
    let literal_len = usize::from(control) + 1;
    let end = cursor
        .checked_add(literal_len)
        .ok_or("compressed vocab literal range overflow")?;

    let chunk = source
        .get(*cursor..end)
        .ok_or("compressed vocab blob literal chunk is truncated")?;

    append_chunk(
        decoded,
        chunk,
        expected_size,
        "compressed vocab blob literal",
    )?;
    *cursor = end;

    Ok(())
}

fn decode_repeat_chunk(
    source: &[u8],
    control: u8,
    cursor: &mut usize,
    decoded: &mut Vec<u8>,
    expected_size: usize,
) -> Result<(), DynError> {
    let repeat_len = usize::from(control - REPEAT_CONTROL_THRESHOLD) + REPEAT_CHUNK_MIN;
    let value = *source
        .get(*cursor)
        .ok_or("compressed vocab blob repeat chunk is truncated")?;
    *cursor += 1;

    let next_size = decoded
        .len()
        .checked_add(repeat_len)
        .ok_or("compressed vocab blob size overflow")?;
    if next_size > expected_size {
        return Err("compressed vocab blob repeat exceeds expected decoded size".into());
    }

    decoded.resize(next_size, value);

    Ok(())
}

fn append_chunk(
    decoded: &mut Vec<u8>,
    chunk: &[u8],
    expected_size: usize,
    context: &str,
) -> Result<(), DynError> {
    let next_size = decoded
        .len()
        .checked_add(chunk.len())
        .ok_or("compressed vocab blob size overflow")?;
    if next_size > expected_size {
        return Err(format!("{context} exceeds expected decoded size").into());
    }

    decoded.extend_from_slice(chunk);
    Ok(())
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

fn bytes_for_count(count: usize, element_size: u64, context: &str) -> Result<u64, DynError> {
    let count = u64_from_usize(count, context)?;
    count
        .checked_mul(element_size)
        .ok_or_else(|| format!("{context} byte size overflow").into())
}
