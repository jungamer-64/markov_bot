use super::super::{
    CHECKSUM, CompiledStorage, DynError, FLAGS, HEADER_SIZE, Header, MAGIC, NORMALIZATION_FLAGS,
    PAIR3_RECORD_SIZE, PREFIX1_RECORD_SIZE, PREFIX2_RECORD_SIZE, PREFIX3_RECORD_SIZE,
    START_RECORD_SIZE, SectionCounts, SectionSizes, TOKENIZER_VERSION, VERSION, align_to_eight,
    bytes_for_len, checked_add, u32_from_usize, u64_from_usize, usize_from_u64,
};

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
            super::super::EDGE_RECORD_SIZE,
            "model3 edges",
        )?,
        model2_prefixes: bytes_for_len(
            compiled.model2_prefixes.len(),
            PREFIX2_RECORD_SIZE,
            "model2 prefixes",
        )?,
        model2_edges: bytes_for_len(
            compiled.model2_edges.len(),
            super::super::EDGE_RECORD_SIZE,
            "model2 edges",
        )?,
        model1_prefixes: bytes_for_len(
            compiled.model1_prefixes.len(),
            PREFIX1_RECORD_SIZE,
            "model1 prefixes",
        )?,
        model1_edges: bytes_for_len(
            compiled.model1_edges.len(),
            super::super::EDGE_RECORD_SIZE,
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
