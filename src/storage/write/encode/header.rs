use super::super::super::{
    CHECKSUM_PLACEHOLDER, CompiledStorage, DynError, FLAGS, HEADER_SIZE, Header, MAGIC,
    NORMALIZATION_FLAGS, PAIR3_RECORD_SIZE, PREFIX1_RECORD_SIZE, PREFIX2_RECORD_SIZE,
    PREFIX3_RECORD_SIZE, START_RECORD_SIZE, SectionCounts, SectionSizes, TOKENIZER_VERSION,
    VERSION, align_to_eight, bytes_for_len, checked_add, u32_from_usize, u64_from_usize,
};

pub(super) fn build_header(compiled: &CompiledStorage) -> Result<Header, DynError> {
    let counts = section_counts(compiled)?;
    let sizes = section_sizes(compiled)?;

    let mut offset = align_to_eight(u64_from_usize(HEADER_SIZE, "header size")?);

    let vocab_offsets_offset =
        advance_with_alignment(&mut offset, sizes.vocab_offsets, "vocab offsets end")?;
    let vocab_blob_offset =
        advance_with_alignment(&mut offset, sizes.vocab_blob, "vocab blob end")?;
    let start_offset = advance_with_alignment(&mut offset, sizes.starts, "start records end")?;
    let model3_pair_offset =
        advance_with_alignment(&mut offset, sizes.model3_pairs, "model3 pair records end")?;
    let model3_prefix_offset = advance_with_alignment(
        &mut offset,
        sizes.model3_prefixes,
        "model3 prefix records end",
    )?;
    let model3_edge_offset =
        advance_with_alignment(&mut offset, sizes.model3_edges, "model3 edge records end")?;
    let model2_prefix_offset = advance_with_alignment(
        &mut offset,
        sizes.model2_prefixes,
        "model2 prefix records end",
    )?;
    let model2_edge_offset =
        advance_with_alignment(&mut offset, sizes.model2_edges, "model2 edge records end")?;
    let model1_prefix_offset = advance_with_alignment(
        &mut offset,
        sizes.model1_prefixes,
        "model1 prefix records end",
    )?;

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
        checksum: CHECKSUM_PLACEHOLDER,
    })
}

pub(super) fn encode_header(header: Header) -> Vec<u8> {
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
            super::super::super::EDGE_RECORD_SIZE,
            "model3 edges",
        )?,
        model2_prefixes: bytes_for_len(
            compiled.model2_prefixes.len(),
            PREFIX2_RECORD_SIZE,
            "model2 prefixes",
        )?,
        model2_edges: bytes_for_len(
            compiled.model2_edges.len(),
            super::super::super::EDGE_RECORD_SIZE,
            "model2 edges",
        )?,
        model1_prefixes: bytes_for_len(
            compiled.model1_prefixes.len(),
            PREFIX1_RECORD_SIZE,
            "model1 prefixes",
        )?,
        model1_edges: bytes_for_len(
            compiled.model1_edges.len(),
            super::super::super::EDGE_RECORD_SIZE,
            "model1 edges",
        )?,
    })
}

fn advance_with_alignment(offset: &mut u64, size: u64, context: &str) -> Result<u64, DynError> {
    let current = *offset;
    *offset = align_to_eight(checked_add(*offset, size, context)?);
    Ok(current)
}

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}
