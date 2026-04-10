use super::super::{
    DynError, EDGE_RECORD_SIZE, FLAGS, HEADER_SIZE, Header, MAGIC, NORMALIZATION_FLAGS,
    PAIR2_RECORD_SIZE, PAIR3_RECORD_SIZE, PREFIX1_RECORD_SIZE, PREFIX2_RECORD_SIZE,
    PREFIX3_RECORD_SIZE, START_RECORD_SIZE, SectionRanges, TOKENIZER_VERSION, VERSION,
    align_to_eight, bytes_for_len, checked_add, compute_checksum, u64_from_usize, usize_from_u32,
    usize_from_u64,
};
use super::{read_exact, read_u32_value, read_u64_value};

struct ModelRanges {
    starts: std::ops::Range<usize>,
    model3_pairs: std::ops::Range<usize>,
    model3_prefixes: std::ops::Range<usize>,
    model3_edges: std::ops::Range<usize>,
    model2_pairs: std::ops::Range<usize>,
    model2_prefixes: std::ops::Range<usize>,
    model2_edges: std::ops::Range<usize>,
    model1_prefixes: std::ops::Range<usize>,
    model1_edges: std::ops::Range<usize>,
}

pub(super) fn validate_header(bytes: &[u8]) -> Result<Header, DynError> {
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
        header.model2_pair_offset,
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

    if ordered_offsets.iter().any(|offset| offset % 8 != 0) {
        return Err("section offsets must be 8-byte aligned".into());
    }

    let expected_checksum = compute_checksum(bytes)?;
    if header.checksum != expected_checksum {
        return Err(format!(
            "checksum mismatch: header={}, expected={expected_checksum}",
            header.checksum
        )
        .into());
    }

    Ok(header)
}

pub(super) fn build_section_ranges(
    bytes: &[u8],
    header: &Header,
) -> Result<SectionRanges, DynError> {
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

    let model_ranges = build_model_ranges(header)?;

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
        starts: model_ranges.starts,
        model3_pairs: model_ranges.model3_pairs,
        model3_prefixes: model_ranges.model3_prefixes,
        model3_edges: model_ranges.model3_edges,
        model2_pairs: model_ranges.model2_pairs,
        model2_prefixes: model_ranges.model2_prefixes,
        model2_edges: model_ranges.model2_edges,
        model1_prefixes: model_ranges.model1_prefixes,
        model1_edges: model_ranges.model1_edges,
    };

    validate_section_layout(bytes, header, &ranges)?;

    Ok(ranges)
}

fn build_model_ranges(header: &Header) -> Result<ModelRanges, DynError> {
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
    let model2_pairs = record_range(
        header.model2_pair_offset,
        header.model2_pair_count,
        PAIR2_RECORD_SIZE,
        header.file_size,
        "model2 pairs",
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

    Ok(ModelRanges {
        starts,
        model3_pairs,
        model3_prefixes,
        model3_edges,
        model2_pairs,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    })
}

fn validate_section_layout(
    bytes: &[u8],
    header: &Header,
    ranges: &SectionRanges,
) -> Result<(), DynError> {
    let aligned_header_end = aligned_header_end()?;

    ensure_vocab_offsets_after_header(ranges, aligned_header_end)?;
    validate_non_overlapping_sections(ranges)?;
    validate_padding_regions(bytes, ranges, aligned_header_end)?;
    validate_final_section_ends_at_file_size(header, ranges)?;

    Ok(())
}

fn aligned_header_end() -> Result<usize, DynError> {
    let aligned = align_to_eight(u64_from_usize(HEADER_SIZE, "header size")?);
    usize_from_u64(aligned, "aligned header size")
}

fn ensure_vocab_offsets_after_header(
    ranges: &SectionRanges,
    aligned_header_end: usize,
) -> Result<(), DynError> {
    if ranges.vocab_offsets.start < aligned_header_end {
        return Err("vocab offsets section starts before aligned header end".into());
    }

    Ok(())
}

fn validate_non_overlapping_sections(ranges: &SectionRanges) -> Result<(), DynError> {
    let checks = [
        (
            "vocab offsets",
            &ranges.vocab_offsets,
            "vocab blob",
            ranges.vocab_blob_area.start,
        ),
        (
            "vocab blob",
            &ranges.vocab_blob_area,
            "start records",
            ranges.starts.start,
        ),
        (
            "start records",
            &ranges.starts,
            "model3 pairs",
            ranges.model3_pairs.start,
        ),
        (
            "model3 pairs",
            &ranges.model3_pairs,
            "model3 prefixes",
            ranges.model3_prefixes.start,
        ),
        (
            "model3 prefixes",
            &ranges.model3_prefixes,
            "model3 edges",
            ranges.model3_edges.start,
        ),
        (
            "model3 edges",
            &ranges.model3_edges,
            "model2 pairs",
            ranges.model2_pairs.start,
        ),
        (
            "model2 pairs",
            &ranges.model2_pairs,
            "model2 prefixes",
            ranges.model2_prefixes.start,
        ),
        (
            "model2 prefixes",
            &ranges.model2_prefixes,
            "model2 edges",
            ranges.model2_edges.start,
        ),
        (
            "model2 edges",
            &ranges.model2_edges,
            "model1 prefixes",
            ranges.model1_prefixes.start,
        ),
        (
            "model1 prefixes",
            &ranges.model1_prefixes,
            "model1 edges",
            ranges.model1_edges.start,
        ),
    ];

    for (left_name, left_range, right_name, right_start) in checks {
        ensure_non_overlapping(left_name, left_range, right_name, right_start)?;
    }

    Ok(())
}

fn validate_padding_regions(
    bytes: &[u8],
    ranges: &SectionRanges,
    aligned_header_end: usize,
) -> Result<(), DynError> {
    let checks = [
        (aligned_header_end, ranges.vocab_offsets.start, "header"),
        (
            ranges.vocab_offsets.end,
            ranges.vocab_blob_area.start,
            "vocab offsets",
        ),
        (
            ranges.vocab_blob_area.end,
            ranges.starts.start,
            "vocab blob",
        ),
        (
            ranges.starts.end,
            ranges.model3_pairs.start,
            "start records",
        ),
        (
            ranges.model3_pairs.end,
            ranges.model3_prefixes.start,
            "model3 pairs",
        ),
        (
            ranges.model3_prefixes.end,
            ranges.model3_edges.start,
            "model3 prefixes",
        ),
        (
            ranges.model3_edges.end,
            ranges.model2_pairs.start,
            "model3 edges",
        ),
        (
            ranges.model2_pairs.end,
            ranges.model2_prefixes.start,
            "model2 pairs",
        ),
        (
            ranges.model2_prefixes.end,
            ranges.model2_edges.start,
            "model2 prefixes",
        ),
        (
            ranges.model2_edges.end,
            ranges.model1_prefixes.start,
            "model2 edges",
        ),
        (
            ranges.model1_prefixes.end,
            ranges.model1_edges.start,
            "model1 prefixes",
        ),
    ];

    for (start, end, context) in checks {
        validate_zero_padding(bytes, start, end, context)?;
    }

    Ok(())
}

fn validate_final_section_ends_at_file_size(
    header: &Header,
    ranges: &SectionRanges,
) -> Result<(), DynError> {
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

fn validate_zero_padding(
    bytes: &[u8],
    start: usize,
    end: usize,
    context: &str,
) -> Result<(), DynError> {
    if start > end {
        return Err(format!("padding range after {context} is invalid").into());
    }

    let slice = bytes
        .get(start..end)
        .ok_or_else(|| format!("padding range after {context} is out of bounds"))?;
    if slice.iter().any(|byte| *byte != 0) {
        return Err(format!("non-zero padding detected after {context} section").into());
    }

    Ok(())
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
    let model2_pair_count = read_u32_value(bytes, &mut cursor)?;
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
    let model2_pair_offset = read_u64_value(bytes, &mut cursor)?;
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
        model2_pair_count,
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
        model2_pair_offset,
        model2_prefix_offset,
        model2_edge_offset,
        model1_prefix_offset,
        model1_edge_offset,
        file_size,
        checksum,
    })
}
