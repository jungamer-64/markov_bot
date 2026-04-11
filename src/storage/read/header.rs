use super::super::{
    DynError, HEADER_SIZE, Header, MAGIC, NORMALIZATION_FLAGS, SECTION_COUNT_U32, SUPPORTED_FLAGS,
    SectionDescriptor, SectionEntry, SectionKind, SectionTable, TOKENIZER_VERSION, VERSION,
    aligned_metadata_end, compute_checksum, u64_from_usize, usize_from_u32, usize_from_u64,
    vocab_blob_compression_flags,
};
use super::{read_exact, read_u32_value, read_u64_value};

pub(super) fn validate_header(bytes: &[u8]) -> Result<Header, DynError> {
    let header = decode_header(bytes)?;

    if header.magic != MAGIC {
        return Err("invalid magic".into());
    }
    if header.version != VERSION {
        return Err(format!("unsupported version: {}", header.version).into());
    }
    if header.flags & !SUPPORTED_FLAGS != 0 {
        return Err(format!("unsupported flags: {}", header.flags).into());
    }
    let _ = vocab_blob_compression_flags(header.flags)?;
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
    if header.section_count != SECTION_COUNT_U32 {
        return Err(format!("unsupported section_count: {}", header.section_count).into());
    }

    let file_size = u64_from_usize(bytes.len(), "file size")?;
    if header.file_size != file_size {
        return Err(format!(
            "file_size mismatch: header={}, actual={file_size}",
            header.file_size
        )
        .into());
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

pub(super) fn build_section_table(bytes: &[u8], header: &Header) -> Result<SectionTable, DynError> {
    let section_count = usize_from_u32(header.section_count, "section count")?;
    let descriptors = decode_descriptors(bytes, header.section_count)?;
    let metadata_end = usize_from_u64(aligned_metadata_end(section_count)?, "metadata size")?;

    let mut entries: Vec<SectionEntry> = Vec::with_capacity(section_count);
    let mut cursor = metadata_end;

    for (index, descriptor) in descriptors.into_iter().enumerate() {
        let kind = SectionKind::from_u32(descriptor.kind)
            .ok_or_else(|| format!("unknown section kind: {}", descriptor.kind))?;
        let expected_kind = SectionKind::ALL
            .get(index)
            .copied()
            .ok_or("section descriptors exceed canonical section count")?;
        if kind != expected_kind {
            return Err(format!(
                "section descriptors are not in canonical order: expected {}, got {}",
                expected_kind.label(),
                kind.label()
            )
            .into());
        }
        if descriptor.flags != 0 {
            return Err(format!(
                "unsupported {} descriptor flags: {}",
                kind.label(),
                descriptor.flags
            )
            .into());
        }
        if descriptor.offset % 8 != 0 {
            return Err(format!("{} section offset must be 8-byte aligned", kind.label()).into());
        }

        let range = section_range(
            descriptor.offset,
            descriptor.size,
            header.file_size,
            kind.label(),
        )?;
        if range.start < metadata_end {
            return Err(format!(
                "{} section starts before aligned metadata end",
                kind.label()
            )
            .into());
        }
        if range.start < cursor {
            return Err(format!("{} section overlaps previous section", kind.label()).into());
        }

        let padding_context = match entries.last() {
            Some(entry) => {
                let kind = SectionKind::from_u32(entry.descriptor.kind)
                    .ok_or("validated descriptor kinds must remain known")?;
                kind.label()
            }
            None => "metadata",
        };
        validate_zero_padding(bytes, cursor, range.start, padding_context)?;

        cursor = range.end;
        entries.push(SectionEntry { descriptor, range });
    }

    let file_size = usize_from_u64(header.file_size, "file size")?;
    if cursor != file_size {
        return Err("final section must end at file_size".into());
    }

    Ok(SectionTable { entries })
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

    Ok(Header {
        magic,
        version: read_u32_value(bytes, &mut cursor)?,
        flags: read_u32_value(bytes, &mut cursor)?,
        tokenizer_version: read_u32_value(bytes, &mut cursor)?,
        normalization_flags: read_u32_value(bytes, &mut cursor)?,
        section_count: read_u32_value(bytes, &mut cursor)?,
        file_size: read_u64_value(bytes, &mut cursor)?,
        checksum: read_u64_value(bytes, &mut cursor)?,
    })
}

fn decode_descriptors(
    bytes: &[u8],
    section_count: u32,
) -> Result<Vec<SectionDescriptor>, DynError> {
    let descriptor_count = usize_from_u32(section_count, "section count")?;
    let mut cursor = HEADER_SIZE;
    let mut descriptors = Vec::with_capacity(descriptor_count);

    for _ in 0..descriptor_count {
        descriptors.push(SectionDescriptor {
            kind: read_u32_value(bytes, &mut cursor)?,
            flags: read_u32_value(bytes, &mut cursor)?,
            offset: read_u64_value(bytes, &mut cursor)?,
            size: read_u64_value(bytes, &mut cursor)?,
        });
    }

    let metadata_end = usize_from_u64(aligned_metadata_end(descriptor_count)?, "metadata size")?;
    let _ = bytes
        .get(..metadata_end)
        .ok_or("metadata extends beyond file_size")?;

    Ok(descriptors)
}

fn section_range(
    offset: u64,
    size: u64,
    file_size: u64,
    context: &str,
) -> Result<std::ops::Range<usize>, DynError> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| format!("{context} range overflow"))?;
    if end > file_size {
        return Err(format!("{context} exceeds file_size").into());
    }

    Ok(usize_from_u64(offset, context)?..usize_from_u64(end, context)?)
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
