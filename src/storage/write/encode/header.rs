use super::super::super::{
    CHECKSUM_PLACEHOLDER, DESCRIPTOR_SIZE, DynError, FLAGS, HEADER_SIZE, Header, MAGIC,
    NORMALIZATION_FLAGS, SECTION_COUNT, SECTION_COUNT_U32, SectionDescriptor, SectionKind,
    TOKENIZER_VERSION, VERSION, align_to_eight, aligned_metadata_end, checked_add, u64_from_usize,
};
use super::sections::SectionPayload;

pub(super) fn build_header(
    payloads: &[SectionPayload],
    flags: u32,
) -> Result<(Header, Vec<SectionDescriptor>), DynError> {
    if payloads.len() != SECTION_COUNT {
        return Err(format!(
            "expected {SECTION_COUNT} section payloads, got {}",
            payloads.len()
        )
        .into());
    }

    let mut offset = aligned_metadata_end(payloads.len())?;
    let mut file_size = offset;
    let mut descriptors = Vec::with_capacity(payloads.len());

    for (index, payload) in payloads.iter().enumerate() {
        let expected_kind = SectionKind::ALL[index];
        if payload.kind != expected_kind {
            return Err(format!(
                "section payload order mismatch: expected {}, got {}",
                expected_kind.label(),
                payload.kind.label()
            )
            .into());
        }

        let size = u64_from_usize(payload.bytes.len(), payload.kind.label())?;
        descriptors.push(SectionDescriptor {
            kind: payload.kind.as_u32(),
            flags: FLAGS,
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
            section_count: SECTION_COUNT_U32,
            file_size,
            checksum: CHECKSUM_PLACEHOLDER,
        },
        descriptors,
    ))
}

pub(super) fn encode_metadata(header: Header, descriptors: &[SectionDescriptor]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_SIZE + descriptors.len() * DESCRIPTOR_SIZE);

    bytes.extend_from_slice(header.magic.as_slice());
    write_u32(&mut bytes, header.version);
    write_u32(&mut bytes, header.flags);
    write_u32(&mut bytes, header.tokenizer_version);
    write_u32(&mut bytes, header.normalization_flags);
    write_u32(&mut bytes, header.section_count);
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

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}
