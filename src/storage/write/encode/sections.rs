use super::super::super::{
    DynError, FixedRecord, SectionDescriptor, SectionKind, StorageSections, aligned_metadata_end,
    fixed_section_bytes, usize_from_u64,
};

pub(super) struct SectionPayload {
    pub(super) kind: SectionKind,
    pub(super) bytes: Vec<u8>,
}

pub(super) fn build_section_payloads(
    sections: &StorageSections,
    encoded_vocab_blob: &[u8],
) -> Result<Vec<SectionPayload>, DynError> {
    Ok(vec![
        build_u64_payload(SectionKind::VocabOffsets, sections.vocab.offsets.as_slice())?,
        SectionPayload {
            kind: SectionKind::VocabBlob,
            bytes: encoded_vocab_blob.to_vec(),
        },
        build_fixed_payload(SectionKind::Starts, sections.starts.as_slice())?,
        build_fixed_payload(SectionKind::Model3Pairs, sections.model3.pairs.as_slice())?,
        build_fixed_payload(
            SectionKind::Model3Prefixes,
            sections.model3.prefixes.as_slice(),
        )?,
        build_fixed_payload(SectionKind::Model3Edges, sections.model3.edges.as_slice())?,
        build_fixed_payload(SectionKind::Model2Pairs, sections.model2.pairs.as_slice())?,
        build_fixed_payload(
            SectionKind::Model2Prefixes,
            sections.model2.prefixes.as_slice(),
        )?,
        build_fixed_payload(SectionKind::Model2Edges, sections.model2.edges.as_slice())?,
        build_fixed_payload(
            SectionKind::Model1Prefixes,
            sections.model1.prefixes.as_slice(),
        )?,
        build_fixed_payload(SectionKind::Model1Edges, sections.model1.edges.as_slice())?,
    ])
}

pub(super) fn write_section_payloads(
    payloads: &[SectionPayload],
    descriptors: &[SectionDescriptor],
    file_size: u64,
) -> Result<Vec<u8>, DynError> {
    if payloads.len() != descriptors.len() {
        return Err("section payload count does not match descriptor count".into());
    }

    let metadata_end = usize_from_u64(aligned_metadata_end(descriptors.len())?, "metadata size")?;
    let mut bytes = vec![0; metadata_end];

    for (payload, descriptor) in payloads.iter().zip(descriptors.iter()) {
        if descriptor.kind != payload.kind.as_u32() {
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

fn build_fixed_payload<T: FixedRecord>(
    kind: SectionKind,
    records: &[T],
) -> Result<SectionPayload, DynError> {
    let capacity = usize_from_u64(fixed_section_bytes(records, kind.label())?, kind.label())?;
    let mut bytes = Vec::with_capacity(capacity);
    for record in records {
        record.encode_into(&mut bytes);
    }

    Ok(SectionPayload { kind, bytes })
}

fn build_u64_payload(kind: SectionKind, values: &[u64]) -> Result<SectionPayload, DynError> {
    let capacity = usize_from_u64(
        super::super::super::bytes_for_len(values.len(), 8, kind.label())?,
        kind.label(),
    )?;
    let mut bytes = Vec::with_capacity(capacity);
    for value in values {
        bytes.extend_from_slice(value.to_le_bytes().as_slice());
    }

    Ok(SectionPayload { kind, bytes })
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
