use super::super::super::{CompiledStorage, DynError, HEADER_SIZE, Header, usize_from_u64};

pub(super) fn write_sections(
    compiled: &CompiledStorage,
    header: Header,
) -> Result<Vec<u8>, DynError> {
    let mut bytes = vec![0; HEADER_SIZE];

    write_vocab_offsets(&mut bytes, compiled, header)?;
    write_vocab_blob(&mut bytes, compiled, header)?;
    write_starts(&mut bytes, compiled, header)?;
    write_model3_pairs(&mut bytes, compiled, header)?;
    write_model3_prefixes(&mut bytes, compiled, header)?;
    write_model3_edges(&mut bytes, compiled, header)?;
    write_model2_prefixes(&mut bytes, compiled, header)?;
    write_model2_edges(&mut bytes, compiled, header)?;
    write_model1_prefixes(&mut bytes, compiled, header)?;
    write_model1_edges(&mut bytes, compiled, header)?;

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

fn write_vocab_offsets(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.vocab_offsets_offset, |target| {
        for offset in &compiled.vocab_offsets {
            write_u64(target, *offset);
        }
    })
}

fn write_vocab_blob(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.vocab_blob_offset, |target| {
        target.extend_from_slice(compiled.vocab_blob.as_slice());
    })
}

fn write_starts(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.start_offset, |target| {
        for record in &compiled.starts {
            write_u32(target, record.prefix_id);
            write_u32(target, record.cumulative);
        }
    })
}

fn write_model3_pairs(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model3_pair_offset, |target| {
        for record in &compiled.model3_pairs {
            write_u32(target, record.w1);
            write_u32(target, record.w2);
            write_u32(target, record.prefix_start);
            write_u32(target, record.prefix_len);
        }
    })
}

fn write_model3_prefixes(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model3_prefix_offset, |target| {
        for record in &compiled.model3_prefixes {
            write_u32(target, record.w3);
            write_u32(target, record.edge_start);
            write_u32(target, record.edge_len);
            write_u32(target, record.total);
        }
    })
}

fn write_model3_edges(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model3_edge_offset, |target| {
        for record in &compiled.model3_edges {
            write_u32(target, record.next);
            write_u32(target, record.cumulative);
        }
    })
}

fn write_model2_prefixes(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model2_prefix_offset, |target| {
        for record in &compiled.model2_prefixes {
            write_u32(target, record.w1);
            write_u32(target, record.w2);
            write_u32(target, record.edge_start);
            write_u32(target, record.edge_len);
            write_u32(target, record.total);
        }
    })
}

fn write_model2_edges(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model2_edge_offset, |target| {
        for record in &compiled.model2_edges {
            write_u32(target, record.next);
            write_u32(target, record.cumulative);
        }
    })
}

fn write_model1_prefixes(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model1_prefix_offset, |target| {
        for record in &compiled.model1_prefixes {
            write_u32(target, record.w1);
            write_u32(target, record.edge_start);
            write_u32(target, record.edge_len);
            write_u32(target, record.total);
        }
    })
}

fn write_model1_edges(
    bytes: &mut Vec<u8>,
    compiled: &CompiledStorage,
    header: Header,
) -> Result<(), DynError> {
    write_at_offset(bytes, header.model1_edge_offset, |target| {
        for record in &compiled.model1_edges {
            write_u32(target, record.next);
            write_u32(target, record.cumulative);
        }
    })
}

fn write_at_offset<F>(target: &mut Vec<u8>, offset: u64, writer: F) -> Result<(), DynError>
where
    F: FnOnce(&mut Vec<u8>),
{
    pad_to_offset(target, offset)?;
    writer(target);
    Ok(())
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
