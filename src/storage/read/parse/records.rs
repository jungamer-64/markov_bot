use super::super::super::{DynError, FixedRecord, usize_from_u64};

pub(super) fn parse_u64_section(bytes: &[u8], context: &str) -> Result<Vec<u64>, DynError> {
    if !bytes.len().is_multiple_of(8) {
        return Err(format!("{context} section size is not a multiple of 8 bytes").into());
    }

    let mut values = Vec::with_capacity(bytes.len() / 8);
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        values.push(read_u64(bytes, &mut cursor)?);
    }

    Ok(values)
}

pub(super) fn parse_fixed_section<T: FixedRecord>(
    bytes: &[u8],
    context: &str,
) -> Result<Vec<T>, DynError> {
    let record_size = usize_from_u64(T::SIZE, context)?;
    if !bytes.len().is_multiple_of(record_size) {
        return Err(format!("{context} section size is not a multiple of record size").into());
    }

    let mut records = Vec::with_capacity(bytes.len() / record_size);
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        records.push(T::decode_from(bytes, &mut cursor)?);
    }

    Ok(records)
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, DynError> {
    let end = cursor.checked_add(8).ok_or("cursor overflow")?;
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(
        bytes
            .get(*cursor..end)
            .ok_or("unexpected EOF while parsing u64 section")?,
    );
    *cursor = end;
    Ok(u64::from_le_bytes(raw))
}
