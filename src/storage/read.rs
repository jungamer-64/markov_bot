use super::{DynError, MarkovChain};

mod header;
mod parse;
mod rebuild;

pub(super) fn decode_chain(bytes: &[u8]) -> Result<MarkovChain, DynError> {
    let header = header::validate_header(bytes)?;
    let ranges = header::build_section_ranges(&header)?;
    let parsed = parse::parse_storage(bytes, &header, &ranges)?;

    rebuild::rebuild_chain(parsed)
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
