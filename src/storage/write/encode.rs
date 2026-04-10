use super::super::{CompiledStorage, DynError, HEADER_SIZE, compute_checksum};

mod header;
mod sections;

pub(super) fn encode_storage(compiled: &CompiledStorage) -> Result<Vec<u8>, DynError> {
    let mut header = header::build_header(compiled)?;
    let mut bytes = sections::write_sections(compiled, header)?;

    let header_without_checksum = header::encode_header(header);
    bytes[..HEADER_SIZE].copy_from_slice(header_without_checksum.as_slice());

    header.checksum = compute_checksum(bytes.as_slice())?;

    let header_bytes = header::encode_header(header);
    bytes[..HEADER_SIZE].copy_from_slice(header_bytes.as_slice());

    Ok(bytes)
}
