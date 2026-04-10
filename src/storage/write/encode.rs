use super::super::{
    CompiledStorage, DynError, HEADER_SIZE, StorageCompressionMode, compute_checksum,
};

mod compression;
mod header;
mod sections;

pub(super) fn encode_storage(
    compiled: &CompiledStorage,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    let encoded_vocab_blob =
        compression::encode_vocab_blob(compiled.vocab_blob.as_slice(), compression_mode);

    let mut header = header::build_header(
        compiled,
        encoded_vocab_blob.bytes.len(),
        encoded_vocab_blob.flags,
    )?;
    let mut bytes =
        sections::write_sections(compiled, encoded_vocab_blob.bytes.as_slice(), header)?;

    let header_without_checksum = header::encode_header(header);
    bytes[..HEADER_SIZE].copy_from_slice(header_without_checksum.as_slice());

    header.checksum = compute_checksum(bytes.as_slice())?;

    let header_bytes = header::encode_header(header);
    bytes[..HEADER_SIZE].copy_from_slice(header_bytes.as_slice());

    Ok(bytes)
}
