use super::super::{DynError, StorageCompressionMode, StorageSections, compute_checksum};

mod compression;
mod header;
mod sections;

pub(super) fn encode_storage(
    sections: &StorageSections,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    let encoded_vocab_blob =
        compression::encode_vocab_blob(sections.vocab.blob.as_slice(), compression_mode)?;
    let payloads = sections::build_section_payloads(sections, encoded_vocab_blob.bytes.as_slice())?;
    let (mut header, descriptors) =
        header::build_header(payloads.as_slice(), encoded_vocab_blob.flags)?;

    let metadata_without_checksum = header::encode_metadata(header, descriptors.as_slice());
    let mut bytes = sections::write_section_payloads(
        payloads.as_slice(),
        descriptors.as_slice(),
        header.file_size,
    )?;
    bytes[..metadata_without_checksum.len()].copy_from_slice(metadata_without_checksum.as_slice());

    header.checksum = compute_checksum(bytes.as_slice())?;

    let metadata = header::encode_metadata(header, descriptors.as_slice());
    bytes[..metadata.len()].copy_from_slice(metadata.as_slice());

    Ok(bytes)
}
