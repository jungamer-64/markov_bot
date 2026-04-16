use std::fs;

use markov_core::{MarkovChain, NgramOrder};
use tempfile::{Builder, TempPath};

use crate::{
    CHECKSUM_OFFSET, DESCRIPTOR_SIZE, HEADER_SIZE, SECTION_METADATA_COUNT, StorageCompressionMode,
    StorageError, compute_checksum, decode_chain, descriptor_count_for_ngram_order,
    encode_chain,
};

pub(super) const TEST_MIN_EDGE_COUNT: u64 = 1;
pub(super) const TEST_COMPRESSION_MODE: StorageCompressionMode = StorageCompressionMode::Auto;

pub(super) const VERSION_OFFSET: usize = 8;
pub(super) const FLAGS_OFFSET: usize = 12;
pub(super) const NGRAM_ORDER_OFFSET: usize = 24;
pub(super) const SECTION_COUNT_OFFSET: usize = 28;
pub(super) const UNSUPPORTED_FLAG: u32 = 1 << 31;

#[derive(Debug, Clone, Copy)]
pub(super) struct DescriptorView {
    pub(super) kind: u32,
    pub(super) flags: u32,
    pub(super) offset: u64,
    pub(super) size: u64,
}

pub(super) fn ensure(condition: bool, message: &str) -> Result<(), StorageError> {
    if condition {
        Ok(())
    } else {
        Err(StorageError::Format(message.to_owned()))
    }
}

pub(super) fn ensure_eq<L, R>(left: &L, right: &R, message: &str) -> Result<(), StorageError>
where
    L: PartialEq<R> + ?Sized,
    R: ?Sized,
{
    ensure(left == right, message)
}

pub(super) fn ensure_ne<L, R>(left: &L, right: &R, message: &str) -> Result<(), StorageError>
where
    L: PartialEq<R> + ?Sized,
    R: ?Sized,
{
    ensure(left != right, message)
}

pub(super) fn write_sample_file(
    prefix: &str,
    chain: &MarkovChain,
) -> Result<TempPath, StorageError> {
    write_sample_file_with_settings(prefix, chain, TEST_MIN_EDGE_COUNT, TEST_COMPRESSION_MODE)
}

pub(super) fn write_sample_file_with_mode(
    prefix: &str,
    chain: &MarkovChain,
    compression_mode: StorageCompressionMode,
) -> Result<TempPath, StorageError> {
    write_sample_file_with_settings(prefix, chain, TEST_MIN_EDGE_COUNT, compression_mode)
}

pub(super) fn write_sample_file_with_settings(
    prefix: &str,
    chain: &MarkovChain,
    min_edge_count: u64,
    compression_mode: StorageCompressionMode,
) -> Result<TempPath, StorageError> {
    let file_path = temp_file_path(prefix)?;
    let payload = encode_chain(chain, markov_core::Count::new(min_edge_count), compression_mode)?;
    fs::write(&file_path, payload)?;
    Ok(file_path)
}

pub(super) fn load_sample_file(
    path: &TempPath,
    expected_ngram_order: NgramOrder,
) -> Result<MarkovChain, StorageError> {
    let bytes = fs::read(path)?;
    decode_chain(bytes.as_slice(), expected_ngram_order)
}

pub(super) fn sample_chain_with_order(ngram_order: NgramOrder) -> Result<MarkovChain, StorageError> {
    let mut chain = MarkovChain::new(ngram_order)?;
    for tokens in [
        vec!["a", "b", "c", "d"],
        vec!["a", "b", "x"],
        vec!["a"],
        vec!["b", "c"],
    ] {
        chain.train_tokens(
            &tokens
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
        )?;
    }
    ensure(
        chain.models().len() == ngram_order.as_usize()?,
        "model count should match ngram order",
    )?;
    Ok(chain)
}

pub(super) fn temp_file_path(prefix: &str) -> Result<TempPath, StorageError> {
    let file = Builder::new()
        .prefix(&format!("markov_bot_{prefix}_"))
        .suffix(".mkv3")
        .tempfile()?;
    Ok(file.into_temp_path())
}

pub(super) fn descriptor(bytes: &[u8], index: usize) -> Result<DescriptorView, StorageError> {
    let offset = descriptor_start_offset(index)?;
    Ok(DescriptorView {
        kind: read_u32_at(bytes, offset)?,
        flags: read_u32_at(bytes, offset + 4)?,
        offset: read_u64_at(bytes, offset + 8)?,
        size: read_u64_at(bytes, offset + 16)?,
    })
}

pub(super) fn descriptor_count(bytes: &[u8]) -> Result<usize, StorageError> {
    let section_count = read_u64_at(bytes, SECTION_COUNT_OFFSET)?;
    usize::try_from(section_count).map_err(|_error| StorageError::Format("section count should fit usize".to_owned()))
}

pub(super) fn model_descriptor_index(bytes: &[u8], order: usize) -> Result<usize, StorageError> {
    let expected_flags =
        u32::try_from(order).map_err(|_error| StorageError::Format("order exceeds u32 range".to_owned()))?;
    let count = descriptor_count(bytes)?;

    for index in 0..count {
        let view = descriptor(bytes, index)?;
        if view.kind == 4 && view.flags == expected_flags {
            return Ok(index);
        }
    }

    Err(StorageError::Format(format!("descriptor not found for model order {order}")))
}

pub(super) fn section_body_offset(bytes: &[u8], index: usize) -> Result<usize, StorageError> {
    usize::try_from(descriptor(bytes, index)?.offset)
        .map_err(|_error| StorageError::Format("section offset should fit usize".to_owned()))
}

pub(super) fn descriptor_flags_offset(index: usize) -> Result<usize, StorageError> {
    let offset = descriptor_start_offset(index)?;
    offset.checked_add(4)
        .ok_or_else(|| StorageError::Format("descriptor flags offset overflow".to_owned()))
}

pub(super) fn descriptor_size_offset(index: usize) -> Result<usize, StorageError> {
    let offset = descriptor_start_offset(index)?;
    offset.checked_add(16)
        .ok_or_else(|| StorageError::Format("descriptor size offset overflow".to_owned()))
}

pub(super) fn rewrite_checksum(bytes: &mut [u8]) -> Result<(), StorageError> {
    let checksum = compute_checksum(bytes)?;
    write_u64_at(bytes, CHECKSUM_OFFSET, checksum)
}

pub(super) fn expected_section_count(order: usize) -> Result<u64, StorageError> {
    descriptor_count_for_ngram_order(order)
}

pub(super) fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, StorageError> {
    let end = offset + 4;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| StorageError::Format("u32 read range must be within buffer".to_owned()))?;
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(slice);
    Ok(u32::from_le_bytes(raw))
}

pub(super) fn write_u32_at(
    bytes: &mut [u8],
    offset: usize,
    value: u32,
) -> Result<(), StorageError> {
    let end = offset + 4;
    let target = bytes
        .get_mut(offset..end)
        .ok_or_else(|| StorageError::Format("u32 write range must be within buffer".to_owned()))?;
    target.copy_from_slice(value.to_le_bytes().as_slice());
    Ok(())
}

pub(super) fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, StorageError> {
    let end = offset + 8;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| StorageError::Format("u64 read range must be within buffer".to_owned()))?;
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(slice);
    Ok(u64::from_le_bytes(raw))
}

pub(super) fn write_u64_at(
    bytes: &mut [u8],
    offset: usize,
    value: u64,
) -> Result<(), StorageError> {
    let end = offset + 8;
    let target = bytes
        .get_mut(offset..end)
        .ok_or_else(|| StorageError::Format("u64 write range must be within buffer".to_owned()))?;
    target.copy_from_slice(value.to_le_bytes().as_slice());
    Ok(())
}

fn descriptor_start_offset(index: usize) -> Result<usize, StorageError> {
    HEADER_SIZE
        .checked_add(
            index
                .checked_mul(DESCRIPTOR_SIZE)
                .ok_or_else(|| StorageError::Format("descriptor start offset overflow".to_owned()))?,
        )
        .ok_or_else(|| StorageError::Format("descriptor start offset overflow".to_owned()))
}

#[test]
fn section_count_formula_stays_dynamic() -> Result<(), StorageError> {
    ensure(
        expected_section_count(7)? == SECTION_METADATA_COUNT + 7,
        "section count should be 3 + ngram_order",
    )
}
