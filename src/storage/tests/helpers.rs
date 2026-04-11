use std::future::Future;

use tempfile::{Builder, TempPath};

use crate::config::DynError;
use crate::markov::MarkovChain;

use super::super::{
    CHECKSUM_OFFSET, DESCRIPTOR_SIZE, HEADER_SIZE, SECTION_COUNT, SectionKind,
    StorageCompressionMode, aligned_metadata_end, compute_checksum, save_chain,
};

pub(super) const TEST_MIN_EDGE_COUNT: u64 = 1;
pub(super) const TEST_COMPRESSION_MODE: StorageCompressionMode = StorageCompressionMode::Auto;

pub(super) const VERSION_OFFSET: usize = 8;
pub(super) const FLAGS_OFFSET: usize = 12;
pub(super) const TOKENIZER_VERSION_OFFSET: usize = 16;
pub(super) const NORMALIZATION_FLAGS_OFFSET: usize = 20;
pub(super) const SECTION_COUNT_OFFSET: usize = 24;
pub(super) const FILE_SIZE_OFFSET: usize = 28;
pub(super) const UNSUPPORTED_FLAG: u32 = 1 << 31;

pub(super) struct DescriptorView {
    pub(super) offset: u64,
    pub(super) size: u64,
}

pub(super) fn descriptor(bytes: &[u8], kind: SectionKind) -> DescriptorView {
    let base = descriptor_base(kind);
    DescriptorView {
        offset: read_u64_at(bytes, base + 8).unwrap_or(0),
        size: read_u64_at(bytes, base + 16).unwrap_or(0),
    }
}

pub(super) fn descriptor_kind_offset(kind: SectionKind) -> usize {
    descriptor_base(kind)
}

pub(super) fn descriptor_offset_offset(kind: SectionKind) -> usize {
    descriptor_base(kind) + 8
}

pub(super) fn descriptor_size_offset(kind: SectionKind) -> usize {
    descriptor_base(kind) + 16
}

pub(super) fn section_body_offset(bytes: &[u8], kind: SectionKind) -> Result<usize, DynError> {
    usize::try_from(descriptor(bytes, kind).offset)
        .map_err(|_error| "section offset should fit usize".into())
}

pub(super) fn section_body_size(bytes: &[u8], kind: SectionKind) -> Result<usize, DynError> {
    usize::try_from(descriptor(bytes, kind).size)
        .map_err(|_error| "section size should fit usize".into())
}

pub(super) fn first_padding_offset(bytes: &[u8]) -> Result<Option<usize>, DynError> {
    let metadata_end = aligned_metadata_end(SECTION_COUNT)?;
    let mut cursor =
        usize::try_from(metadata_end).map_err(|_error| "metadata size should fit usize")?;

    for kind in SectionKind::ALL {
        let descriptor = descriptor(bytes, kind);
        let offset =
            usize::try_from(descriptor.offset).map_err(|_error| "offset should fit usize")?;
        if cursor < offset {
            return Ok(Some(cursor));
        }

        let size = usize::try_from(descriptor.size).map_err(|_error| "size should fit usize")?;
        let end = offset
            .checked_add(size)
            .ok_or("section end should fit usize")?;
        cursor = end;
    }

    Ok(None)
}

pub(super) fn first_fixed_section_with_gap(bytes: &[u8]) -> Result<Option<SectionKind>, DynError> {
    for pair in SectionKind::ALL.windows(2) {
        let [left, right] =
            <&[SectionKind; 2]>::try_from(pair).map_err(|_error| "section pair should be valid")?;
        if matches!(left, SectionKind::VocabOffsets | SectionKind::VocabBlob) {
            continue;
        }

        let left_descriptor = descriptor(bytes, *left);
        let left_end = left_descriptor
            .offset
            .checked_add(left_descriptor.size)
            .ok_or("left section end should fit u64")?;
        let right_start = descriptor(bytes, *right).offset;
        if left_end < right_start {
            return Ok(Some(*left));
        }
    }

    Ok(None)
}

pub(super) fn rewrite_checksum(bytes: &mut [u8]) -> Result<(), DynError> {
    let checksum = compute_checksum(bytes)?;
    write_u64_at(bytes, CHECKSUM_OFFSET, checksum)
}

pub(super) fn run_async_test<F>(future: F) -> Result<(), DynError>
where
    F: Future<Output = Result<(), DynError>>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(future)
}

pub(super) async fn write_sample_file(
    prefix: &str,
    chain: &MarkovChain,
) -> Result<TempPath, DynError> {
    write_sample_file_with_settings(prefix, chain, TEST_MIN_EDGE_COUNT, TEST_COMPRESSION_MODE).await
}

pub(super) async fn write_sample_file_with_mode(
    prefix: &str,
    chain: &MarkovChain,
    compression_mode: StorageCompressionMode,
) -> Result<TempPath, DynError> {
    write_sample_file_with_settings(prefix, chain, TEST_MIN_EDGE_COUNT, compression_mode).await
}

pub(super) async fn write_sample_file_with_settings(
    prefix: &str,
    chain: &MarkovChain,
    min_edge_count: u64,
    compression_mode: StorageCompressionMode,
) -> Result<TempPath, DynError> {
    let file_path = temp_file_path(prefix)?;
    save_chain(&file_path, chain, min_edge_count, compression_mode).await?;
    Ok(file_path)
}

pub(super) fn sample_chain() -> Result<MarkovChain, DynError> {
    let mut chain = MarkovChain::default();
    chain.train_tokens(&[
        "a".to_owned(),
        "b".to_owned(),
        "c".to_owned(),
        "d".to_owned(),
    ])?;
    chain.train_tokens(&["a".to_owned()])?;
    Ok(chain)
}

pub(super) fn temp_file_path(prefix: &str) -> Result<TempPath, DynError> {
    let file = Builder::new()
        .prefix(&format!("markov_bot_{prefix}_"))
        .suffix(".mkv3")
        .tempfile()?;
    Ok(file.into_temp_path())
}

pub(super) fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32, DynError> {
    let end = offset + 4;
    let slice = bytes
        .get(offset..end)
        .ok_or("u32 read range must be within buffer")?;
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(slice);
    Ok(u32::from_le_bytes(raw))
}

pub(super) fn write_u32_at(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), DynError> {
    let end = offset + 4;
    let target = bytes
        .get_mut(offset..end)
        .ok_or("u32 write range must be within buffer")?;
    target.copy_from_slice(value.to_le_bytes().as_slice());
    Ok(())
}

pub(super) fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, DynError> {
    let end = offset + 8;
    let slice = bytes
        .get(offset..end)
        .ok_or("u64 read range must be within buffer")?;
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(slice);
    Ok(u64::from_le_bytes(raw))
}

pub(super) fn write_u64_at(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), DynError> {
    let end = offset + 8;
    let target = bytes
        .get_mut(offset..end)
        .ok_or("u64 write range must be within buffer")?;
    target.copy_from_slice(value.to_le_bytes().as_slice());
    Ok(())
}

fn descriptor_base(kind: SectionKind) -> usize {
    HEADER_SIZE + kind.index() * DESCRIPTOR_SIZE
}
