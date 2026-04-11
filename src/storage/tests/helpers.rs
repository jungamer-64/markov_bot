use std::future::Future;

use tempfile::{Builder, TempPath};

use crate::{config::DynError, markov::MarkovChain, test_support::ensure};

use super::super::types::SectionKind;
use super::super::{
    CHECKSUM_OFFSET, DESCRIPTOR_SIZE, HEADER_SIZE, SECTION_COUNT_BASE, StorageCompressionMode,
    compute_checksum, descriptor_count_for_ngram_order, save_chain,
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

pub(super) fn sample_chain_with_order(ngram_order: usize) -> Result<MarkovChain, DynError> {
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
        chain.models.len() == ngram_order,
        "model count should match ngram order",
    )?;
    Ok(chain)
}

pub(super) fn temp_file_path(prefix: &str) -> Result<TempPath, DynError> {
    let file = Builder::new()
        .prefix(&format!("markov_bot_{prefix}_"))
        .suffix(".mkv3")
        .tempfile()?;
    Ok(file.into_temp_path())
}

pub(super) fn descriptor(bytes: &[u8], index: usize) -> Result<DescriptorView, DynError> {
    let base = descriptor_base(index)?;
    Ok(DescriptorView {
        kind: read_u32_at(bytes, base)?,
        flags: read_u32_at(bytes, base + 4)?,
        offset: read_u64_at(bytes, base + 8)?,
        size: read_u64_at(bytes, base + 16)?,
    })
}

pub(super) fn descriptor_count(bytes: &[u8]) -> Result<usize, DynError> {
    let section_count = read_u64_at(bytes, SECTION_COUNT_OFFSET)?;
    usize::try_from(section_count).map_err(|_error| "section count should fit usize".into())
}

pub(super) fn find_descriptor_index(
    bytes: &[u8],
    kind: SectionKind,
    flags: u32,
) -> Result<usize, DynError> {
    let descriptor_count = descriptor_count(bytes)?;
    for index in 0..descriptor_count {
        let view = descriptor(bytes, index)?;
        if view.kind == kind.as_u32() && view.flags == flags {
            return Ok(index);
        }
    }

    Err(format!("descriptor not found: {} flags={flags}", kind.label()).into())
}

pub(super) fn model_descriptor_index(bytes: &[u8], order: usize) -> Result<usize, DynError> {
    let flags = u32::try_from(order).map_err(|_error| "order exceeds u32 range")?;
    find_descriptor_index(bytes, SectionKind::Model, flags)
}

pub(super) fn section_body_offset(bytes: &[u8], index: usize) -> Result<usize, DynError> {
    usize::try_from(descriptor(bytes, index)?.offset)
        .map_err(|_error| "section offset should fit usize".into())
}

pub(super) fn descriptor_flags_offset(index: usize) -> Result<usize, DynError> {
    let base = descriptor_base(index)?;
    base.checked_add(4)
        .ok_or_else(|| "descriptor flags offset overflow".into())
}

pub(super) fn descriptor_size_offset(index: usize) -> Result<usize, DynError> {
    let base = descriptor_base(index)?;
    base.checked_add(16)
        .ok_or_else(|| "descriptor size offset overflow".into())
}

pub(super) fn rewrite_checksum(bytes: &mut [u8]) -> Result<(), DynError> {
    let checksum = compute_checksum(bytes)?;
    write_u64_at(bytes, CHECKSUM_OFFSET, checksum)
}

pub(super) fn expected_section_count(order: usize) -> Result<u64, DynError> {
    descriptor_count_for_ngram_order(order)
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

fn descriptor_base(index: usize) -> Result<usize, DynError> {
    let descriptor_count = HEADER_SIZE
        .checked_add(
            index
                .checked_mul(DESCRIPTOR_SIZE)
                .ok_or("descriptor base overflow")?,
        )
        .ok_or("descriptor base overflow")?;
    Ok(descriptor_count)
}

#[test]
fn section_count_formula_stays_dynamic() -> Result<(), DynError> {
    ensure(
        expected_section_count(7)? == SECTION_COUNT_BASE + 7,
        "section count should be 3 + ngram_order",
    )?;
    Ok(())
}
