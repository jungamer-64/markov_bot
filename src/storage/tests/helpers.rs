use tempfile::{Builder, TempPath};

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
        offset: read_u64_at(bytes, base + 8),
        size: read_u64_at(bytes, base + 16),
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

pub(super) fn section_body_offset(bytes: &[u8], kind: SectionKind) -> usize {
    usize::try_from(descriptor(bytes, kind).offset).expect("section offset should fit usize")
}

pub(super) fn section_body_size(bytes: &[u8], kind: SectionKind) -> usize {
    usize::try_from(descriptor(bytes, kind).size).expect("section size should fit usize")
}

pub(super) fn first_padding_offset(bytes: &[u8]) -> Option<usize> {
    let mut cursor =
        usize::try_from(aligned_metadata_end(SECTION_COUNT).expect("metadata size should fit"))
            .expect("metadata size should fit usize");

    for kind in SectionKind::ALL {
        let descriptor = descriptor(bytes, kind);
        let offset = usize::try_from(descriptor.offset).expect("offset should fit usize");
        if cursor < offset {
            return Some(cursor);
        }

        cursor = offset
            .checked_add(usize::try_from(descriptor.size).expect("size should fit usize"))
            .expect("section end should fit usize");
    }

    None
}

pub(super) fn first_fixed_section_with_gap(bytes: &[u8]) -> Option<SectionKind> {
    for pair in SectionKind::ALL.windows(2) {
        let [left, right] = [pair[0], pair[1]];
        if matches!(left, SectionKind::VocabOffsets | SectionKind::VocabBlob) {
            continue;
        }

        let left_descriptor = descriptor(bytes, left);
        let left_end = left_descriptor
            .offset
            .checked_add(left_descriptor.size)
            .expect("left section end should fit u64");
        let right_start = descriptor(bytes, right).offset;
        if left_end < right_start {
            return Some(left);
        }
    }

    None
}

pub(super) fn rewrite_checksum(bytes: &mut [u8]) {
    let checksum = compute_checksum(bytes).expect("checksum should be computed");
    write_u64_at(bytes, CHECKSUM_OFFSET, checksum);
}

pub(super) async fn write_sample_file(prefix: &str, chain: &MarkovChain) -> TempPath {
    write_sample_file_with_settings(prefix, chain, TEST_MIN_EDGE_COUNT, TEST_COMPRESSION_MODE).await
}

pub(super) async fn write_sample_file_with_mode(
    prefix: &str,
    chain: &MarkovChain,
    compression_mode: StorageCompressionMode,
) -> TempPath {
    write_sample_file_with_settings(prefix, chain, TEST_MIN_EDGE_COUNT, compression_mode).await
}

pub(super) async fn write_sample_file_with_settings(
    prefix: &str,
    chain: &MarkovChain,
    min_edge_count: u64,
    compression_mode: StorageCompressionMode,
) -> TempPath {
    let file_path = temp_file_path(prefix);
    save_chain(&file_path, chain, min_edge_count, compression_mode)
        .await
        .expect("save should succeed");
    file_path
}

pub(super) fn sample_chain() -> MarkovChain {
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(&[
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ])
        .expect("training should succeed");
    chain
        .train_tokens(&["a".to_owned()])
        .expect("training should succeed");
    chain
}

pub(super) fn temp_file_path(prefix: &str) -> TempPath {
    Builder::new()
        .prefix(&format!("markov_bot_{prefix}_"))
        .suffix(".mkv3")
        .tempfile()
        .expect("tempfile should be created")
        .into_temp_path()
}

pub(super) fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    let end = offset + 4;
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(&bytes[offset..end]);
    u32::from_le_bytes(raw)
}

pub(super) fn write_u32_at(bytes: &mut [u8], offset: usize, value: u32) {
    let end = offset + 4;
    bytes[offset..end].copy_from_slice(value.to_le_bytes().as_slice());
}

pub(super) fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    let end = offset + 8;
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(&bytes[offset..end]);
    u64::from_le_bytes(raw)
}

pub(super) fn write_u64_at(bytes: &mut [u8], offset: usize, value: u64) {
    let end = offset + 8;
    bytes[offset..end].copy_from_slice(value.to_le_bytes().as_slice());
}

fn descriptor_base(kind: SectionKind) -> usize {
    HEADER_SIZE + kind.index() * DESCRIPTOR_SIZE
}
