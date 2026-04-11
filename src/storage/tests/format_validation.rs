use tokio::fs;

use super::super::{
    CHECKSUM_OFFSET, NORMALIZATION_FLAGS, SECTION_COUNT_U32, SectionKind, TOKENIZER_VERSION,
    VERSION, load_chain,
};
use super::helpers::{
    FILE_SIZE_OFFSET, FLAGS_OFFSET, NORMALIZATION_FLAGS_OFFSET, SECTION_COUNT_OFFSET,
    TOKENIZER_VERSION_OFFSET, UNSUPPORTED_FLAG, VERSION_OFFSET, descriptor_kind_offset,
    descriptor_offset_offset, descriptor_size_offset, first_fixed_section_with_gap,
    first_padding_offset, read_u32_at, read_u64_at, rewrite_checksum, sample_chain,
    write_sample_file, write_u32_at, write_u64_at,
};

#[tokio::test]
async fn rejects_invalid_magic() {
    let file_path = write_sample_file("invalid_magic", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    bytes[0] = b'X';
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_version_mismatch() {
    let file_path = write_sample_file("version_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(&mut bytes, VERSION_OFFSET, VERSION.saturating_add(1));
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_flags_mismatch() {
    let file_path = write_sample_file("flags_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(&mut bytes, FLAGS_OFFSET, UNSUPPORTED_FLAG);
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_tokenizer_or_normalization_mismatch() {
    let file_path = write_sample_file("preprocess_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(
        &mut bytes,
        TOKENIZER_VERSION_OFFSET,
        TOKENIZER_VERSION.saturating_add(1),
    );
    fs::write(&file_path, &bytes)
        .await
        .expect("write should succeed");
    assert!(load_chain(&file_path).await.is_err());

    write_u32_at(&mut bytes, TOKENIZER_VERSION_OFFSET, TOKENIZER_VERSION);
    write_u32_at(
        &mut bytes,
        NORMALIZATION_FLAGS_OFFSET,
        NORMALIZATION_FLAGS.saturating_add(1),
    );
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");
    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_section_count_mismatch() {
    let file_path = write_sample_file("section_count_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(
        &mut bytes,
        SECTION_COUNT_OFFSET,
        SECTION_COUNT_U32.saturating_sub(1),
    );
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_file_size_mismatch() {
    let file_path = write_sample_file("filesize_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    let file_size = read_u64_at(&bytes, FILE_SIZE_OFFSET);
    write_u64_at(&mut bytes, FILE_SIZE_OFFSET, file_size.saturating_add(1));
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_checksum_mismatch() {
    let file_path = write_sample_file("checksum_non_zero", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u64_at(&mut bytes, CHECKSUM_OFFSET, 1);
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn writes_non_zero_checksum() {
    let file_path = write_sample_file("checksum_written", &sample_chain()).await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let checksum = read_u64_at(&bytes, CHECKSUM_OFFSET);
    assert_ne!(checksum, 0);
}

#[tokio::test]
async fn rejects_unknown_section_kind() {
    let file_path = write_sample_file("unknown_section_kind", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(
        &mut bytes,
        descriptor_kind_offset(SectionKind::Model2Pairs),
        99,
    );
    rewrite_checksum(&mut bytes);
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_duplicate_section_kind() {
    let file_path = write_sample_file("duplicate_section_kind", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(
        &mut bytes,
        descriptor_kind_offset(SectionKind::Model2Pairs),
        SectionKind::Model3Edges.as_u32(),
    );
    rewrite_checksum(&mut bytes);
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_descriptor_order_violation() {
    let file_path = write_sample_file("descriptor_order", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let left_offset = descriptor_kind_offset(SectionKind::Model2Pairs);
    let right_offset = descriptor_kind_offset(SectionKind::Model2Prefixes);
    let left_kind = read_u32_at(&bytes, left_offset);
    let right_kind = read_u32_at(&bytes, right_offset);
    write_u32_at(&mut bytes, left_offset, right_kind);
    write_u32_at(&mut bytes, right_offset, left_kind);

    rewrite_checksum(&mut bytes);
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_non_8_byte_aligned_section_offset() {
    let file_path = write_sample_file("unaligned_offset", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let offset_location = descriptor_offset_offset(SectionKind::Model2Pairs);
    let offset = read_u64_at(&bytes, offset_location);
    write_u64_at(&mut bytes, offset_location, offset.saturating_add(1));
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_overlapping_sections() {
    let file_path = write_sample_file("section_overlap", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let overlap_target = read_u64_at(&bytes, descriptor_offset_offset(SectionKind::Model3Edges));
    write_u64_at(
        &mut bytes,
        descriptor_offset_offset(SectionKind::Model2Pairs),
        overlap_target,
    );
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_non_zero_padding() {
    let file_path = write_sample_file("padding_corrupt", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let padding_offset = first_padding_offset(&bytes).expect("sample file should contain padding");
    bytes[padding_offset] = 1;
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_fixed_size_section_size_misalignment() {
    let file_path = write_sample_file("section_size_misalignment", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let kind = first_fixed_section_with_gap(&bytes)
        .expect("sample file should contain a gapped fixed section");
    let size_offset = descriptor_size_offset(kind);
    let size = read_u64_at(&bytes, size_offset);
    write_u64_at(&mut bytes, size_offset, size.saturating_add(1));
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}
