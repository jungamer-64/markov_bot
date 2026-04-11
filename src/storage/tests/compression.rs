use tokio::fs;

use crate::markov::MarkovChain;

use super::super::{
    FLAG_VOCAB_BLOB_LZ4_FLEX, FLAG_VOCAB_BLOB_RLE, FLAG_VOCAB_BLOB_ZSTD, SectionKind,
    StorageCompressionMode, load_chain,
};
use super::helpers::{
    FLAGS_OFFSET, read_u32_at, rewrite_checksum, sample_chain, section_body_offset,
    section_body_size, write_sample_file, write_sample_file_with_mode, write_u64_at,
};

#[tokio::test]
async fn writes_and_loads_compressed_vocab_blob() {
    let repeated = "a".repeat(1024);
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(std::slice::from_ref(&repeated))
        .expect("training should succeed");

    let file_path = write_sample_file("compressed_vocab_blob", &chain).await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_ne!(flags, 0);

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    assert!(loaded.token_to_id.contains_key(&repeated));
}

#[tokio::test]
async fn writes_uncompressed_vocab_blob_when_mode_none() {
    let repeated = "a".repeat(1024);
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(std::slice::from_ref(&repeated))
        .expect("training should succeed");

    let file_path = write_sample_file_with_mode(
        "vocab_blob_mode_none",
        &chain,
        StorageCompressionMode::Uncompressed,
    )
    .await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_eq!(flags & FLAG_VOCAB_BLOB_RLE, 0);
    assert_eq!(flags & FLAG_VOCAB_BLOB_ZSTD, 0);
    assert_eq!(flags & FLAG_VOCAB_BLOB_LZ4_FLEX, 0);

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    assert!(loaded.token_to_id.contains_key(&repeated));
}

#[tokio::test]
async fn writes_rle_vocab_blob_when_mode_rle() {
    let token = "abcdefg".to_owned();
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(std::slice::from_ref(&token))
        .expect("training should succeed");

    let file_path =
        write_sample_file_with_mode("vocab_blob_mode_rle", &chain, StorageCompressionMode::Rle)
            .await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_ne!(flags & FLAG_VOCAB_BLOB_RLE, 0);

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    assert!(loaded.token_to_id.contains_key(&token));
}

#[tokio::test]
async fn writes_zstd_vocab_blob_when_mode_zstd() {
    let token = "abcdefg".repeat(128);
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(std::slice::from_ref(&token))
        .expect("training should succeed");

    let file_path =
        write_sample_file_with_mode("vocab_blob_mode_zstd", &chain, StorageCompressionMode::Zstd)
            .await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_ne!(flags & FLAG_VOCAB_BLOB_ZSTD, 0);

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    assert!(loaded.token_to_id.contains_key(&token));
}

#[tokio::test]
async fn writes_lz4_flex_vocab_blob_when_mode_lz4_flex() {
    let token = "abcdefg".repeat(128);
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(std::slice::from_ref(&token))
        .expect("training should succeed");

    let file_path = write_sample_file_with_mode(
        "vocab_blob_mode_lz4_flex",
        &chain,
        StorageCompressionMode::Lz4Flex,
    )
    .await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_ne!(flags & FLAG_VOCAB_BLOB_LZ4_FLEX, 0);

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    assert!(loaded.token_to_id.contains_key(&token));
}

#[tokio::test]
async fn rejects_corrupted_compressed_vocab_blob() {
    let repeated = "a".repeat(1024);
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(&[repeated])
        .expect("training should succeed");

    let file_path = write_sample_file_with_mode(
        "compressed_vocab_blob_corrupt",
        &chain,
        StorageCompressionMode::Rle,
    )
    .await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_ne!(flags & FLAG_VOCAB_BLOB_RLE, 0);

    let vocab_blob_offset = section_body_offset(&bytes, SectionKind::VocabBlob);
    bytes[vocab_blob_offset] = 127;
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_multiple_compression_flags() {
    let file_path = write_sample_file("multiple_compression_flags", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    super::helpers::write_u32_at(
        &mut bytes,
        FLAGS_OFFSET,
        FLAG_VOCAB_BLOB_RLE | FLAG_VOCAB_BLOB_ZSTD,
    );
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_compressed_vocab_blob_with_impossible_decoded_size() {
    let repeated = "a".repeat(1024);
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(std::slice::from_ref(&repeated))
        .expect("training should succeed");

    let file_path = write_sample_file_with_mode(
        "compressed_vocab_blob_impossible_size",
        &chain,
        StorageCompressionMode::Rle,
    )
    .await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let flags = read_u32_at(&bytes, FLAGS_OFFSET);
    assert_ne!(flags & FLAG_VOCAB_BLOB_RLE, 0);

    let vocab_offsets_offset = section_body_offset(&bytes, SectionKind::VocabOffsets);
    let offset_count = section_body_size(&bytes, SectionKind::VocabOffsets) / 8;
    write_u64_at(
        &mut bytes,
        vocab_offsets_offset + (offset_count - 1) * 8,
        10_000,
    );
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}
