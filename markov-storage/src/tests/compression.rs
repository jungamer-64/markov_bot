use std::fs;

use super::super::StorageCompressionMode;
use super::helpers::{
    FLAGS_OFFSET, ensure_eq, ensure_ne, load_sample_file, sample_chain_with_order,
    write_sample_file_with_mode,
};
use crate::{FLAG_VOCAB_BLOB_LZ4_FLEX, FLAG_VOCAB_BLOB_RLE, FLAG_VOCAB_BLOB_ZSTD};

#[test]
fn auto_compresses_repeated_vocab_blob_when_helpful() -> Result<(), crate::StorageError> {
    let mut chain = sample_chain_with_order(7)?;
    for i in 0..40 {
        chain.train_tokens(&[format!("token_{:064}", i)])?;
    }

    let path =
        write_sample_file_with_mode("compression_auto", &chain, StorageCompressionMode::Auto)?;
    let bytes = fs::read(&path)?;
    let flags = super::helpers::read_u32_at(bytes.as_slice(), FLAGS_OFFSET)?;

    ensure_ne(
        &flags,
        &0,
        "auto mode should choose a compressed representation",
    )?;
    let loaded = load_sample_file(&path, 7)?;
    ensure_eq(
        loaded.id_to_token(),
        chain.id_to_token(),
        "auto-compressed file should round-trip vocabulary",
    )?;
    Ok(())
}

#[test]
fn explicit_compression_modes_round_trip() -> Result<(), crate::StorageError> {
    let chain = sample_chain_with_order(6)?;

    for (mode, expected_flag) in [
        (StorageCompressionMode::Uncompressed, 0),
        (StorageCompressionMode::Rle, FLAG_VOCAB_BLOB_RLE),
        (StorageCompressionMode::Zstd, FLAG_VOCAB_BLOB_ZSTD),
        (StorageCompressionMode::Lz4Flex, FLAG_VOCAB_BLOB_LZ4_FLEX),
    ] {
        let path = write_sample_file_with_mode("compression_modes", &chain, mode)?;
        let bytes = fs::read(&path)?;
        let flags = super::helpers::read_u32_at(bytes.as_slice(), FLAGS_OFFSET)?;

        ensure_eq(
            &flags,
            &expected_flag,
            "compression flag should match requested mode",
        )?;

        let loaded = load_sample_file(&path, 6)?;
        ensure_eq(
            loaded.id_to_token(),
            chain.id_to_token(),
            "compressed file should preserve vocabulary",
        )?;
    }

    Ok(())
}
