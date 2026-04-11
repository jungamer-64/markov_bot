use std::fs;

use crate::{
    config::DynError,
    storage::{FLAG_VOCAB_BLOB_LZ4_FLEX, FLAG_VOCAB_BLOB_RLE, FLAG_VOCAB_BLOB_ZSTD, load_chain},
    test_support::{ensure_eq, ensure_ne},
};

use super::super::StorageCompressionMode;
use super::helpers::{
    FLAGS_OFFSET, run_async_test, sample_chain_with_order, write_sample_file_with_mode,
};

#[test]
fn auto_compresses_repeated_vocab_blob_when_helpful() -> Result<(), DynError> {
    run_async_test(async {
        let mut chain = sample_chain_with_order(7)?;
        for _ in 0..40 {
            chain.train_tokens(&["aaaaaaaaaaaaaaaaaaaaaaaa".to_owned()])?;
        }

        let path =
            write_sample_file_with_mode("compression_auto", &chain, StorageCompressionMode::Auto)
                .await?;
        let bytes = fs::read(&path)?;
        let flags = super::helpers::read_u32_at(bytes.as_slice(), FLAGS_OFFSET)?;

        ensure_ne(
            &flags,
            &0,
            "auto mode should choose a compressed representation",
        )?;
        let loaded = load_chain(&path, 7).await?;
        ensure_eq(
            &loaded.id_to_token,
            &chain.id_to_token,
            "auto-compressed file should round-trip vocabulary",
        )?;
        Ok(())
    })
}

#[test]
fn explicit_compression_modes_round_trip() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(6)?;

        for (mode, expected_flag) in [
            (StorageCompressionMode::Uncompressed, 0),
            (StorageCompressionMode::Rle, FLAG_VOCAB_BLOB_RLE),
            (StorageCompressionMode::Zstd, FLAG_VOCAB_BLOB_ZSTD),
            (StorageCompressionMode::Lz4Flex, FLAG_VOCAB_BLOB_LZ4_FLEX),
        ] {
            let path = write_sample_file_with_mode("compression_modes", &chain, mode).await?;
            let bytes = fs::read(&path)?;
            let flags = super::helpers::read_u32_at(bytes.as_slice(), FLAGS_OFFSET)?;

            ensure_eq(
                &flags,
                &expected_flag,
                "compression flag should match requested mode",
            )?;

            let loaded = load_chain(&path, 6).await?;
            ensure_eq(
                &loaded.id_to_token,
                &chain.id_to_token,
                "compressed file should preserve vocabulary",
            )?;
        }

        Ok(())
    })
}
