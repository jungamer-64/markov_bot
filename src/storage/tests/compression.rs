use tokio::fs;

use crate::config::DynError;
use crate::markov::MarkovChain;
use crate::test_support::{ensure, ensure_eq, ensure_ne};

use super::super::{
    FLAG_VOCAB_BLOB_LZ4_FLEX, FLAG_VOCAB_BLOB_RLE, FLAG_VOCAB_BLOB_ZSTD, SectionKind,
    StorageCompressionMode, load_chain,
};
use super::helpers::{
    FLAGS_OFFSET, read_u32_at, rewrite_checksum, run_async_test, sample_chain, section_body_offset,
    section_body_size, write_sample_file, write_sample_file_with_mode, write_u64_at,
};

#[test]
fn writes_and_loads_compressed_vocab_blob() -> Result<(), DynError> {
    run_async_test(async {
        let repeated = "a".repeat(1024);
        let mut chain = MarkovChain::default();
        chain.train_tokens(std::slice::from_ref(&repeated))?;

        let file_path = write_sample_file("compressed_vocab_blob", &chain).await?;
        let bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_ne(
            &flags,
            &0,
            "compressed sample should set a compression flag",
        )?;

        let loaded = load_chain(&file_path).await?;
        ensure(
            loaded.token_to_id.contains_key(&repeated),
            "loaded chain should retain the repeated token",
        )?;
        Ok(())
    })
}

#[test]
fn writes_uncompressed_vocab_blob_when_mode_none() -> Result<(), DynError> {
    run_async_test(async {
        let repeated = "a".repeat(1024);
        let mut chain = MarkovChain::default();
        chain.train_tokens(std::slice::from_ref(&repeated))?;

        let file_path = write_sample_file_with_mode(
            "vocab_blob_mode_none",
            &chain,
            StorageCompressionMode::Uncompressed,
        )
        .await?;
        let bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_eq(
            &(flags & FLAG_VOCAB_BLOB_RLE),
            &0,
            "uncompressed mode should not set the RLE flag",
        )?;
        ensure_eq(
            &(flags & FLAG_VOCAB_BLOB_ZSTD),
            &0,
            "uncompressed mode should not set the Zstd flag",
        )?;
        ensure_eq(
            &(flags & FLAG_VOCAB_BLOB_LZ4_FLEX),
            &0,
            "uncompressed mode should not set the LZ4 flag",
        )?;

        let loaded = load_chain(&file_path).await?;
        ensure(
            loaded.token_to_id.contains_key(&repeated),
            "loaded chain should retain the repeated token",
        )?;
        Ok(())
    })
}

#[test]
fn writes_rle_vocab_blob_when_mode_rle() -> Result<(), DynError> {
    run_async_test(async {
        let token = "abcdefg".to_owned();
        let mut chain = MarkovChain::default();
        chain.train_tokens(std::slice::from_ref(&token))?;

        let file_path =
            write_sample_file_with_mode("vocab_blob_mode_rle", &chain, StorageCompressionMode::Rle)
                .await?;
        let bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_ne(
            &(flags & FLAG_VOCAB_BLOB_RLE),
            &0,
            "rle mode should set the RLE flag",
        )?;

        let loaded = load_chain(&file_path).await?;
        ensure(
            loaded.token_to_id.contains_key(&token),
            "loaded chain should retain the token",
        )?;
        Ok(())
    })
}

#[test]
fn writes_zstd_vocab_blob_when_mode_zstd() -> Result<(), DynError> {
    run_async_test(async {
        let token = "abcdefg".repeat(128);
        let mut chain = MarkovChain::default();
        chain.train_tokens(std::slice::from_ref(&token))?;

        let file_path = write_sample_file_with_mode(
            "vocab_blob_mode_zstd",
            &chain,
            StorageCompressionMode::Zstd,
        )
        .await?;
        let bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_ne(
            &(flags & FLAG_VOCAB_BLOB_ZSTD),
            &0,
            "zstd mode should set the Zstd flag",
        )?;

        let loaded = load_chain(&file_path).await?;
        ensure(
            loaded.token_to_id.contains_key(&token),
            "loaded chain should retain the token",
        )?;
        Ok(())
    })
}

#[test]
fn writes_lz4_flex_vocab_blob_when_mode_lz4_flex() -> Result<(), DynError> {
    run_async_test(async {
        let token = "abcdefg".repeat(128);
        let mut chain = MarkovChain::default();
        chain.train_tokens(std::slice::from_ref(&token))?;

        let file_path = write_sample_file_with_mode(
            "vocab_blob_mode_lz4_flex",
            &chain,
            StorageCompressionMode::Lz4Flex,
        )
        .await?;
        let bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_ne(
            &(flags & FLAG_VOCAB_BLOB_LZ4_FLEX),
            &0,
            "lz4 mode should set the LZ4 flag",
        )?;

        let loaded = load_chain(&file_path).await?;
        ensure(
            loaded.token_to_id.contains_key(&token),
            "loaded chain should retain the token",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_corrupted_compressed_vocab_blob() -> Result<(), DynError> {
    run_async_test(async {
        let repeated = "a".repeat(1024);
        let mut chain = MarkovChain::default();
        chain.train_tokens(&[repeated])?;

        let file_path = write_sample_file_with_mode(
            "compressed_vocab_blob_corrupt",
            &chain,
            StorageCompressionMode::Rle,
        )
        .await?;
        let mut bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_ne(
            &(flags & FLAG_VOCAB_BLOB_RLE),
            &0,
            "corrupted RLE sample should still advertise RLE compression",
        )?;

        let vocab_blob_offset = section_body_offset(&bytes, SectionKind::VocabBlob)?;
        let byte = bytes
            .get_mut(vocab_blob_offset)
            .ok_or("vocab blob offset must be within the sample file")?;
        *byte = 127;
        rewrite_checksum(&mut bytes)?;

        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "corrupted compressed vocab blob should fail to load",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_multiple_compression_flags() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("multiple_compression_flags", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        super::helpers::write_u32_at(
            &mut bytes,
            FLAGS_OFFSET,
            FLAG_VOCAB_BLOB_RLE | FLAG_VOCAB_BLOB_ZSTD,
        )?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "files with multiple compression flags should fail to load",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_compressed_vocab_blob_with_impossible_decoded_size() -> Result<(), DynError> {
    run_async_test(async {
        let repeated = "a".repeat(1024);
        let mut chain = MarkovChain::default();
        chain.train_tokens(std::slice::from_ref(&repeated))?;

        let file_path = write_sample_file_with_mode(
            "compressed_vocab_blob_impossible_size",
            &chain,
            StorageCompressionMode::Rle,
        )
        .await?;
        let mut bytes = fs::read(&file_path).await?;

        let flags = read_u32_at(&bytes, FLAGS_OFFSET)?;
        ensure_ne(
            &(flags & FLAG_VOCAB_BLOB_RLE),
            &0,
            "RLE sample should set the RLE flag",
        )?;

        let vocab_offsets_offset = section_body_offset(&bytes, SectionKind::VocabOffsets)?;
        let offset_count = section_body_size(&bytes, SectionKind::VocabOffsets)? / 8;
        write_u64_at(
            &mut bytes,
            vocab_offsets_offset + (offset_count - 1) * 8,
            10_000,
        )?;
        rewrite_checksum(&mut bytes)?;

        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "impossible decoded size should fail to load",
        )?;
        Ok(())
    })
}
