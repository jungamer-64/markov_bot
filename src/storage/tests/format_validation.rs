use std::fs;

use crate::{config::DynError, storage::load_chain, test_support::ensure};

use super::helpers::{
    FLAGS_OFFSET, NGRAM_ORDER_OFFSET, SECTION_COUNT_OFFSET, UNSUPPORTED_FLAG, VERSION_OFFSET,
    descriptor, descriptor_flags_offset, model_descriptor_index, read_u64_at, rewrite_checksum,
    run_async_test, sample_chain_with_order, write_sample_file, write_u32_at, write_u64_at,
};

#[test]
fn rejects_version_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("invalid_version", &chain).await?;
        let mut bytes = fs::read(&path)?;
        write_u32_at(bytes.as_mut_slice(), VERSION_OFFSET, 6)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(result.is_err(), "version mismatch should be rejected")?;
        Ok(())
    })
}

#[test]
fn rejects_header_ngram_order_zero() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("invalid_ngram_order", &chain).await?;
        let mut bytes = fs::read(&path)?;
        write_u32_at(bytes.as_mut_slice(), NGRAM_ORDER_OFFSET, 0)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(result.is_err(), "ngram order 0 should be rejected")?;
        Ok(())
    })
}

#[test]
fn rejects_section_count_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("invalid_section_count", &chain).await?;
        let mut bytes = fs::read(&path)?;
        let current = read_u64_at(bytes.as_slice(), SECTION_COUNT_OFFSET)?;
        write_u64_at(bytes.as_mut_slice(), SECTION_COUNT_OFFSET, current + 1)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(result.is_err(), "section count mismatch should be rejected")?;
        Ok(())
    })
}

#[test]
fn rejects_unsupported_header_flags() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("invalid_flags", &chain).await?;
        let mut bytes = fs::read(&path)?;
        write_u32_at(bytes.as_mut_slice(), FLAGS_OFFSET, UNSUPPORTED_FLAG)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(
            result.is_err(),
            "unsupported header flags should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_duplicate_model_descriptor_orders() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("duplicate_model_order", &chain).await?;
        let mut bytes = fs::read(&path)?;

        let order7_index = model_descriptor_index(bytes.as_slice(), 7)?;
        let order6_index = model_descriptor_index(bytes.as_slice(), 6)?;
        let duplicate_flags_offset = descriptor_flags_offset(order6_index)?;
        let source_flags = descriptor(bytes.as_slice(), order7_index)?.flags;
        write_u32_at(bytes.as_mut_slice(), duplicate_flags_offset, source_flags)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(
            result.is_err(),
            "duplicate model descriptor orders should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_ascending_model_descriptor_orders() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("ascending_model_order", &chain).await?;
        let mut bytes = fs::read(&path)?;

        let order7_index = model_descriptor_index(bytes.as_slice(), 7)?;
        let order6_index = model_descriptor_index(bytes.as_slice(), 6)?;
        let order7_flags_offset = descriptor_flags_offset(order7_index)?;
        let order6_flags = descriptor(bytes.as_slice(), order6_index)?.flags;
        write_u32_at(bytes.as_mut_slice(), order7_flags_offset, order6_flags)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(
            result.is_err(),
            "ascending model descriptor order should be rejected",
        )?;
        Ok(())
    })
}
