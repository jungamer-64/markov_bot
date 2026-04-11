use std::fs;

use crate::{config::DynError, storage::load_chain, test_support::ensure};

use super::helpers::{
    descriptor, descriptor_size_offset, model_descriptor_index, rewrite_checksum, run_async_test,
    sample_chain_with_order, section_body_offset, write_sample_file, write_u32_at, write_u64_at,
};

#[test]
fn rejects_saved_ngram_order_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("saved_order_mismatch", &chain).await?;

        let result = load_chain(&path, 6).await;
        ensure(
            result.is_err(),
            "saved ngram order mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_model_section_size_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("model_size_mismatch", &chain).await?;
        let mut bytes = fs::read(&path)?;

        let descriptor_index = model_descriptor_index(bytes.as_slice(), 7)?;
        let descriptor_size_offset = descriptor_size_offset(descriptor_index)?;
        let size = descriptor(bytes.as_slice(), descriptor_index)?.size;
        write_u64_at(bytes.as_mut_slice(), descriptor_size_offset, size - 1)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(
            result.is_err(),
            "model section size mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_invalid_model_edge_range() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain_with_order(7)?;
        let path = write_sample_file("invalid_edge_range", &chain).await?;
        let mut bytes = fs::read(&path)?;

        let descriptor_index = model_descriptor_index(bytes.as_slice(), 7)?;
        let section_offset = section_body_offset(bytes.as_slice(), descriptor_index)?;
        let edge_len_offset = section_offset
            .checked_add(4 + 4 + 7 * 4 + 4)
            .ok_or("edge length offset overflow")?;
        write_u32_at(bytes.as_mut_slice(), edge_len_offset, u32::MAX)?;
        rewrite_checksum(bytes.as_mut_slice())?;
        fs::write(&path, bytes)?;

        let result = load_chain(&path, 7).await;
        ensure(
            result.is_err(),
            "invalid model edge range should be rejected",
        )?;
        Ok(())
    })
}
