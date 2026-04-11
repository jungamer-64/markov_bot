use tokio::fs;

use crate::config::DynError;
use crate::markov::BOS_ID;
use crate::test_support::{ensure, ensure_eq};

use super::super::{SectionKind, load_chain};
use super::helpers::{
    read_u64_at, rewrite_checksum, run_async_test, sample_chain, section_body_offset,
    write_sample_file, write_u32_at, write_u64_at,
};

#[test]
fn rejects_broken_bos_token() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("broken_bos", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let vocab_blob_offset = section_body_offset(&bytes, SectionKind::VocabBlob)?;
        let token = bytes
            .get_mut(vocab_blob_offset)
            .ok_or("vocab blob should contain BOS token bytes")?;
        *token = b'X';
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "broken BOS token should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_non_monotonic_edge_cumulative() -> Result<(), DynError> {
    run_async_test(async {
        let mut chain = crate::markov::MarkovChain::default();
        chain.train_tokens(&["a".to_owned()])?;
        chain.train_tokens(&["b".to_owned()])?;

        let file_path = write_sample_file("edge_non_monotonic", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model3_edge_offset = section_body_offset(&bytes, SectionKind::Model3Edges)?;
        let first_cumulative = read_u64_at(
            &bytes,
            model3_edge_offset
                .checked_add(4)
                .ok_or("model3 edge cumulative offset should fit usize")?,
        )?;
        write_u64_at(
            &mut bytes,
            model3_edge_offset
                .checked_add(16)
                .ok_or("model3 edge write offset should fit usize")?,
            first_cumulative,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "non-monotonic cumulative counts should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_start_prefix_id_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("start_prefix_out_of_bounds", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let start_offset = section_body_offset(&bytes, SectionKind::Starts)?;
        write_u32_at(&mut bytes, start_offset, u32::MAX)?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds start prefix id should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_model2_pair_range_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("model2_pair_range_oob", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model2_pair_offset = section_body_offset(&bytes, SectionKind::Model2Pairs)?;
        write_u32_at(
            &mut bytes,
            model2_pair_offset
                .checked_add(8)
                .ok_or("model2 pair range offset should fit usize")?,
            u32::MAX,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds model2 range should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_model3_edge_range_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("model3_edge_range_oob", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model3_prefix_offset = section_body_offset(&bytes, SectionKind::Model3Prefixes)?;
        write_u32_at(
            &mut bytes,
            model3_prefix_offset
                .checked_add(8)
                .ok_or("model3 prefix range offset should fit usize")?,
            u32::MAX,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds model3 edge range should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_model4_edge_range_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("model4_edge_range_oob", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model4_prefix_offset = section_body_offset(&bytes, SectionKind::Model4Prefixes)?;
        write_u32_at(
            &mut bytes,
            model4_prefix_offset
                .checked_add(8)
                .ok_or("model4 prefix range offset should fit usize")?,
            u32::MAX,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds model4 edge range should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_model5_edge_range_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("model5_edge_range_oob", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model5_prefix_offset = section_body_offset(&bytes, SectionKind::Model5Prefixes)?;
        write_u32_at(
            &mut bytes,
            model5_prefix_offset
                .checked_add(8)
                .ok_or("model5 prefix range offset should fit usize")?,
            u32::MAX,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds model5 edge range should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_model6_edge_range_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("model6_edge_range_oob", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model6_prefix_offset = section_body_offset(&bytes, SectionKind::Model6Prefixes)?;
        write_u32_at(
            &mut bytes,
            model6_prefix_offset
                .checked_add(8)
                .ok_or("model6 prefix range offset should fit usize")?,
            u32::MAX,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds model6 edge range should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_edge_token_id_out_of_bounds() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("edge_token_oob", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let model1_edge_offset = section_body_offset(&bytes, SectionKind::Model1Edges)?;
        write_u32_at(&mut bytes, model1_edge_offset, u32::MAX)?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "out-of-bounds edge token id should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn loads_cumulative_values_beyond_u32_max() -> Result<(), DynError> {
    run_async_test(async {
        let mut chain = crate::markov::MarkovChain::default();
        chain.train_tokens(&["x".to_owned()])?;

        let file_path = write_sample_file("u64_cumulative", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let start_offset = section_body_offset(&bytes, SectionKind::Starts)?;
        let model6_prefix_offset = section_body_offset(&bytes, SectionKind::Model6Prefixes)?;
        let model6_edge_offset = section_body_offset(&bytes, SectionKind::Model6Edges)?;

        let huge = u64::from(u32::MAX) + 10;

        write_u64_at(
            &mut bytes,
            start_offset
                .checked_add(4)
                .ok_or("start cumulative offset should fit usize")?,
            huge,
        )?;
        write_u64_at(
            &mut bytes,
            model6_prefix_offset
                .checked_add(12)
                .ok_or("model6 prefix cumulative offset should fit usize")?,
            huge,
        )?;
        write_u64_at(
            &mut bytes,
            model6_edge_offset
                .checked_add(4)
                .ok_or("model6 edge cumulative offset should fit usize")?,
            huge,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        let loaded = load_chain(&file_path).await?;
        let Some(x_id) = loaded.token_to_id.get("x").copied() else {
            return Err("token id for 'x' should exist".into());
        };
        ensure_eq(
            &loaded
                .starts
                .get(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, x_id]),
            &Some(&huge),
            "start cumulative values should preserve u64 precision",
        )?;

        let Some(edges) = loaded
            .model6
            .get(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID])
        else {
            return Err("model6 prefix should exist".into());
        };
        let total: u64 = edges.values().copied().sum();
        ensure_eq(
            &total,
            &huge,
            "edge cumulative values should preserve u64 precision",
        )?;
        Ok(())
    })
}
