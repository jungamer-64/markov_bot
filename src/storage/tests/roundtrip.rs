use rand::{SeedableRng, rngs::StdRng};
use tempfile::tempdir;

use crate::config::DynError;
use crate::markov::{BOS_ID, MarkovChain};
use crate::test_support::{ensure, ensure_eq};

use super::super::{StorageCompressionMode, load_chain, save_chain};
use super::helpers::{
    TEST_COMPRESSION_MODE, TEST_MIN_EDGE_COUNT, run_async_test, temp_file_path,
    write_sample_file_with_settings,
};

#[test]
fn load_returns_default_for_missing_file() -> Result<(), DynError> {
    run_async_test(async {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("missing.mkv3");
        let chain = load_chain(file_path.as_path()).await?;

        ensure(
            chain.starts.is_empty(),
            "missing file should load as an empty chain",
        )?;
        Ok(())
    })
}

#[test]
fn save_and_load_roundtrip() -> Result<(), DynError> {
    run_async_test(async {
        let file_path = temp_file_path("roundtrip")?;
        let mut chain = MarkovChain::default();
        chain.train_tokens(&[
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ])?;
        chain.train_tokens(&["a".to_owned()])?;

        save_chain(
            &file_path,
            &chain,
            TEST_MIN_EDGE_COUNT,
            TEST_COMPRESSION_MODE,
        )
        .await?;
        let loaded = load_chain(&file_path).await?;

        ensure(
            !loaded.starts.is_empty(),
            "loaded chain should contain start prefixes",
        )?;
        ensure(
            !loaded.model6.is_empty(),
            "loaded chain should contain model6 data",
        )?;
        ensure(
            !loaded.model5.is_empty(),
            "loaded chain should contain model5 data",
        )?;
        ensure(
            !loaded.model4.is_empty(),
            "loaded chain should contain model4 data",
        )?;

        let mut left_rng = StdRng::seed_from_u64(7);
        let mut right_rng = StdRng::seed_from_u64(7);
        let left = chain.generate_sentence(&mut left_rng, 10);
        let right = loaded.generate_sentence(&mut right_rng, 10);

        ensure_eq(
            &left,
            &right,
            "saved and loaded chains should generate the same sentence",
        )?;
        Ok(())
    })
}

#[test]
fn save_and_load_roundtrip_with_mode_auto() -> Result<(), DynError> {
    run_async_test(async {
        let file_path = temp_file_path("roundtrip_mode_auto")?;
        let mut chain = MarkovChain::default();
        chain.train_tokens(&[
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ])?;
        chain.train_tokens(&["a".to_owned()])?;

        save_chain(
            &file_path,
            &chain,
            TEST_MIN_EDGE_COUNT,
            StorageCompressionMode::Auto,
        )
        .await?;
        let loaded = load_chain(&file_path).await?;

        ensure(
            !loaded.starts.is_empty(),
            "loaded chain should contain start prefixes",
        )?;
        ensure(
            !loaded.model6.is_empty(),
            "loaded chain should contain model6 data",
        )?;
        ensure(
            !loaded.model5.is_empty(),
            "loaded chain should contain model5 data",
        )?;
        ensure(
            !loaded.model4.is_empty(),
            "loaded chain should contain model4 data",
        )?;

        let mut left_rng = StdRng::seed_from_u64(7);
        let mut right_rng = StdRng::seed_from_u64(7);
        let left = chain.generate_sentence(&mut left_rng, 10);
        let right = loaded.generate_sentence(&mut right_rng, 10);

        ensure_eq(
            &left,
            &right,
            "auto-compressed roundtrip should preserve generation",
        )?;
        Ok(())
    })
}

#[test]
fn pruning_roundtrip_keeps_remaining_start_and_backoff_data() -> Result<(), DynError> {
    run_async_test(async {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&["keep".to_owned()])?;
        chain.train_tokens(&["keep".to_owned()])?;
        chain.train_tokens(&["drop".to_owned()])?;

        let file_path = write_sample_file_with_settings(
            "pruning_roundtrip",
            &chain,
            2,
            StorageCompressionMode::Auto,
        )
        .await?;
        let loaded = load_chain(&file_path).await?;

        let Some(keep_id) = loaded.token_to_id.get("keep").copied() else {
            return Err("keep token should exist".into());
        };
        let Some(drop_id) = loaded.token_to_id.get("drop").copied() else {
            return Err("drop token should exist".into());
        };

        ensure_eq(
            &loaded
                .starts
                .get(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, keep_id]),
            &Some(&2),
            "retained start prefix should preserve its cumulative count",
        )?;
        ensure(
            !loaded
                .starts
                .contains_key(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, drop_id]),
            "pruned start prefix should be removed",
        )?;

        let Some(start_edges) = loaded
            .model6
            .get(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID])
        else {
            return Err("model6 start prefix should exist".into());
        };
        ensure_eq(
            &start_edges.get(&keep_id),
            &Some(&2),
            "retained start edge should preserve its cumulative count",
        )?;
        ensure(
            !start_edges.contains_key(&drop_id),
            "pruned start edge should be removed",
        )?;

        let Some(model5_edges) = loaded.model5.get(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID])
        else {
            return Err("model5 start prefix should exist".into());
        };
        ensure_eq(
            &model5_edges.get(&keep_id),
            &Some(&2),
            "retained model5 start edge should preserve its cumulative count",
        )?;
        ensure(
            !model5_edges.contains_key(&drop_id),
            "pruned model5 start edge should be removed",
        )?;

        let Some(model4_edges) = loaded.model4.get(&[BOS_ID, BOS_ID, BOS_ID, BOS_ID]) else {
            return Err("model4 start prefix should exist".into());
        };
        ensure_eq(
            &model4_edges.get(&keep_id),
            &Some(&2),
            "retained model4 start edge should preserve its cumulative count",
        )?;
        ensure(
            !model4_edges.contains_key(&drop_id),
            "pruned model4 start edge should be removed",
        )?;
        Ok(())
    })
}
