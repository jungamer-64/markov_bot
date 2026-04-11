use rand::{SeedableRng, rngs::StdRng};
use tempfile::tempdir;

use crate::markov::{BOS_ID, MarkovChain};

use super::super::{StorageCompressionMode, load_chain, save_chain};
use super::helpers::{
    TEST_COMPRESSION_MODE, TEST_MIN_EDGE_COUNT, temp_file_path, write_sample_file_with_settings,
};

#[tokio::test]
async fn load_returns_default_for_missing_file() {
    let temp_dir = tempdir().expect("tempdir should be created");
    let file_path = temp_dir.path().join("missing.mkv3");
    let chain = load_chain(file_path.as_path())
        .await
        .expect("load should succeed");

    assert!(chain.starts.is_empty());
}

#[tokio::test]
async fn save_and_load_roundtrip() {
    let file_path = temp_file_path("roundtrip");
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

    save_chain(
        &file_path,
        &chain,
        TEST_MIN_EDGE_COUNT,
        TEST_COMPRESSION_MODE,
    )
    .await
    .expect("save should succeed");
    let loaded = load_chain(&file_path).await.expect("load should succeed");

    assert!(!loaded.starts.is_empty());

    let mut left_rng = StdRng::seed_from_u64(7);
    let mut right_rng = StdRng::seed_from_u64(7);
    let left = chain.generate_sentence(&mut left_rng, 10);
    let right = loaded.generate_sentence(&mut right_rng, 10);

    assert_eq!(left, right);
}

#[tokio::test]
async fn save_and_load_roundtrip_with_mode_auto() {
    let file_path = temp_file_path("roundtrip_mode_auto");
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

    save_chain(
        &file_path,
        &chain,
        TEST_MIN_EDGE_COUNT,
        StorageCompressionMode::Auto,
    )
    .await
    .expect("save should succeed");
    let loaded = load_chain(&file_path).await.expect("load should succeed");

    assert!(!loaded.starts.is_empty());

    let mut left_rng = StdRng::seed_from_u64(7);
    let mut right_rng = StdRng::seed_from_u64(7);
    let left = chain.generate_sentence(&mut left_rng, 10);
    let right = loaded.generate_sentence(&mut right_rng, 10);

    assert_eq!(left, right);
}

#[tokio::test]
async fn pruning_roundtrip_keeps_remaining_start_and_backoff_data() {
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(&["keep".to_owned()])
        .expect("training should succeed");
    chain
        .train_tokens(&["keep".to_owned()])
        .expect("training should succeed");
    chain
        .train_tokens(&["drop".to_owned()])
        .expect("training should succeed");

    let file_path = write_sample_file_with_settings(
        "pruning_roundtrip",
        &chain,
        2,
        StorageCompressionMode::Auto,
    )
    .await;
    let loaded = load_chain(&file_path).await.expect("load should succeed");

    let keep_id = *loaded
        .token_to_id
        .get("keep")
        .expect("keep token should exist");
    let drop_id = *loaded
        .token_to_id
        .get("drop")
        .expect("drop token should exist");

    assert_eq!(loaded.starts.get(&[BOS_ID, BOS_ID, keep_id]), Some(&2));
    assert!(!loaded.starts.contains_key(&[BOS_ID, BOS_ID, drop_id]));

    let start_edges = loaded
        .model3
        .get(&[BOS_ID, BOS_ID, BOS_ID])
        .expect("model3 start prefix should exist");
    assert_eq!(start_edges.get(&keep_id), Some(&2));
    assert!(!start_edges.contains_key(&drop_id));
}
