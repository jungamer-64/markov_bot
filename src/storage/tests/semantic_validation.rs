use tokio::fs;

use crate::markov::BOS_ID;

use super::super::{SectionKind, load_chain};
use super::helpers::{
    rewrite_checksum, sample_chain, section_body_offset, write_sample_file, write_u32_at,
    write_u64_at,
};

#[tokio::test]
async fn rejects_broken_bos_token() {
    let file_path = write_sample_file("broken_bos", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let vocab_blob_offset = section_body_offset(&bytes, SectionKind::VocabBlob);
    bytes[vocab_blob_offset] = b'X';
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_non_monotonic_edge_cumulative() {
    let mut chain = crate::markov::MarkovChain::default();
    chain
        .train_tokens(&["a".to_owned()])
        .expect("training should succeed");
    chain
        .train_tokens(&["b".to_owned()])
        .expect("training should succeed");

    let file_path = write_sample_file("edge_non_monotonic", &chain).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let model3_edge_offset = section_body_offset(&bytes, SectionKind::Model3Edges);
    let first_cumulative = super::helpers::read_u64_at(&bytes, model3_edge_offset + 4);
    write_u64_at(&mut bytes, model3_edge_offset + 16, first_cumulative);
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_start_prefix_id_out_of_bounds() {
    let file_path = write_sample_file("start_prefix_out_of_bounds", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let start_offset = section_body_offset(&bytes, SectionKind::Starts);
    write_u32_at(&mut bytes, start_offset, u32::MAX);
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_model2_pair_range_out_of_bounds() {
    let file_path = write_sample_file("model2_pair_range_oob", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let model2_pair_offset = section_body_offset(&bytes, SectionKind::Model2Pairs);
    write_u32_at(&mut bytes, model2_pair_offset + 8, u32::MAX);
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_model3_edge_range_out_of_bounds() {
    let file_path = write_sample_file("model3_edge_range_oob", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let model3_prefix_offset = section_body_offset(&bytes, SectionKind::Model3Prefixes);
    write_u32_at(&mut bytes, model3_prefix_offset + 8, u32::MAX);
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn rejects_edge_token_id_out_of_bounds() {
    let file_path = write_sample_file("edge_token_oob", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let model1_edge_offset = section_body_offset(&bytes, SectionKind::Model1Edges);
    write_u32_at(&mut bytes, model1_edge_offset, u32::MAX);
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());
}

#[tokio::test]
async fn loads_cumulative_values_beyond_u32_max() {
    let mut chain = crate::markov::MarkovChain::default();
    chain
        .train_tokens(&["x".to_owned()])
        .expect("training should succeed");

    let file_path = write_sample_file("u64_cumulative", &chain).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let start_offset = section_body_offset(&bytes, SectionKind::Starts);
    let model3_prefix_offset = section_body_offset(&bytes, SectionKind::Model3Prefixes);
    let model3_edge_offset = section_body_offset(&bytes, SectionKind::Model3Edges);

    let huge = u64::from(u32::MAX) + 10;

    write_u64_at(&mut bytes, start_offset + 4, huge);
    write_u64_at(&mut bytes, model3_prefix_offset + 12, huge);
    write_u64_at(&mut bytes, model3_edge_offset + 4, huge);
    rewrite_checksum(&mut bytes);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    let x_id = *loaded
        .token_to_id
        .get("x")
        .expect("token id for 'x' should exist");
    assert_eq!(loaded.starts.get(&[BOS_ID, BOS_ID, x_id]), Some(&huge));

    let edges = loaded
        .model3
        .get(&[BOS_ID, BOS_ID, BOS_ID])
        .expect("model3 prefix should exist");
    let total: u64 = edges.values().copied().sum();
    assert_eq!(total, huge);
}
