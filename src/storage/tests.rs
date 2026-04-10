use rand::{SeedableRng, rngs::StdRng};
use tempfile::{Builder, TempPath, tempdir};
use tokio::fs;

use crate::markov::{BOS_ID, MarkovChain};

use super::{
    CHECKSUM_OFFSET, FLAGS, HEADER_SIZE, NORMALIZATION_FLAGS, TOKENIZER_VERSION, VERSION,
    compute_checksum, load_chain, save_chain,
};

const VERSION_OFFSET: usize = 8;
const FLAGS_OFFSET: usize = 12;
const TOKENIZER_VERSION_OFFSET: usize = 16;
const NORMALIZATION_FLAGS_OFFSET: usize = 20;
const TOKEN_COUNT_OFFSET: usize = 24;
const VOCAB_OFFSETS_OFFSET_OFFSET: usize = 64;
const START_OFFSET_OFFSET: usize = 80;
const MODEL3_PREFIX_OFFSET_OFFSET: usize = 96;
const MODEL3_EDGE_OFFSET_OFFSET: usize = 104;
const MODEL2_PAIR_OFFSET_OFFSET: usize = 112;
const VOCAB_BLOB_OFFSET_OFFSET: usize = 72;
const FILE_SIZE_OFFSET: usize = 152;

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

    save_chain(&file_path, &chain)
        .await
        .expect("save should succeed");
    let loaded = load_chain(&file_path).await.expect("load should succeed");

    assert!(!loaded.starts.is_empty());

    let mut left_rng = StdRng::seed_from_u64(7);
    let mut right_rng = StdRng::seed_from_u64(7);
    let left = chain.generate_sentence(&mut left_rng, 10);
    let right = loaded.generate_sentence(&mut right_rng, 10);

    assert_eq!(left, right);

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_invalid_magic() {
    let file_path = write_sample_file("invalid_magic", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    bytes[0] = b'X';
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_version_mismatch() {
    let file_path = write_sample_file("version_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(&mut bytes, VERSION_OFFSET, VERSION.saturating_add(1));
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_flags_mismatch() {
    let file_path = write_sample_file("flags_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(&mut bytes, FLAGS_OFFSET, FLAGS.saturating_add(1));
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_tokenizer_or_normalization_mismatch() {
    let file_path = write_sample_file("preprocess_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u32_at(
        &mut bytes,
        TOKENIZER_VERSION_OFFSET,
        TOKENIZER_VERSION.saturating_add(1),
    );
    fs::write(&file_path, &bytes)
        .await
        .expect("write should succeed");
    assert!(load_chain(&file_path).await.is_err());

    write_u32_at(&mut bytes, TOKENIZER_VERSION_OFFSET, TOKENIZER_VERSION);
    write_u32_at(
        &mut bytes,
        NORMALIZATION_FLAGS_OFFSET,
        NORMALIZATION_FLAGS.saturating_add(1),
    );
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");
    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_file_size_mismatch() {
    let file_path = write_sample_file("filesize_mismatch", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    let file_size = read_u64_at(&bytes, FILE_SIZE_OFFSET);
    write_u64_at(&mut bytes, FILE_SIZE_OFFSET, file_size.saturating_add(1));
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_checksum_mismatch() {
    let file_path = write_sample_file("checksum_non_zero", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    write_u64_at(&mut bytes, CHECKSUM_OFFSET, 1);
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn writes_non_zero_checksum() {
    let file_path = write_sample_file("checksum_written", &sample_chain()).await;
    let bytes = fs::read(&file_path).await.expect("read should succeed");

    let checksum = read_u64_at(&bytes, CHECKSUM_OFFSET);
    assert_ne!(checksum, 0);

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_non_zero_header_padding() {
    let file_path = write_sample_file("header_padding_corrupt", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let token_count = usize::try_from(read_u32_at(&bytes, TOKEN_COUNT_OFFSET)).expect("fits");
    let vocab_offsets_offset =
        usize::try_from(read_u64_at(&bytes, VOCAB_OFFSETS_OFFSET_OFFSET)).expect("offset fits");
    let vocab_blob_offset =
        usize::try_from(read_u64_at(&bytes, VOCAB_BLOB_OFFSET_OFFSET)).expect("offset fits");
    let start_offset =
        usize::try_from(read_u64_at(&bytes, START_OFFSET_OFFSET)).expect("offset fits");

    let vocab_offsets_end = vocab_offsets_offset
        .checked_add(
            token_count
                .checked_add(1)
                .expect("offset len overflow")
                .checked_mul(8)
                .expect("offset byte size overflow"),
        )
        .expect("offset end overflow");
    let vocab_blob_size =
        usize::try_from(read_u64_at(&bytes, vocab_offsets_offset + token_count * 8))
            .expect("blob size fits");
    let vocab_blob_end = vocab_blob_offset
        .checked_add(vocab_blob_size)
        .expect("blob end overflow");

    let header_aligned_end = HEADER_SIZE.next_multiple_of(8);
    let padding_ranges = [
        (header_aligned_end, vocab_offsets_offset),
        (vocab_offsets_end, vocab_blob_offset),
        (vocab_blob_end, start_offset),
    ];
    let (padding_start, _) = padding_ranges
        .into_iter()
        .find(|(start, end)| start < end)
        .expect("sample file should contain at least one padding range");

    bytes[padding_start] = 1;

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_overlapping_sections() {
    let file_path = write_sample_file("section_overlap", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");
    let token_count = read_u32_at(&bytes, TOKEN_COUNT_OFFSET);
    write_u32_at(
        &mut bytes,
        TOKEN_COUNT_OFFSET,
        token_count.saturating_add(1),
    );
    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_non_monotonic_edge_cumulative() {
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(&["a".to_owned()])
        .expect("training should succeed");
    chain
        .train_tokens(&["b".to_owned()])
        .expect("training should succeed");

    let file_path = write_sample_file("edge_non_monotonic", &chain).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let model3_edge_offset =
        usize::try_from(read_u64_at(&bytes, MODEL3_EDGE_OFFSET_OFFSET)).expect("offset fits");
    let first_cumulative = read_u64_at(&bytes, model3_edge_offset + 4);
    write_u64_at(&mut bytes, model3_edge_offset + 16, first_cumulative);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_start_prefix_id_out_of_bounds() {
    let file_path = write_sample_file("start_prefix_out_of_bounds", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let start_offset =
        usize::try_from(read_u64_at(&bytes, START_OFFSET_OFFSET)).expect("offset fits");
    write_u32_at(&mut bytes, start_offset, u32::MAX);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_broken_bos_token() {
    let file_path = write_sample_file("broken_bos", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let vocab_blob_offset =
        usize::try_from(read_u64_at(&bytes, VOCAB_BLOB_OFFSET_OFFSET)).expect("offset fits");
    bytes[vocab_blob_offset] = b'X';

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn rejects_model2_pair_range_out_of_bounds() {
    let file_path = write_sample_file("model2_pair_range_oob", &sample_chain()).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let model2_pair_offset =
        usize::try_from(read_u64_at(&bytes, MODEL2_PAIR_OFFSET_OFFSET)).expect("offset fits");
    write_u32_at(&mut bytes, model2_pair_offset + 8, u32::MAX);

    let checksum = compute_checksum(bytes.as_slice()).expect("checksum should be computed");
    write_u64_at(&mut bytes, CHECKSUM_OFFSET, checksum);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    assert!(load_chain(&file_path).await.is_err());

    let _ = fs::remove_file(file_path).await;
}

#[tokio::test]
async fn loads_cumulative_values_beyond_u32_max() {
    let mut chain = MarkovChain::default();
    chain
        .train_tokens(&["x".to_owned()])
        .expect("training should succeed");

    let file_path = write_sample_file("u64_cumulative", &chain).await;
    let mut bytes = fs::read(&file_path).await.expect("read should succeed");

    let start_offset =
        usize::try_from(read_u64_at(&bytes, START_OFFSET_OFFSET)).expect("offset fits");
    let model3_prefix_offset =
        usize::try_from(read_u64_at(&bytes, MODEL3_PREFIX_OFFSET_OFFSET)).expect("offset fits");
    let model3_edge_offset =
        usize::try_from(read_u64_at(&bytes, MODEL3_EDGE_OFFSET_OFFSET)).expect("offset fits");

    let huge = u64::from(u32::MAX) + 10;

    write_u64_at(&mut bytes, start_offset + 4, huge);
    write_u64_at(&mut bytes, model3_prefix_offset + 12, huge);
    write_u64_at(&mut bytes, model3_edge_offset + 4, huge);

    let checksum = compute_checksum(bytes.as_slice()).expect("checksum should be computed");
    write_u64_at(&mut bytes, CHECKSUM_OFFSET, checksum);

    fs::write(&file_path, bytes)
        .await
        .expect("write should succeed");

    let loaded = load_chain(&file_path).await.expect("load should succeed");
    assert_eq!(loaded.starts.get(&[BOS_ID, BOS_ID, BOS_ID]), Some(&huge));

    let edges = loaded
        .model3
        .get(&[BOS_ID, BOS_ID, BOS_ID])
        .expect("model3 prefix should exist");
    let total: u64 = edges.values().copied().sum();
    assert_eq!(total, huge);

    let _ = fs::remove_file(file_path).await;
}

fn sample_chain() -> MarkovChain {
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
    chain
}

async fn write_sample_file(prefix: &str, chain: &MarkovChain) -> TempPath {
    let file_path = temp_file_path(prefix);
    save_chain(&file_path, chain)
        .await
        .expect("save should succeed");
    file_path
}

fn write_u32_at(bytes: &mut [u8], offset: usize, value: u32) {
    let end = offset + 4;
    bytes[offset..end].copy_from_slice(value.to_le_bytes().as_slice());
}

fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    let end = offset + 4;
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(&bytes[offset..end]);
    u32::from_le_bytes(raw)
}

fn write_u64_at(bytes: &mut [u8], offset: usize, value: u64) {
    let end = offset + 8;
    bytes[offset..end].copy_from_slice(value.to_le_bytes().as_slice());
}

fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    let end = offset + 8;
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(&bytes[offset..end]);
    u64::from_le_bytes(raw)
}

fn temp_file_path(prefix: &str) -> TempPath {
    Builder::new()
        .prefix(&format!("markov_bot_{prefix}_"))
        .suffix(".mkv3")
        .tempfile()
        .expect("tempfile should be created")
        .into_temp_path()
}
