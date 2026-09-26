use crate::Codec;
use crate::{
    SnapshotEdge, SnapshotEntry, SnapshotModel, SnapshotModelEntry, SnapshotSource,
    StorageCompressionMode, StorageError, StorageSnapshot,
};

use super::test_support::ensure;

fn valid_snapshot() -> StorageSnapshot {
    StorageSnapshot {
        schema_version: 1,
        source: SnapshotSource {
            storage_version: 8,
            ngram_order: 2,
            compression: crate::StorageCompression::Uncompressed,
        },
        tokens: vec!["<BOS>".to_owned(), "<EOS>".to_owned(), "a".to_owned()],
        starts: vec![SnapshotEntry {
            prefix: vec![0, 2],
            count: 1,
        }],
        models: vec![
            SnapshotModel {
                order: 2,
                entries: vec![SnapshotModelEntry {
                    prefix: vec![0, 2],
                    edges: vec![SnapshotEdge { next: 1, count: 1 }],
                }],
            },
            SnapshotModel {
                order: 1,
                entries: vec![SnapshotModelEntry {
                    prefix: vec![2],
                    edges: vec![SnapshotEdge { next: 1, count: 1 }],
                }],
            },
        ],
    }
}

#[test]
fn rejects_missing_special_tokens() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    *snapshot
        .tokens
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("tokens[0] missing".to_owned()))? = "wrong".to_owned();
    ensure(
        Codec::default()
            .encode_snapshot(snapshot.clone(), StorageCompressionMode::Uncompressed)
            .is_err(),
        "snapshot without BOS should be rejected",
    )
}

#[test]
fn rejects_prefix_length_mismatch() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot
        .starts
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("starts[0] missing".to_owned()))?
        .prefix = vec![2];
    ensure(
        Codec::default()
            .encode_snapshot(snapshot.clone(), StorageCompressionMode::Uncompressed)
            .is_err(),
        "snapshot start prefix length mismatch should be rejected",
    )
}

#[test]
fn rejects_out_of_range_token_id() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot
        .models
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0] missing".to_owned()))?
        .entries
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0].entries[0] missing".to_owned()))?
        .edges
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0].entries[0].edges[0] missing".to_owned()))?
        .next = 99;
    ensure(
        Codec::default()
            .encode_snapshot(snapshot.clone(), StorageCompressionMode::Uncompressed)
            .is_err(),
        "snapshot edge token id out of range should be rejected",
    )
}

#[test]
fn rejects_zero_count() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot
        .models
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0] missing".to_owned()))?
        .entries
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0].entries[0] missing".to_owned()))?
        .edges
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0].entries[0].edges[0] missing".to_owned()))?
        .count = 0;
    ensure(
        Codec::default()
            .encode_snapshot(snapshot.clone(), StorageCompressionMode::Uncompressed)
            .is_err(),
        "snapshot zero count should be rejected",
    )
}

#[test]
fn rejects_duplicate_edge_target() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot
        .models
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0] missing".to_owned()))?
        .entries
        .get_mut(0)
        .ok_or_else(|| StorageError::Format("models[0].entries[0] missing".to_owned()))?
        .edges
        .push(SnapshotEdge { next: 1, count: 2 });
    ensure(
        Codec::default()
            .encode_snapshot(snapshot.clone(), StorageCompressionMode::Uncompressed)
            .is_err(),
        "snapshot duplicate edge target should be rejected",
    )
}

#[test]
fn rejects_duplicate_model_orders_and_empty_edges() -> Result<(), StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot.models.push(
        snapshot
            .models
            .first()
            .ok_or_else(|| StorageError::Format("missing model".into()))?
            .clone(),
    );
    ensure(
        Codec::default().snapshot_to_chain(snapshot).is_err(),
        "duplicate models cannot overwrite",
    )?;
    let mut snapshot = valid_snapshot();
    for model in &mut snapshot.models {
        for entry in &mut model.entries {
            entry.edges.clear();
        }
    }
    ensure(
        Codec::default().snapshot_to_chain(snapshot).is_err(),
        "empty edge sets cannot enter core",
    )
}

#[test]
fn rejects_cumulative_count_overflow() -> Result<(), StorageError> {
    let mut snapshot = valid_snapshot();
    let model = snapshot
        .models
        .first_mut()
        .ok_or_else(|| StorageError::Format("missing model".into()))?;
    let entry = model
        .entries
        .first_mut()
        .ok_or_else(|| StorageError::Format("missing entry".into()))?;
    entry.edges = vec![
        SnapshotEdge {
            next: 1,
            count: u64::MAX,
        },
        SnapshotEdge { next: 2, count: 1 },
    ];
    ensure(
        Codec::default().snapshot_to_chain(snapshot).is_err(),
        "edge cumulative overflow rejected at construction",
    )?;
    let mut snapshot = valid_snapshot();
    snapshot.starts.push(SnapshotEntry {
        prefix: vec![0, 0],
        count: u64::MAX,
    });
    ensure(
        Codec::default().snapshot_to_chain(snapshot).is_err(),
        "start cumulative overflow rejected at construction",
    )
}

#[test]
fn start_generation_can_fall_back_to_lower_order() -> Result<(), StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot
        .models
        .first_mut()
        .ok_or_else(|| StorageError::Format("missing model".into()))?
        .entries
        .clear();
    let chain = Codec::default().snapshot_to_chain(snapshot)?;
    let options = markov_core::GenerationOptions::new(
        markov_core::MaxWords::new(3)?,
        markov_core::Temperature::new(1.0)?,
        markov_core::MinWordsBeforeEos::new(0),
    )?;
    ensure(
        chain
            .generate_sentence_with_options(&mut rand::rng(), options)
            .as_deref()
            == Some("a"),
        "start remains valid when generation uses a lower order",
    )
}
