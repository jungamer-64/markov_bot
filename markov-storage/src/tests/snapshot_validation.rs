use crate::{
    SnapshotEdge, SnapshotEntry, SnapshotModel, SnapshotModelEntry, SnapshotSource,
    StorageCompressionMode, StorageSnapshot, encode_snapshot,
};

use super::helpers::ensure;

fn valid_snapshot() -> StorageSnapshot {
    StorageSnapshot {
        schema_version: 1,
        source: SnapshotSource {
            storage_version: 8,
            ngram_order: 2,
            compression: StorageCompressionMode::Uncompressed,
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
    *snapshot.tokens.get_mut(0).ok_or("tokens[0] missing")? = "wrong".to_owned();
    ensure(
        encode_snapshot(&snapshot, StorageCompressionMode::Uncompressed).is_err(),
        "snapshot without BOS should be rejected",
    )
}

#[test]
fn rejects_prefix_length_mismatch() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot.starts.get_mut(0).ok_or("starts[0] missing")?.prefix = vec![2];
    ensure(
        encode_snapshot(&snapshot, StorageCompressionMode::Uncompressed).is_err(),
        "snapshot start prefix length mismatch should be rejected",
    )
}

#[test]
fn rejects_out_of_range_token_id() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot.models.get_mut(0).ok_or("models[0] missing")?
        .entries.get_mut(0).ok_or("models[0].entries[0] missing")?
        .edges.get_mut(0).ok_or("models[0].entries[0].edges[0] missing")?.next = 99;
    ensure(
        encode_snapshot(&snapshot, StorageCompressionMode::Uncompressed).is_err(),
        "snapshot edge token id out of range should be rejected",
    )
}

#[test]
fn rejects_zero_count() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot.models.get_mut(0).ok_or("models[0] missing")?
        .entries.get_mut(0).ok_or("models[0].entries[0] missing")?
        .edges.get_mut(0).ok_or("models[0].entries[0].edges[0] missing")?.count = 0;
    ensure(
        encode_snapshot(&snapshot, StorageCompressionMode::Uncompressed).is_err(),
        "snapshot zero count should be rejected",
    )
}

#[test]
fn rejects_duplicate_edge_target() -> Result<(), crate::StorageError> {
    let mut snapshot = valid_snapshot();
    snapshot.models.get_mut(0).ok_or("models[0] missing")?
        .entries.get_mut(0).ok_or("models[0].entries[0] missing")?
        .edges
        .push(SnapshotEdge { next: 1, count: 2 });
    ensure(
        encode_snapshot(&snapshot, StorageCompressionMode::Uncompressed).is_err(),
        "snapshot duplicate edge target should be rejected",
    )
}
