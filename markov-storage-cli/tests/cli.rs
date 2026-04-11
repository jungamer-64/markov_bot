use std::{fs, process::Command};

use anyhow::{Result, anyhow, bail};
use markov_storage::{
    SnapshotEdge, SnapshotEntry, SnapshotModel, SnapshotModelEntry, SnapshotSource,
    StorageCompressionMode, StorageSnapshot, decode_v8_snapshot, encode_v8_snapshot,
};
use tempfile::tempdir;

const MAGIC: [u8; 8] = *b"MKV3BIN\0";
const HEADER_SIZE: usize = 44;
const DESCRIPTOR_SIZE: usize = 24;
const SECTION_COUNT: usize = 20;
const CHECKSUM_OFFSET: usize = HEADER_SIZE - std::mem::size_of::<u64>();
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[test]
fn inspect_v8_prints_summary() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    fs::write(
        &input,
        encode_v8_snapshot(&sample_v8_snapshot(), StorageCompressionMode::Uncompressed)?,
    )?;

    let output = run_cli(&["inspect", "--input", input.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8(output.stdout)?;

    assert!(stdout.contains("version=8"));
    assert!(stdout.contains("ngram_order=2"));
    assert!(stdout.contains("model[order=2]: entries=1, edges=1"));
    Ok(())
}

#[test]
fn export_then_import_round_trips_v8_snapshot() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    let exported = temp_dir.path().join("snapshot.json");
    let output = temp_dir.path().join("output.mkv3");
    let expected = sample_v8_snapshot();

    fs::write(
        &input,
        encode_v8_snapshot(&expected, StorageCompressionMode::Uncompressed)?,
    )?;

    run_cli(&[
        "export",
        "--input",
        input.to_string_lossy().as_ref(),
        "--output",
        exported.to_string_lossy().as_ref(),
    ])?;
    run_cli(&[
        "import",
        "--input",
        exported.to_string_lossy().as_ref(),
        "--output",
        output.to_string_lossy().as_ref(),
    ])?;

    let rebuilt = decode_v8_snapshot(fs::read(&output)?.as_slice())?;
    assert_eq!(rebuilt.tokens, expected.tokens);
    assert_eq!(rebuilt.starts, expected.starts);
    assert_eq!(rebuilt.models, expected.models);
    assert_eq!(rebuilt.source.ngram_order, expected.source.ngram_order);
    Ok(())
}

#[test]
fn inspect_and_migrate_v6_fixture() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("legacy.mkv3");
    let migrated = temp_dir.path().join("migrated.mkv3");
    fs::write(&input, build_v6_fixture()?)?;

    let inspect = run_cli(&["inspect", "--input", input.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8(inspect.stdout)?;
    assert!(stdout.contains("version=6"));
    assert!(stdout.contains("ngram_order=6"));
    assert!(stdout.contains("model[order=6]: entries=1, edges=1"));

    run_cli(&[
        "migrate",
        "--input",
        input.to_string_lossy().as_ref(),
        "--output",
        migrated.to_string_lossy().as_ref(),
    ])?;

    let rebuilt = decode_v8_snapshot(fs::read(&migrated)?.as_slice())?;
    let expected = expected_v6_snapshot();
    assert_eq!(rebuilt.tokens, expected.tokens);
    assert_eq!(rebuilt.starts, expected.starts);
    assert_eq!(rebuilt.models, expected.models);
    assert_eq!(rebuilt.source.storage_version, 8);
    Ok(())
}

#[test]
fn export_v6_then_import_preserves_semantics() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("legacy.mkv3");
    let exported = temp_dir.path().join("legacy.json");
    let output = temp_dir.path().join("imported.mkv3");
    fs::write(&input, build_v6_fixture()?)?;

    run_cli(&[
        "export",
        "--input",
        input.to_string_lossy().as_ref(),
        "--output",
        exported.to_string_lossy().as_ref(),
    ])?;

    let exported_snapshot: StorageSnapshot =
        serde_json::from_slice(fs::read(&exported)?.as_slice())?;
    assert_eq!(exported_snapshot.source.storage_version, 6);
    assert_eq!(exported_snapshot.source.ngram_order, 6);

    run_cli(&[
        "import",
        "--input",
        exported.to_string_lossy().as_ref(),
        "--output",
        output.to_string_lossy().as_ref(),
    ])?;

    let rebuilt = decode_v8_snapshot(fs::read(&output)?.as_slice())?;
    let expected = expected_v6_snapshot();
    assert_eq!(rebuilt.tokens, expected.tokens);
    assert_eq!(rebuilt.starts, expected.starts);
    assert_eq!(rebuilt.models, expected.models);
    Ok(())
}

#[test]
fn rejects_same_input_and_output_path() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("legacy.mkv3");
    fs::write(&input, build_v6_fixture()?)?;

    let output = Command::new(binary_path())
        .args([
            "migrate",
            "--input",
            input.to_string_lossy().as_ref(),
            "--output",
            input.to_string_lossy().as_ref(),
        ])
        .output()?;
    if output.status.success() {
        bail!("migrate should reject identical input and output paths");
    }

    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("input and output paths must differ"));
    Ok(())
}

fn run_cli(args: &[&str]) -> Result<std::process::Output> {
    let output = Command::new(binary_path()).args(args).output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "command failed: {}\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output)
}

fn binary_path() -> &'static str {
    env!("CARGO_BIN_EXE_markov-storage")
}

fn sample_v8_snapshot() -> StorageSnapshot {
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

fn expected_v6_snapshot() -> StorageSnapshot {
    let make_model = |order: usize, prefix: Vec<u32>| SnapshotModel {
        order,
        entries: vec![SnapshotModelEntry {
            prefix,
            edges: vec![SnapshotEdge { next: 1, count: 1 }],
        }],
    };

    StorageSnapshot {
        schema_version: 1,
        source: SnapshotSource {
            storage_version: 6,
            ngram_order: 6,
            compression: StorageCompressionMode::Uncompressed,
        },
        tokens: vec!["<BOS>".to_owned(), "<EOS>".to_owned(), "a".to_owned()],
        starts: vec![SnapshotEntry {
            prefix: vec![0, 0, 0, 0, 0, 2],
            count: 1,
        }],
        models: vec![
            make_model(6, vec![0, 0, 0, 0, 0, 2]),
            make_model(5, vec![0, 0, 0, 0, 2]),
            make_model(4, vec![0, 0, 0, 2]),
            make_model(3, vec![0, 0, 2]),
            make_model(2, vec![0, 2]),
            make_model(1, vec![2]),
        ],
    }
}

fn build_v6_fixture() -> Result<Vec<u8>> {
    let vocab_blob = b"<BOS><EOS>a".to_vec();
    let vocab_offsets = encode_u64_values(&[0, 5, 10, 11]);
    let starts = encode_start_records(&[(0, 1)]);
    let model6_pairs = encode_u32_values(&[0, 0, 0, 0, 0, 0, 1]);
    let model6_prefixes = encode_u32_u32_u32_u64_records(&[(2, 0, 1, 1)]);
    let model6_edges = encode_edge_records(&[(1, 1)]);
    let model5_pairs = encode_u32_values(&[0, 0, 0, 0, 0, 1]);
    let model5_prefixes = encode_u32_u32_u32_u64_records(&[(2, 0, 1, 1)]);
    let model5_edges = encode_edge_records(&[(1, 1)]);
    let model4_pairs = encode_u32_values(&[0, 0, 0, 0, 1]);
    let model4_prefixes = encode_u32_u32_u32_u64_records(&[(2, 0, 1, 1)]);
    let model4_edges = encode_edge_records(&[(1, 1)]);
    let model3_pairs = encode_u32_values(&[0, 0, 0, 1]);
    let model3_prefixes = encode_u32_u32_u32_u64_records(&[(2, 0, 1, 1)]);
    let model3_edges = encode_edge_records(&[(1, 1)]);
    let model2_pairs = encode_u32_values(&[0, 0, 1]);
    let model2_prefixes = encode_u32_u32_u32_u32_u64_records(&[(0, 2, 0, 1, 1)]);
    let model2_edges = encode_edge_records(&[(1, 1)]);
    let model1_prefixes = encode_u32_u32_u32_u64_records(&[(2, 0, 1, 1)]);
    let model1_edges = encode_edge_records(&[(1, 1)]);

    let sections = [
        (1_u32, vocab_offsets),
        (2_u32, vocab_blob),
        (3_u32, starts),
        (4_u32, model6_pairs),
        (5_u32, model6_prefixes),
        (6_u32, model6_edges),
        (7_u32, model5_pairs),
        (8_u32, model5_prefixes),
        (9_u32, model5_edges),
        (10_u32, model4_pairs),
        (11_u32, model4_prefixes),
        (12_u32, model4_edges),
        (13_u32, model3_pairs),
        (14_u32, model3_prefixes),
        (15_u32, model3_edges),
        (16_u32, model2_pairs),
        (17_u32, model2_prefixes),
        (18_u32, model2_edges),
        (19_u32, model1_prefixes),
        (20_u32, model1_edges),
    ];

    let metadata_end = (HEADER_SIZE + DESCRIPTOR_SIZE * SECTION_COUNT).next_multiple_of(8);
    let mut descriptors = Vec::with_capacity(SECTION_COUNT);
    let mut current_offset =
        u64::try_from(metadata_end).map_err(|_error| anyhow!("metadata end exceeds u64 range"))?;

    for (kind, body) in &sections {
        current_offset = align_to_eight(current_offset);
        descriptors.push((*kind, current_offset, u64::try_from(body.len())?));
        current_offset = current_offset
            .checked_add(u64::try_from(body.len())?)
            .ok_or_else(|| anyhow!("file size overflow"))?;
    }

    let file_size = current_offset;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC.as_slice());
    write_u32(&mut bytes, 6);
    write_u32(&mut bytes, 0);
    write_u32(&mut bytes, 1);
    write_u32(&mut bytes, 0);
    write_u32(&mut bytes, 20);
    write_u64(&mut bytes, file_size);
    write_u64(&mut bytes, 0);

    for (kind, offset, size) in &descriptors {
        write_u32(&mut bytes, *kind);
        write_u32(&mut bytes, 0);
        write_u64(&mut bytes, *offset);
        write_u64(&mut bytes, *size);
    }

    pad_to_offset(&mut bytes, metadata_end)?;
    for ((_, body), (_, offset, _)) in sections.iter().zip(descriptors.iter()) {
        let offset = usize::try_from(*offset)
            .map_err(|_error| anyhow!("section offset exceeds usize range"))?;
        pad_to_offset(&mut bytes, offset)?;
        bytes.extend_from_slice(body.as_slice());
    }

    let checksum = compute_checksum(bytes.as_slice())?;
    bytes[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 8].copy_from_slice(checksum.to_le_bytes().as_slice());

    Ok(bytes)
}

fn encode_u64_values(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        write_u64(&mut bytes, *value);
    }
    bytes
}

fn encode_u32_values(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        write_u32(&mut bytes, *value);
    }
    bytes
}

fn encode_start_records(records: &[(u32, u64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (prefix_id, cumulative) in records {
        write_u32(&mut bytes, *prefix_id);
        write_u64(&mut bytes, *cumulative);
    }
    bytes
}

fn encode_u32_u32_u32_u64_records(records: &[(u32, u32, u32, u64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (w, edge_start, edge_len, total) in records {
        write_u32(&mut bytes, *w);
        write_u32(&mut bytes, *edge_start);
        write_u32(&mut bytes, *edge_len);
        write_u64(&mut bytes, *total);
    }
    bytes
}

fn encode_u32_u32_u32_u32_u64_records(records: &[(u32, u32, u32, u32, u64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (w1, w2, edge_start, edge_len, total) in records {
        write_u32(&mut bytes, *w1);
        write_u32(&mut bytes, *w2);
        write_u32(&mut bytes, *edge_start);
        write_u32(&mut bytes, *edge_len);
        write_u64(&mut bytes, *total);
    }
    bytes
}

fn encode_edge_records(records: &[(u32, u64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (next, cumulative) in records {
        write_u32(&mut bytes, *next);
        write_u64(&mut bytes, *cumulative);
    }
    bytes
}

fn pad_to_offset(target: &mut Vec<u8>, offset: usize) -> Result<()> {
    if target.len() > offset {
        bail!("target already exceeds requested offset");
    }
    target.resize(offset, 0);
    Ok(())
}

const fn align_to_eight(value: u64) -> u64 {
    value.next_multiple_of(8)
}

fn compute_checksum(bytes: &[u8]) -> Result<u64> {
    let mut hash = FNV1A64_OFFSET_BASIS;
    for (index, byte) in bytes.iter().enumerate() {
        let normalized = if (CHECKSUM_OFFSET..CHECKSUM_OFFSET + 8).contains(&index) {
            0_u8
        } else {
            *byte
        };
        hash ^= u64::from(normalized);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    Ok(hash)
}

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}
