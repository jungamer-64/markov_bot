use std::{fs, process::Command};

use anyhow::{Result, anyhow, bail, ensure};
use markov_storage::{
    SnapshotEdge, SnapshotEntry, SnapshotModel, SnapshotModelEntry, SnapshotSource,
    StorageCompressionMode, StorageSnapshot, decode_snapshot, encode_snapshot,
};
use tempfile::tempdir;

#[test]
fn inspect_prints_summary() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    fs::write(
        &input,
        encode_snapshot(&sample_snapshot(), StorageCompressionMode::Uncompressed)?,
    )?;

    let output = run_cli(&["inspect", "--input", input.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8(output.stdout)?;

    ensure!(stdout.contains("version=8"), "stdout should contain version=8");
    ensure!(stdout.contains("ngram_order=2"), "stdout should contain ngram_order=2");
    ensure!(stdout.contains("model[order=2]: entries=1, edges=1"), "stdout should contain model summary");
    Ok(())
}

#[test]
fn export_then_import_round_trips_snapshot() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    let exported = temp_dir.path().join("snapshot.json");
    let output = temp_dir.path().join("output.mkv3");
    let expected = sample_snapshot();

    fs::write(
        &input,
        encode_snapshot(&expected, StorageCompressionMode::Uncompressed)?,
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

    let rebuilt = decode_snapshot(fs::read(&output)?.as_slice())?;
    ensure!(rebuilt.tokens == expected.tokens, "tokens mismatch");
    ensure!(rebuilt.starts == expected.starts, "starts mismatch");
    ensure!(rebuilt.models == expected.models, "models mismatch");
    ensure!(rebuilt.source.ngram_order == expected.source.ngram_order, "ngram_order mismatch");
    Ok(())
}

#[test]
fn rejects_same_input_and_output_path() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    fs::write(
        &input,
        encode_snapshot(&sample_snapshot(), StorageCompressionMode::Uncompressed)?,
    )?;

    let output = Command::new(binary_path())
        .args([
            "export",
            "--input",
            input.to_string_lossy().as_ref(),
            "--output",
            input.to_string_lossy().as_ref(),
        ])
        .output()?;
    if output.status.success() {
        bail!("export should reject identical input and output paths");
    }

    let stderr = String::from_utf8(output.stderr)?;
    ensure!(stderr.contains("input and output paths must differ"), "stderr should contain error message");
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

const fn binary_path() -> &'static str {
    env!("CARGO_BIN_EXE_markov-storage")
}

fn sample_snapshot() -> StorageSnapshot {
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
