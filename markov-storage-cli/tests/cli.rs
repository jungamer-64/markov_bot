use markov_storage::Codec;
use std::{fs, process::Command};

use anyhow::{Result, anyhow, bail, ensure};
use markov_storage::{
    SnapshotEdge, SnapshotEntry, SnapshotModel, SnapshotModelEntry, SnapshotSource,
    StorageCompressionMode, StorageSnapshot,
};
use tempfile::tempdir;

#[test]
fn inspect_prints_summary() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    fs::write(
        &input,
        Codec::default()
            .encode_snapshot(sample_snapshot(), StorageCompressionMode::Uncompressed)?,
    )?;

    let output = run_cli(&["inspect", "--input", input.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8(output.stdout)?;

    ensure!(
        stdout.contains("version=8"),
        "stdout should contain version=8"
    );
    ensure!(
        stdout.contains("ngram_order=2"),
        "stdout should contain ngram_order=2"
    );
    ensure!(
        stdout.contains("model[order=2]: entries=1, edges=1"),
        "stdout should contain model summary"
    );
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
        Codec::default().encode_snapshot(expected.clone(), StorageCompressionMode::Uncompressed)?,
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

    let rebuilt = Codec::default().decode_snapshot(fs::read(&output)?.as_slice())?;
    ensure!(rebuilt.tokens == expected.tokens, "tokens mismatch");
    ensure!(rebuilt.starts == expected.starts, "starts mismatch");
    ensure!(rebuilt.models == expected.models, "models mismatch");
    ensure!(
        rebuilt.source.ngram_order == expected.source.ngram_order,
        "ngram_order mismatch"
    );
    Ok(())
}

#[test]
fn rejects_same_input_and_output_path() -> Result<()> {
    let temp_dir = tempdir()?;
    let input = temp_dir.path().join("input.mkv3");
    fs::write(
        &input,
        Codec::default()
            .encode_snapshot(sample_snapshot(), StorageCompressionMode::Uncompressed)?,
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
    ensure!(
        stderr.contains("input and output refer to the same file"),
        "stderr should contain error message"
    );
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
            compression: markov_storage::StorageCompression::Uncompressed,
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
fn output_limits_aliases_and_writer_contention_leave_file_intact() -> Result<()> {
    let directory = tempdir()?;
    let input = directory.path().join("input.mkv3");
    let alias = directory.path().join("alias.mkv3");
    let output = directory.path().join("output.json");
    let bytes = Codec::default()
        .encode_snapshot(sample_snapshot(), StorageCompressionMode::Uncompressed)?;
    fs::write(&input, &bytes)?;
    fs::hard_link(&input, &alias)?;
    let result = Command::new(binary_path())
        .arg("export")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&alias)
        .output()?;
    ensure!(!result.status.success(), "hard-link alias must fail");
    ensure!(fs::read(&input)? == bytes, "input preserved");
    fs::write(&output, b"previous")?;
    let _lease =
        markov_storage::FileStore::open(&output, markov_storage::StorageLimits::default())?;
    let result = Command::new(binary_path())
        .arg("export")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()?;
    ensure!(!result.status.success(), "active writer must fail");
    ensure!(fs::read(&output)? == b"previous", "output preserved");
    let result = Command::new(binary_path())
        .arg("inspect")
        .arg("--input")
        .arg(&input)
        .args(["--max-file-bytes", "1"])
        .output()?;
    ensure!(!result.status.success(), "input limit enforced");
    Ok(())
}

// Linux syscall injection observes the real publication path without adding any
// test hooks to the library. CI installs strace; these are process-crash tests,
// not evidence of power-loss behavior of the host filesystem or physical drive.
#[cfg(target_os = "linux")]
#[test]
fn publication_failures_and_process_crashes_preserve_complete_generations() -> Result<()> {
    let directory = tempdir()?;
    let input = directory.path().join("input.json");
    let output = directory.path().join("model.mkv3");
    let trace = directory.path().join("trace");
    fs::write(&input, serde_json::to_vec(&sample_snapshot())?)?;
    let expected = Codec::default()
        .encode_snapshot(sample_snapshot(), StorageCompressionMode::Uncompressed)?;
    let mut earlier = sample_snapshot();
    earlier.starts.clear();
    for model in &mut earlier.models {
        model.entries.clear();
    }
    let previous =
        Codec::default().encode_snapshot(earlier, StorageCompressionMode::Uncompressed)?;
    let previous = previous.as_slice();
    let invoke = |injection: &str| trace_import(&input, &output, &trace, injection);
    // Locate exclusive temporary creation from a real successful run. Loader openat
    // counts vary by platform, so do not assume a fixed syscall ordinal.
    fs::write(&output, previous)?;
    let baseline = invoke("inject=openat:error=EACCES:when=65535")?;
    ensure!(
        baseline.status.success(),
        "baseline publication must succeed"
    );
    let create_failure = temporary_creation_failure(&trace)?;
    for (injection, stage, published) in [
        (create_failure.as_str(), "CreateTemporary", false),
        ("inject=write:error=ENOSPC:when=1", "Write", false),
        ("inject=fsync:error=EIO:when=2", "SyncFile", false),
        (
            "inject=rename,renameat,renameat2:error=EIO:when=1",
            "Replace",
            false,
        ),
        ("inject=fsync:error=EIO:when=3", "SyncDirectory", true),
    ] {
        fs::write(&output, previous)?;
        let result = invoke(injection)?;
        ensure!(
            !result.status.success(),
            "injected {stage} failure must fail"
        );
        let stderr = String::from_utf8_lossy(&result.stderr);
        ensure!(
            stderr.contains(stage),
            "expected {stage}: {stderr}; trace={}",
            fs::read_to_string(&trace)?
        );
        ensure!(
            fs::read(&output)?
                == if published {
                    expected.as_slice()
                } else {
                    previous
                },
            "generation after {stage}"
        );
        if published {
            ensure!(
                stderr.contains("PublishedNotDurable"),
                "post-publication classification"
            );
        } else if stage == "Replace" {
            ensure!(
                stderr.contains("Unknown"),
                "uncertain replacement classification"
            );
        } else {
            ensure!(
                stderr.contains("NotPublished"),
                "pre-publication classification"
            );
        }
    }
    for (injection, published) in [
        (
            "inject=rename,renameat,renameat2:signal=SIGKILL:when=1",
            false,
        ),
        ("inject=fsync:signal=SIGKILL:when=3", true),
    ] {
        fs::write(&output, previous)?;
        let result = invoke(injection)?;
        ensure!(!result.status.success(), "child must be killed");
        ensure!(
            fs::read(&output)?
                == if published {
                    expected.as_slice()
                } else {
                    previous
                },
            "crash must retain one complete generation"
        );
        // Kernel releases the writer lock on abrupt termination, despite the sidecar remaining.
        let reopened =
            markov_storage::FileStore::open(&output, markov_storage::StorageLimits::default())?;
        let decoded = Codec::default().decode_snapshot(&fs::read(&output)?)?;
        let chain = Codec::default().snapshot_to_chain(decoded)?;
        reopened.load(chain.order())?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn trace_import(
    input: &std::path::Path,
    output: &std::path::Path,
    trace: &std::path::Path,
    injection: &str,
) -> Result<std::process::Output> {
    Ok(Command::new("strace")
        .args([
            "-qq",
            "-e",
            "trace=fsync,rename,renameat,renameat2,write,openat",
            "-e",
            injection,
            "-o",
        ])
        .arg(trace)
        .arg("--")
        .arg(binary_path())
        .arg("import")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .output()?)
}

#[cfg(target_os = "linux")]
fn temporary_creation_failure(trace: &std::path::Path) -> Result<String> {
    let creation_call = fs::read_to_string(trace)?
        .lines()
        .filter(|line| line.starts_with("openat("))
        .position(|line| line.contains(".markov-") && line.contains("O_EXCL"))
        .ok_or_else(|| anyhow!("temporary creation missing from syscall trace"))?
        + 1;
    Ok(format!("inject=openat:error=EACCES:when={creation_call}"))
}
