use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, Command};
use markov_storage::{StorageSnapshot, decode_v8_snapshot, encode_v8_snapshot};

mod legacy_v6;

const STORAGE_MAGIC: [u8; 8] = *b"MKV3BIN\0";

fn main() -> Result<()> {
    let matches = Command::new("markov-storage")
        .about("Inspect, migrate, and edit markov storage files")
        .subcommand(
            Command::new("inspect").arg(
                Arg::new("input")
                    .long("input")
                    .required(true),
            ),
        )
        .subcommand(
            Command::new("export")
                .arg(
                    Arg::new("input")
                        .long("input")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("import")
                .arg(
                    Arg::new("input")
                        .long("input")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("migrate")
                .arg(
                    Arg::new("input")
                        .long("input")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .required(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("inspect", sub)) => {
            let input = sub
                .get_one::<String>("input")
                .ok_or_else(|| anyhow!("missing input"))?;
            inspect_command(Path::new(input))
        }
        Some(("export", sub)) => {
            let input = sub
                .get_one::<String>("input")
                .ok_or_else(|| anyhow!("missing input"))?;
            let output = sub
                .get_one::<String>("output")
                .ok_or_else(|| anyhow!("missing output"))?;
            export_command(Path::new(input), Path::new(output))
        }
        Some(("import", sub)) => {
            let input = sub
                .get_one::<String>("input")
                .ok_or_else(|| anyhow!("missing input"))?;
            let output = sub
                .get_one::<String>("output")
                .ok_or_else(|| anyhow!("missing output"))?;
            import_command(Path::new(input), Path::new(output))
        }
        Some(("migrate", sub)) => {
            let input = sub
                .get_one::<String>("input")
                .ok_or_else(|| anyhow!("missing input"))?;
            let output = sub
                .get_one::<String>("output")
                .ok_or_else(|| anyhow!("missing output"))?;
            migrate_command(Path::new(input), Path::new(output))
        }
        _ => bail!("no subcommand provided"),
    }
}

fn inspect_command(input: &Path) -> Result<()> {
    let bytes = read_bytes(input)?;
    let snapshot = decode_snapshot(bytes.as_slice())?;

    println!("version={}", snapshot.source.storage_version);
    println!("compression={}", snapshot.source.compression.as_env_value());
    println!("ngram_order={}", snapshot.source.ngram_order);
    println!("token_count={}", snapshot.tokens.len());
    println!("start_count={}", snapshot.starts.len());
    for model in &snapshot.models {
        let edge_count = model
            .entries
            .iter()
            .map(|entry| entry.edges.len())
            .sum::<usize>();
        println!(
            "model[order={}]: entries={}, edges={}",
            model.order,
            model.entries.len(),
            edge_count
        );
    }

    Ok(())
}

fn export_command(input: &Path, output: &Path) -> Result<()> {
    ensure_distinct_paths(input, output)?;
    let bytes = read_bytes(input)?;
    let snapshot = decode_snapshot(bytes.as_slice())?;
    write_json(output, &snapshot)
}

fn import_command(input: &Path, output: &Path) -> Result<()> {
    ensure_distinct_paths(input, output)?;
    let snapshot = read_json(input)?;
    let payload = encode_v8_snapshot(&snapshot, snapshot.source.compression)?;
    write_bytes(output, payload.as_slice())
}

fn migrate_command(input: &Path, output: &Path) -> Result<()> {
    ensure_distinct_paths(input, output)?;
    let bytes = read_bytes(input)?;
    let version = detect_storage_version(bytes.as_slice())?;
    if version != 6 {
        bail!("migrate only accepts v6 input, got v{version}");
    }

    let snapshot = legacy_v6::decode_snapshot(bytes.as_slice())?;
    let payload = encode_v8_snapshot(&snapshot, snapshot.source.compression)?;
    write_bytes(output, payload.as_slice())
}

fn decode_snapshot(bytes: &[u8]) -> Result<StorageSnapshot> {
    match detect_storage_version(bytes)? {
        6 => legacy_v6::decode_snapshot(bytes),
        8 => decode_v8_snapshot(bytes).map_err(Into::into),
        version => bail!("unsupported storage version: {version}"),
    }
}

fn detect_storage_version(bytes: &[u8]) -> Result<u32> {
    let magic = bytes
        .get(..STORAGE_MAGIC.len())
        .ok_or_else(|| anyhow!("storage file is shorter than the header"))?;
    if magic != STORAGE_MAGIC {
        bail!("storage magic mismatch");
    }

    let version_bytes = bytes
        .get(STORAGE_MAGIC.len()..(STORAGE_MAGIC.len() + 4))
        .ok_or_else(|| anyhow!("storage file is missing version"))?;
    let version = <[u8; 4]>::try_from(version_bytes)
        .map(u32::from_le_bytes)
        .map_err(|_error| anyhow!("storage version bytes are invalid"))?;

    Ok(version)
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn read_json(path: &Path) -> Result<StorageSnapshot> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader)
        .with_context(|| format!("failed to parse JSON from {}", path.display()))
}

fn write_json(path: &Path, snapshot: &StorageSnapshot) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, snapshot)
        .with_context(|| format!("failed to write JSON to {}", path.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn ensure_distinct_paths(input: &Path, output: &Path) -> Result<()> {
    if input == output {
        bail!("input and output paths must differ");
    }

    Ok(())
}
