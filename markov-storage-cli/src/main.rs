use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, Command};
use markov_storage::{StorageSnapshot, decode_snapshot as decode_storage_snapshot, encode_snapshot as encode_storage_snapshot};

fn main() -> Result<()> {
    let matches = Command::new("markov-storage")
        .about("Inspect, and edit markov storage files")
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
        _ => bail!("no subcommand provided"),
    }
}

fn inspect_command(input: &Path) -> Result<()> {
    let bytes = read_bytes(input)?;
    let snapshot = decode_storage_snapshot(bytes.as_slice())?;

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
    let snapshot = decode_storage_snapshot(bytes.as_slice())?;
    write_json(output, &snapshot)
}

fn import_command(input: &Path, output: &Path) -> Result<()> {
    ensure_distinct_paths(input, output)?;
    let snapshot = read_json(input)?;
    let payload = encode_storage_snapshot(&snapshot, snapshot.source.compression)?;
    write_bytes(output, payload.as_slice())
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
