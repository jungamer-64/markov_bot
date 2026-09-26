use std::{
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, anyhow};
use clap::{Arg, ArgMatches, Command, value_parser};
use markov_storage::{
    Codec, FileStore, StorageCompressionMode, StorageLimits, StorageSnapshot,
    ensure_distinct_paths, read_file,
};

#[derive(Debug)]
enum Operation {
    Inspect { input: PathBuf },
    Export { input: PathBuf, output: PathBuf },
    Import { input: PathBuf, output: PathBuf },
}

fn arguments() -> Result<(Operation, StorageLimits)> {
    let input = Arg::new("input")
        .long("input")
        .required(true)
        .value_parser(value_parser!(PathBuf));
    let output = Arg::new("output")
        .long("output")
        .required(true)
        .value_parser(value_parser!(PathBuf));
    let matches = Command::new("markov-storage")
        .about("Inspect and edit Markov storage files")
        .subcommand_required(true)
        .arg(
            Arg::new("max-file-bytes")
                .long("max-file-bytes")
                .global(true)
                .value_parser(value_parser!(u64)),
        )
        .arg(
            Arg::new("max-vocab-bytes")
                .long("max-vocab-bytes")
                .global(true)
                .value_parser(value_parser!(u64)),
        )
        .subcommand(Command::new("inspect").arg(input.clone()))
        .subcommand(
            Command::new("export")
                .arg(input.clone())
                .arg(output.clone()),
        )
        .subcommand(Command::new("import").arg(input).arg(output))
        .get_matches();
    let defaults = StorageLimits::default();
    let limits = StorageLimits::new(
        matches
            .get_one::<u64>("max-file-bytes")
            .copied()
            .unwrap_or_else(|| defaults.file_bytes()),
        matches
            .get_one::<u64>("max-vocab-bytes")
            .copied()
            .unwrap_or_else(|| defaults.vocab_bytes()),
    )?;
    let (name, command) = matches
        .subcommand()
        .ok_or_else(|| anyhow!("missing command"))?;
    let input = argument_path(command, "input")?;
    let operation = match name {
        "inspect" => Operation::Inspect { input },
        "export" => Operation::Export {
            input,
            output: argument_path(command, "output")?,
        },
        "import" => Operation::Import {
            input,
            output: argument_path(command, "output")?,
        },
        _ => return Err(anyhow!("unknown command: {name}")),
    };
    Ok((operation, limits))
}

fn argument_path(arguments: &ArgMatches, name: &str) -> Result<PathBuf> {
    arguments
        .get_one::<PathBuf>(name)
        .cloned()
        .ok_or_else(|| anyhow!("missing {name}"))
}

fn main() -> Result<()> {
    let (operation, limits) = arguments()?;
    let codec = Codec::new(limits);
    match operation {
        Operation::Inspect { input } => {
            let snapshot = codec.decode_snapshot(&read_file(&input, limits)?)?;
            println!("version={}", snapshot.source.storage_version);
            println!(
                "compression={}",
                StorageCompressionMode::from(snapshot.source.compression).as_env_value()
            );
            println!("ngram_order={}", snapshot.source.ngram_order);
            println!("token_count={}", snapshot.tokens.len());
            println!("start_count={}", snapshot.starts.len());
            for model in &snapshot.models {
                let edges: usize = model.entries.iter().map(|entry| entry.edges.len()).sum();
                println!(
                    "model[order={}]: entries={}, edges={edges}",
                    model.order,
                    model.entries.len()
                );
            }
        }
        Operation::Export { input, output } => {
            ensure_distinct_paths(&input, &output)?;
            let snapshot = codec.decode_snapshot(&read_file(&input, limits)?)?;
            let mut bytes = BoundedJson {
                bytes: Vec::new(),
                limit: limits.file_bytes(),
            };
            serde_json::to_writer_pretty(&mut bytes, &snapshot)
                .context("cannot prepare JSON export")?;
            bytes.write_all(b"\n")?;
            FileStore::open(&output, limits)?.publish(&bytes.bytes)?;
        }
        Operation::Import { input, output } => {
            ensure_distinct_paths(&input, &output)?;
            let snapshot: StorageSnapshot = serde_json::from_slice(&read_file(&input, limits)?)
                .context("invalid snapshot JSON")?;
            let compression = snapshot.source.compression.into();
            let bytes = codec.encode_snapshot(snapshot, compression)?;
            FileStore::open(&output, limits)?.publish(&bytes)?;
        }
    }
    Ok(())
}

// Serialization is completed and bounded before a writer lease is acquired.
struct BoundedJson {
    bytes: Vec<u8>,
    limit: u64,
}
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let size = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| io::Error::other("JSON output size overflow"))?;
        if size > self.limit {
            return Err(io::Error::other("JSON file byte limit exceeded"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
