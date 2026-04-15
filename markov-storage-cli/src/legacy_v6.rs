use std::{ops::Range, str};

use anyhow::{Context, Result, anyhow, bail};
use lz4_flex::block::decompress_into as lz4_decompress_into;
use markov_storage::{
    SnapshotEdge, SnapshotEntry, SnapshotModel, SnapshotModelEntry, SnapshotSource,
    StorageCompressionMode, StorageSnapshot,
};

const MAGIC: [u8; 8] = *b"MKV3BIN\0";
const VERSION: u32 = 6;
const HEADER_SIZE: usize = 44;
const DESCRIPTOR_SIZE: usize = 24;
const SECTION_COUNT: usize = 20;
const SECTION_COUNT_U32: u32 = 20;
const CHECKSUM_OFFSET: usize = HEADER_SIZE - std::mem::size_of::<u64>();
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;
const FLAG_VOCAB_BLOB_RLE: u32 = 1 << 0;
const FLAG_VOCAB_BLOB_ZSTD: u32 = 1 << 1;
const FLAG_VOCAB_BLOB_LZ4_FLEX: u32 = 1 << 2;
const SUPPORTED_FLAGS: u32 = FLAG_VOCAB_BLOB_RLE | FLAG_VOCAB_BLOB_ZSTD | FLAG_VOCAB_BLOB_LZ4_FLEX;
const TOKENIZER_VERSION: u32 = 1;
const NORMALIZATION_FLAGS: u32 = 0;

const REPEAT_BASE: u8 = 128;
const REPEAT_CHUNK_MIN: usize = 3;
const REPEAT_CHUNK_MAX: usize = 130;
const MAX_RLE_EXPANSION_PER_ENCODED_BYTE: usize = REPEAT_CHUNK_MAX / 2;

#[derive(Debug, Clone, Copy)]
struct Header {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    tokenizer_version: u32,
    normalization_flags: u32,
    section_count: u32,
    file_size: u64,
    checksum: u64,
}

#[derive(Debug, Clone, Copy)]
struct SectionDescriptor {
    kind: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum SectionKind {
    VocabOffsets = 1,
    VocabBlob = 2,
    Starts = 3,
    Model6Pairs = 4,
    Model6Prefixes = 5,
    Model6Edges = 6,
    Model5Pairs = 7,
    Model5Prefixes = 8,
    Model5Edges = 9,
    Model4Pairs = 10,
    Model4Prefixes = 11,
    Model4Edges = 12,
    Model3Pairs = 13,
    Model3Prefixes = 14,
    Model3Edges = 15,
    Model2Pairs = 16,
    Model2Prefixes = 17,
    Model2Edges = 18,
    Model1Prefixes = 19,
    Model1Edges = 20,
}

impl SectionKind {
    const ALL: [Self; SECTION_COUNT] = [
        Self::VocabOffsets,
        Self::VocabBlob,
        Self::Starts,
        Self::Model6Pairs,
        Self::Model6Prefixes,
        Self::Model6Edges,
        Self::Model5Pairs,
        Self::Model5Prefixes,
        Self::Model5Edges,
        Self::Model4Pairs,
        Self::Model4Prefixes,
        Self::Model4Edges,
        Self::Model3Pairs,
        Self::Model3Prefixes,
        Self::Model3Edges,
        Self::Model2Pairs,
        Self::Model2Prefixes,
        Self::Model2Edges,
        Self::Model1Prefixes,
        Self::Model1Edges,
    ];

    const fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::VocabOffsets),
            2 => Some(Self::VocabBlob),
            3 => Some(Self::Starts),
            4 => Some(Self::Model6Pairs),
            5 => Some(Self::Model6Prefixes),
            6 => Some(Self::Model6Edges),
            7 => Some(Self::Model5Pairs),
            8 => Some(Self::Model5Prefixes),
            9 => Some(Self::Model5Edges),
            10 => Some(Self::Model4Pairs),
            11 => Some(Self::Model4Prefixes),
            12 => Some(Self::Model4Edges),
            13 => Some(Self::Model3Pairs),
            14 => Some(Self::Model3Prefixes),
            15 => Some(Self::Model3Edges),
            16 => Some(Self::Model2Pairs),
            17 => Some(Self::Model2Prefixes),
            18 => Some(Self::Model2Edges),
            19 => Some(Self::Model1Prefixes),
            20 => Some(Self::Model1Edges),
            _ => None,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::VocabOffsets => "vocab offsets",
            Self::VocabBlob => "vocab blob",
            Self::Starts => "start records",
            Self::Model6Pairs => "model6 pairs",
            Self::Model6Prefixes => "model6 prefixes",
            Self::Model6Edges => "model6 edges",
            Self::Model5Pairs => "model5 pairs",
            Self::Model5Prefixes => "model5 prefixes",
            Self::Model5Edges => "model5 edges",
            Self::Model4Pairs => "model4 pairs",
            Self::Model4Prefixes => "model4 prefixes",
            Self::Model4Edges => "model4 edges",
            Self::Model3Pairs => "model3 pairs",
            Self::Model3Prefixes => "model3 prefixes",
            Self::Model3Edges => "model3 edges",
            Self::Model2Pairs => "model2 pairs",
            Self::Model2Prefixes => "model2 prefixes",
            Self::Model2Edges => "model2 edges",
            Self::Model1Prefixes => "model1 prefixes",
            Self::Model1Edges => "model1 edges",
        }
    }

    const fn to_index(self) -> usize {
        match self {
            Self::VocabOffsets => 0,
            Self::VocabBlob => 1,
            Self::Starts => 2,
            Self::Model6Pairs => 3,
            Self::Model6Prefixes => 4,
            Self::Model6Edges => 5,
            Self::Model5Pairs => 6,
            Self::Model5Prefixes => 7,
            Self::Model5Edges => 8,
            Self::Model4Pairs => 9,
            Self::Model4Prefixes => 10,
            Self::Model4Edges => 11,
            Self::Model3Pairs => 12,
            Self::Model3Prefixes => 13,
            Self::Model3Edges => 14,
            Self::Model2Pairs => 15,
            Self::Model2Prefixes => 16,
            Self::Model2Edges => 17,
            Self::Model1Prefixes => 18,
            Self::Model1Edges => 19,
        }
    }
}

#[derive(Debug, Clone)]
struct SectionEntry {
    descriptor: SectionDescriptor,
    range: Range<usize>,
}

#[derive(Debug, Clone)]
struct SectionTable {
    entries: Vec<SectionEntry>,
}

impl SectionTable {
    fn entry(&self, kind: SectionKind) -> Result<&SectionEntry> {
        self.entries
            .get(kind.to_index())
            .ok_or_else(|| anyhow!("section table is missing {}", kind.label()))
    }
}

#[derive(Debug, Clone, Copy)]
struct StartRecord {
    prefix_id: u32,
    cumulative: u64,
}

#[derive(Debug, Clone, Copy)]
struct Pair6Record {
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    w5: u32,
    prefix_start: u32,
    prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
struct Pair5Record {
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    prefix_start: u32,
    prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
struct Pair4Record {
    w1: u32,
    w2: u32,
    w3: u32,
    prefix_start: u32,
    prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
struct Pair3Record {
    w1: u32,
    w2: u32,
    prefix_start: u32,
    prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
struct Pair2Record {
    w1: u32,
    prefix_start: u32,
    prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
struct Prefix6Record {
    w6: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

#[derive(Debug, Clone, Copy)]
struct Prefix5Record {
    w5: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

#[derive(Debug, Clone, Copy)]
struct Prefix4Record {
    w4: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

#[derive(Debug, Clone, Copy)]
struct Prefix3Record {
    w3: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

#[derive(Debug, Clone, Copy)]
struct Prefix2Record {
    w1: u32,
    w2: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

#[derive(Debug, Clone, Copy)]
struct Prefix1Record {
    w1: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

#[derive(Debug, Clone, Copy)]
struct EdgeRecord {
    next: u32,
    cumulative: u64,
}

#[derive(Debug, Clone)]
struct VocabSections {
    offsets: Vec<u64>,
    blob: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Model6Sections {
    pairs: Vec<Pair6Record>,
    prefixes: Vec<Prefix6Record>,
    edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
struct Model5Sections {
    pairs: Vec<Pair5Record>,
    prefixes: Vec<Prefix5Record>,
    edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
struct Model4Sections {
    pairs: Vec<Pair4Record>,
    prefixes: Vec<Prefix4Record>,
    edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
struct Model3Sections {
    pairs: Vec<Pair3Record>,
    prefixes: Vec<Prefix3Record>,
    edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
struct Model2Sections {
    pairs: Vec<Pair2Record>,
    prefixes: Vec<Prefix2Record>,
    edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
struct Model1Sections {
    prefixes: Vec<Prefix1Record>,
    edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
struct StorageSections {
    vocab: VocabSections,
    starts: Vec<StartRecord>,
    model6: Model6Sections,
    model5: Model5Sections,
    model4: Model4Sections,
    model3: Model3Sections,
    model2: Model2Sections,
    model1: Model1Sections,
}

trait FixedRecord: Sized {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self>;
}

impl FixedRecord for StartRecord {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            prefix_id: read_u32_value(bytes, cursor)?,
            cumulative: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair6Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            w2: read_u32_value(bytes, cursor)?,
            w3: read_u32_value(bytes, cursor)?,
            w4: read_u32_value(bytes, cursor)?,
            w5: read_u32_value(bytes, cursor)?,
            prefix_start: read_u32_value(bytes, cursor)?,
            prefix_len: read_u32_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair5Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            w2: read_u32_value(bytes, cursor)?,
            w3: read_u32_value(bytes, cursor)?,
            w4: read_u32_value(bytes, cursor)?,
            prefix_start: read_u32_value(bytes, cursor)?,
            prefix_len: read_u32_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair4Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            w2: read_u32_value(bytes, cursor)?,
            w3: read_u32_value(bytes, cursor)?,
            prefix_start: read_u32_value(bytes, cursor)?,
            prefix_len: read_u32_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair3Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            w2: read_u32_value(bytes, cursor)?,
            prefix_start: read_u32_value(bytes, cursor)?,
            prefix_len: read_u32_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair2Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            prefix_start: read_u32_value(bytes, cursor)?,
            prefix_len: read_u32_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix6Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w6: read_u32_value(bytes, cursor)?,
            edge_start: read_u32_value(bytes, cursor)?,
            edge_len: read_u32_value(bytes, cursor)?,
            total: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix5Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w5: read_u32_value(bytes, cursor)?,
            edge_start: read_u32_value(bytes, cursor)?,
            edge_len: read_u32_value(bytes, cursor)?,
            total: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix4Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w4: read_u32_value(bytes, cursor)?,
            edge_start: read_u32_value(bytes, cursor)?,
            edge_len: read_u32_value(bytes, cursor)?,
            total: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix3Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w3: read_u32_value(bytes, cursor)?,
            edge_start: read_u32_value(bytes, cursor)?,
            edge_len: read_u32_value(bytes, cursor)?,
            total: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix2Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            w2: read_u32_value(bytes, cursor)?,
            edge_start: read_u32_value(bytes, cursor)?,
            edge_len: read_u32_value(bytes, cursor)?,
            total: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix1Record {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            w1: read_u32_value(bytes, cursor)?,
            edge_start: read_u32_value(bytes, cursor)?,
            edge_len: read_u32_value(bytes, cursor)?,
            total: read_u64_value(bytes, cursor)?,
        })
    }
}

impl FixedRecord for EdgeRecord {
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        Ok(Self {
            next: read_u32_value(bytes, cursor)?,
            cumulative: read_u64_value(bytes, cursor)?,
        })
    }
}

#[allow(clippy::redundant_pub_crate)]
pub(super) fn decode_snapshot(bytes: &[u8]) -> Result<StorageSnapshot> {
    let header = validate_header(bytes)?;
    let table = build_section_table(bytes, &header)?;
    let sections = parse_storage(bytes, &header, &table)?;

    let tokens = decode_vocab(
        sections.vocab.offsets.as_slice(),
        sections.vocab.blob.as_slice(),
    )?;
    validate_special_tokens(tokens.as_slice())?;
    let token_count =
        u32::try_from(tokens.len()).map_err(|_error| anyhow!("token count exceeds u32 range"))?;

    let model6_keys = build_model6_keys(&sections.model6, token_count)?;
    let model5_keys = build_model5_keys(&sections.model5, token_count)?;
    let model4_keys = build_model4_keys(&sections.model4, token_count)?;
    let model3_keys = build_model3_keys(&sections.model3, token_count)?;
    validate_model2(&sections.model2, token_count)?;
    validate_model1(&sections.model1, token_count)?;

    let starts = decode_starts(sections.starts.as_slice(), model6_keys.as_slice())?;
    let models = vec![
        SnapshotModel {
            order: 6,
            entries: decode_model_entries(
                model6_keys.as_slice(),
                sections.model6.prefixes.as_slice(),
                sections.model6.edges.as_slice(),
            )?,
        },
        SnapshotModel {
            order: 5,
            entries: decode_model_entries(
                model5_keys.as_slice(),
                sections.model5.prefixes.as_slice(),
                sections.model5.edges.as_slice(),
            )?,
        },
        SnapshotModel {
            order: 4,
            entries: decode_model_entries(
                model4_keys.as_slice(),
                sections.model4.prefixes.as_slice(),
                sections.model4.edges.as_slice(),
            )?,
        },
        SnapshotModel {
            order: 3,
            entries: decode_model_entries(
                model3_keys.as_slice(),
                sections.model3.prefixes.as_slice(),
                sections.model3.edges.as_slice(),
            )?,
        },
        SnapshotModel {
            order: 2,
            entries: decode_model2_entries(
                sections.model2.prefixes.as_slice(),
                sections.model2.edges.as_slice(),
            )?,
        },
        SnapshotModel {
            order: 1,
            entries: decode_model1_entries(
                sections.model1.prefixes.as_slice(),
                sections.model1.edges.as_slice(),
            )?,
        },
    ];

    Ok(StorageSnapshot {
        schema_version: 1,
        source: SnapshotSource {
            storage_version: VERSION,
            ngram_order: 6,
            compression: compression_mode_from_flags(header.flags)?,
        },
        tokens,
        starts,
        models,
    })
}

fn validate_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < HEADER_SIZE {
        bail!("storage file is shorter than the header");
    }

    let mut cursor = 0_usize;
    let magic = {
        let raw = read_exact(bytes, &mut cursor, 8)?;
        let mut value = [0_u8; 8];
        value.copy_from_slice(raw);
        value
    };

    let header = Header {
        magic,
        version: read_u32_value(bytes, &mut cursor)?,
        flags: read_u32_value(bytes, &mut cursor)?,
        tokenizer_version: read_u32_value(bytes, &mut cursor)?,
        normalization_flags: read_u32_value(bytes, &mut cursor)?,
        section_count: read_u32_value(bytes, &mut cursor)?,
        file_size: read_u64_value(bytes, &mut cursor)?,
        checksum: read_u64_value(bytes, &mut cursor)?,
    };

    if header.magic != MAGIC {
        bail!("storage magic mismatch");
    }
    if header.version != VERSION {
        bail!("unsupported storage version: {}", header.version);
    }
    if header.flags & !SUPPORTED_FLAGS != 0 {
        bail!("unsupported storage flags: {}", header.flags);
    }
    let _ = vocab_blob_compression_flags(header.flags)?;
    if header.tokenizer_version != TOKENIZER_VERSION {
        bail!(
            "unsupported tokenizer version: {}",
            header.tokenizer_version
        );
    }
    if header.normalization_flags != NORMALIZATION_FLAGS {
        bail!(
            "unsupported normalization flags: {}",
            header.normalization_flags
        );
    }
    if header.section_count != SECTION_COUNT_U32 {
        bail!("unsupported section count: {}", header.section_count);
    }

    let actual_file_size =
        u64::try_from(bytes.len()).map_err(|_error| anyhow!("file size exceeds u64 range"))?;
    if header.file_size != actual_file_size {
        bail!(
            "file size mismatch: header={}, actual={actual_file_size}",
            header.file_size
        );
    }

    let expected_checksum = compute_checksum(bytes);
    if header.checksum != expected_checksum {
        bail!(
            "checksum mismatch: header={}, expected={expected_checksum}",
            header.checksum
        );
    }

    Ok(header)
}

fn build_section_table(bytes: &[u8], header: &Header) -> Result<SectionTable> {
    let mut cursor = HEADER_SIZE;
    let mut descriptors = Vec::with_capacity(SECTION_COUNT);
    for _ in 0..SECTION_COUNT {
        descriptors.push(SectionDescriptor {
            kind: read_u32_value(bytes, &mut cursor)?,
            flags: read_u32_value(bytes, &mut cursor)?,
            offset: read_u64_value(bytes, &mut cursor)?,
            size: read_u64_value(bytes, &mut cursor)?,
        });
    }

    let metadata_end = aligned_metadata_end()?;
    let _ = bytes
        .get(..metadata_end)
        .ok_or_else(|| anyhow!("metadata extends beyond file size"))?;

    let mut entries = Vec::with_capacity(SECTION_COUNT);
    let mut expected_start = metadata_end;

    for (index, descriptor) in descriptors.into_iter().enumerate() {
        let kind = SectionKind::from_u32(descriptor.kind)
            .ok_or_else(|| anyhow!("unknown section kind: {}", descriptor.kind))?;
        let expected_kind = SectionKind::ALL
            .get(index)
            .copied()
            .ok_or_else(|| anyhow!("descriptor index out of bounds"))?;
        if kind != expected_kind {
            bail!(
                "section descriptors are not in canonical order: expected {}, got {}",
                expected_kind.label(),
                kind.label()
            );
        }
        if descriptor.flags != 0 {
            bail!(
                "unsupported {} descriptor flags: {}",
                kind.label(),
                descriptor.flags
            );
        }
        if descriptor.offset % 8 != 0 {
            bail!("{} section offset must be 8-byte aligned", kind.label());
        }

        let range = section_range(
            descriptor.offset,
            descriptor.size,
            header.file_size,
            kind.label(),
        )?;
        if range.start < metadata_end {
            bail!(
                "{} section starts before aligned metadata end",
                kind.label()
            );
        }
        if range.start < expected_start {
            bail!("{} section overlaps previous section", kind.label());
        }
        validate_zero_padding(bytes, expected_start, range.start, kind.label())?;
        expected_start = range.end;
        entries.push(SectionEntry { descriptor, range });
    }

    let file_size = usize::try_from(header.file_size)
        .map_err(|_error| anyhow!("file size exceeds usize range"))?;
    if expected_start != file_size {
        bail!("final section must end at file size");
    }

    Ok(SectionTable { entries })
}

// ALLOW: レガシーフォーマット解析コードのため分割は避ける
#[allow(clippy::too_many_lines)]
fn parse_storage(bytes: &[u8], header: &Header, table: &SectionTable) -> Result<StorageSections> {
    let vocab_offsets = parse_u64_section(
        section_bytes(bytes, table.entry(SectionKind::VocabOffsets)?)?,
        SectionKind::VocabOffsets.label(),
    )?;
    validate_vocab_offsets(vocab_offsets.as_slice())?;
    let vocab_blob_size = *vocab_offsets
        .last()
        .ok_or_else(|| anyhow!("vocab offsets are empty"))?;
    let vocab_blob = decode_vocab_blob(
        section_bytes(bytes, table.entry(SectionKind::VocabBlob)?)?,
        usize::try_from(vocab_blob_size)
            .map_err(|_error| anyhow!("vocab blob size exceeds usize range"))?,
        header.flags,
    )?;

    Ok(StorageSections {
        vocab: VocabSections {
            offsets: vocab_offsets,
            blob: vocab_blob,
        },
        starts: parse_fixed_section(
            section_bytes(bytes, table.entry(SectionKind::Starts)?)?,
            12,
            SectionKind::Starts.label(),
        )?,
        model6: Model6Sections {
            pairs: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model6Pairs)?)?,
                28,
                SectionKind::Model6Pairs.label(),
            )?,
            prefixes: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model6Prefixes)?)?,
                20,
                SectionKind::Model6Prefixes.label(),
            )?,
            edges: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model6Edges)?)?,
                12,
                SectionKind::Model6Edges.label(),
            )?,
        },
        model5: Model5Sections {
            pairs: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model5Pairs)?)?,
                24,
                SectionKind::Model5Pairs.label(),
            )?,
            prefixes: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model5Prefixes)?)?,
                20,
                SectionKind::Model5Prefixes.label(),
            )?,
            edges: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model5Edges)?)?,
                12,
                SectionKind::Model5Edges.label(),
            )?,
        },
        model4: Model4Sections {
            pairs: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model4Pairs)?)?,
                20,
                SectionKind::Model4Pairs.label(),
            )?,
            prefixes: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model4Prefixes)?)?,
                20,
                SectionKind::Model4Prefixes.label(),
            )?,
            edges: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model4Edges)?)?,
                12,
                SectionKind::Model4Edges.label(),
            )?,
        },
        model3: Model3Sections {
            pairs: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model3Pairs)?)?,
                16,
                SectionKind::Model3Pairs.label(),
            )?,
            prefixes: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model3Prefixes)?)?,
                20,
                SectionKind::Model3Prefixes.label(),
            )?,
            edges: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model3Edges)?)?,
                12,
                SectionKind::Model3Edges.label(),
            )?,
        },
        model2: Model2Sections {
            pairs: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model2Pairs)?)?,
                12,
                SectionKind::Model2Pairs.label(),
            )?,
            prefixes: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model2Prefixes)?)?,
                24,
                SectionKind::Model2Prefixes.label(),
            )?,
            edges: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model2Edges)?)?,
                12,
                SectionKind::Model2Edges.label(),
            )?,
        },
        model1: Model1Sections {
            prefixes: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model1Prefixes)?)?,
                20,
                SectionKind::Model1Prefixes.label(),
            )?,
            edges: parse_fixed_section(
                section_bytes(bytes, table.entry(SectionKind::Model1Edges)?)?,
                12,
                SectionKind::Model1Edges.label(),
            )?,
        },
    })
}

fn section_bytes<'a>(bytes: &'a [u8], entry: &SectionEntry) -> Result<&'a [u8]> {
    if let Some(slice) = bytes.get(entry.range.clone()) {
        return Ok(slice);
    }

    let kind = SectionKind::from_u32(entry.descriptor.kind)
        .ok_or_else(|| anyhow!("unknown section kind"))?;
    Err(anyhow!("{} range is out of bounds", kind.label()))
}

fn parse_u64_section(bytes: &[u8], context: &str) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        bail!("{context} section size is not a multiple of 8");
    }

    let mut values = Vec::with_capacity(bytes.len() / 8);
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        values.push(read_u64_value(bytes, &mut cursor)?);
    }

    Ok(values)
}

fn parse_fixed_section<T: FixedRecord>(
    bytes: &[u8],
    record_size: usize,
    context: &str,
) -> Result<Vec<T>> {
    if !bytes.len().is_multiple_of(record_size) {
        bail!("{context} section size is not a multiple of record size");
    }

    let mut records = Vec::with_capacity(bytes.len() / record_size);
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        records.push(T::decode_from(bytes, &mut cursor)?);
    }

    Ok(records)
}

fn decode_vocab(offsets: &[u64], blob: &[u8]) -> Result<Vec<String>> {
    if offsets.is_empty() {
        bail!("vocab offsets are empty");
    }

    let mut tokens = Vec::with_capacity(offsets.len().saturating_sub(1));
    for pair in offsets.windows(2) {
        let [start_offset, end_offset] =
            <&[u64; 2]>::try_from(pair).map_err(|_error| anyhow!("invalid vocab offset pair"))?;
        let start = usize::try_from(*start_offset)
            .map_err(|_error| anyhow!("vocab token start exceeds usize range"))?;
        let end = usize::try_from(*end_offset)
            .map_err(|_error| anyhow!("vocab token end exceeds usize range"))?;
        let token_bytes = blob
            .get(start..end)
            .ok_or_else(|| anyhow!("vocab token range is invalid"))?;
        let token = str::from_utf8(token_bytes)
            .map_err(|_error| anyhow!("vocab token is not valid UTF-8"))?
            .to_owned();
        tokens.push(token);
    }

    Ok(tokens)
}

fn validate_vocab_offsets(offsets: &[u64]) -> Result<()> {
    if offsets.first().copied() != Some(0) {
        bail!("vocab offsets must start with 0");
    }

    for pair in offsets.windows(2) {
        let [start_offset, end_offset] =
            <&[u64; 2]>::try_from(pair).map_err(|_error| anyhow!("invalid vocab offset pair"))?;
        if start_offset > end_offset {
            bail!("vocab offsets must be non-decreasing");
        }
    }

    Ok(())
}

fn validate_special_tokens(tokens: &[String]) -> Result<()> {
    let Some(first) = tokens.first() else {
        bail!("vocabulary is empty");
    };
    if first != "<BOS>" {
        bail!("token id 0 must be <BOS>");
    }

    let Some(second) = tokens.get(1) else {
        bail!("vocabulary is missing <EOS>");
    };
    if second != "<EOS>" {
        bail!("token id 1 must be <EOS>");
    }

    Ok(())
}

fn build_model6_keys(model: &Model6Sections, token_count: u32) -> Result<Vec<Vec<u32>>> {
    let mut full_prefixes = vec![Vec::new(); model.prefixes.len()];
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_pair = None;

    for pair in &model.pairs {
        validate_token_id(pair.w1, token_count, "model6 pair.w1")?;
        validate_token_id(pair.w2, token_count, "model6 pair.w2")?;
        validate_token_id(pair.w3, token_count, "model6 pair.w3")?;
        validate_token_id(pair.w4, token_count, "model6 pair.w4")?;
        validate_token_id(pair.w5, token_count, "model6 pair.w5")?;

        let current_pair = (pair.w1, pair.w2, pair.w3, pair.w4, pair.w5);
        if let Some(previous) = previous_pair
            && current_pair <= previous
        {
            bail!("model6 pair records are not strictly sorted");
        }
        previous_pair = Some(current_pair);

        let prefix_start = usize::try_from(pair.prefix_start)
            .map_err(|_error| anyhow!("model6 prefix start exceeds usize range"))?;
        let prefix_len = usize::try_from(pair.prefix_len)
            .map_err(|_error| anyhow!("model6 prefix len exceeds usize range"))?;
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or_else(|| anyhow!("model6 prefix range overflow"))?;
        if prefix_end > model.prefixes.len() {
            bail!("model6 pair prefix range is out of bounds");
        }

        let mut previous_w6 = None;
        let prefix_slice = model
            .prefixes
            .get(prefix_start..prefix_end)
            .context("model6 prefixes range")?;
        for (relative_index, prefix) in prefix_slice.iter().enumerate()
        {
            let index = prefix_start + relative_index;
            let is_assigned = assigned
                .get_mut(index)
                .ok_or_else(|| anyhow!("model6 pair prefix range is out of bounds"))?;
            if *is_assigned {
                bail!("model6 pair prefix ranges overlap");
            }
            validate_token_id(prefix.w6, token_count, "model6 prefix.w6")?;
            if let Some(previous) = previous_w6
                && prefix.w6 <= previous
            {
                bail!("model6 prefix records are not sorted by w6");
            }
            previous_w6 = Some(prefix.w6);

            validate_prefix_edges(
                model.edges.as_slice(),
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model6 prefix",
            )?;

            *full_prefixes
                .get_mut(index)
                .context("model6 prefix assignment")? =
                vec![pair.w1, pair.w2, pair.w3, pair.w4, pair.w5, prefix.w6];
            *is_assigned = true;
        }
    }

    if assigned.iter().any(|assigned| !assigned) {
        bail!("some model6 prefixes are not covered by pair records");
    }

    Ok(full_prefixes)
}

fn build_model5_keys(model: &Model5Sections, token_count: u32) -> Result<Vec<Vec<u32>>> {
    let mut full_prefixes = vec![Vec::new(); model.prefixes.len()];
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_pair = None;

    for pair in &model.pairs {
        validate_token_id(pair.w1, token_count, "model5 pair.w1")?;
        validate_token_id(pair.w2, token_count, "model5 pair.w2")?;
        validate_token_id(pair.w3, token_count, "model5 pair.w3")?;
        validate_token_id(pair.w4, token_count, "model5 pair.w4")?;

        let current_pair = (pair.w1, pair.w2, pair.w3, pair.w4);
        if let Some(previous) = previous_pair
            && current_pair <= previous
        {
            bail!("model5 pair records are not strictly sorted");
        }
        previous_pair = Some(current_pair);

        let prefix_start = usize::try_from(pair.prefix_start)
            .map_err(|_error| anyhow!("model5 prefix start exceeds usize range"))?;
        let prefix_len = usize::try_from(pair.prefix_len)
            .map_err(|_error| anyhow!("model5 prefix len exceeds usize range"))?;
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or_else(|| anyhow!("model5 prefix range overflow"))?;
        if prefix_end > model.prefixes.len() {
            bail!("model5 pair prefix range is out of bounds");
        }

        let mut previous_w5 = None;
        let prefix_slice = model
            .prefixes
            .get(prefix_start..prefix_end)
            .context("model5 prefixes range")?;
        for (relative_index, prefix) in prefix_slice.iter().enumerate()
        {
            let index = prefix_start + relative_index;
            let is_assigned = assigned
                .get_mut(index)
                .ok_or_else(|| anyhow!("model5 pair prefix range is out of bounds"))?;
            if *is_assigned {
                bail!("model5 pair prefix ranges overlap");
            }
            validate_token_id(prefix.w5, token_count, "model5 prefix.w5")?;
            if let Some(previous) = previous_w5
                && prefix.w5 <= previous
            {
                bail!("model5 prefix records are not sorted by w5");
            }
            previous_w5 = Some(prefix.w5);

            validate_prefix_edges(
                model.edges.as_slice(),
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model5 prefix",
            )?;

            *full_prefixes
                .get_mut(index)
                .context("model5 prefix assignment")? =
                vec![pair.w1, pair.w2, pair.w3, pair.w4, prefix.w5];
            *is_assigned = true;
        }
    }

    if assigned.iter().any(|assigned| !assigned) {
        bail!("some model5 prefixes are not covered by pair records");
    }

    Ok(full_prefixes)
}

fn build_model4_keys(model: &Model4Sections, token_count: u32) -> Result<Vec<Vec<u32>>> {
    let mut full_prefixes = vec![Vec::new(); model.prefixes.len()];
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_pair = None;

    for pair in &model.pairs {
        validate_token_id(pair.w1, token_count, "model4 pair.w1")?;
        validate_token_id(pair.w2, token_count, "model4 pair.w2")?;
        validate_token_id(pair.w3, token_count, "model4 pair.w3")?;

        let current_pair = (pair.w1, pair.w2, pair.w3);
        if let Some(previous) = previous_pair
            && current_pair <= previous
        {
            bail!("model4 pair records are not strictly sorted");
        }
        previous_pair = Some(current_pair);

        let prefix_start = usize::try_from(pair.prefix_start)
            .map_err(|_error| anyhow!("model4 prefix start exceeds usize range"))?;
        let prefix_len = usize::try_from(pair.prefix_len)
            .map_err(|_error| anyhow!("model4 prefix len exceeds usize range"))?;
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or_else(|| anyhow!("model4 prefix range overflow"))?;
        if prefix_end > model.prefixes.len() {
            bail!("model4 pair prefix range is out of bounds");
        }

        let mut previous_w4 = None;
        let prefix_slice = model
            .prefixes
            .get(prefix_start..prefix_end)
            .context("model4 prefixes range")?;
        for (relative_index, prefix) in prefix_slice.iter().enumerate()
        {
            let index = prefix_start + relative_index;
            let is_assigned = assigned
                .get_mut(index)
                .ok_or_else(|| anyhow!("model4 pair prefix range is out of bounds"))?;
            if *is_assigned {
                bail!("model4 pair prefix ranges overlap");
            }
            validate_token_id(prefix.w4, token_count, "model4 prefix.w4")?;
            if let Some(previous) = previous_w4
                && prefix.w4 <= previous
            {
                bail!("model4 prefix records are not sorted by w4");
            }
            previous_w4 = Some(prefix.w4);

            validate_prefix_edges(
                model.edges.as_slice(),
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model4 prefix",
            )?;

            *full_prefixes
                .get_mut(index)
                .context("model4 prefix assignment")? =
                vec![pair.w1, pair.w2, pair.w3, prefix.w4];
            *is_assigned = true;
        }
    }

    if assigned.iter().any(|assigned| !assigned) {
        bail!("some model4 prefixes are not covered by pair records");
    }

    Ok(full_prefixes)
}

fn build_model3_keys(model: &Model3Sections, token_count: u32) -> Result<Vec<Vec<u32>>> {
    let mut full_prefixes = vec![Vec::new(); model.prefixes.len()];
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_pair = None;

    for pair in &model.pairs {
        validate_token_id(pair.w1, token_count, "model3 pair.w1")?;
        validate_token_id(pair.w2, token_count, "model3 pair.w2")?;

        let current_pair = (pair.w1, pair.w2);
        if let Some(previous) = previous_pair
            && current_pair <= previous
        {
            bail!("model3 pair records are not strictly sorted");
        }
        previous_pair = Some(current_pair);

        let prefix_start = usize::try_from(pair.prefix_start)
            .map_err(|_error| anyhow!("model3 prefix start exceeds usize range"))?;
        let prefix_len = usize::try_from(pair.prefix_len)
            .map_err(|_error| anyhow!("model3 prefix len exceeds usize range"))?;
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or_else(|| anyhow!("model3 prefix range overflow"))?;
        if prefix_end > model.prefixes.len() {
            bail!("model3 pair prefix range is out of bounds");
        }

        let mut previous_w3 = None;
        let prefix_slice = model
            .prefixes
            .get(prefix_start..prefix_end)
            .context("model3 prefixes range")?;
        for (relative_index, prefix) in prefix_slice.iter().enumerate()
        {
            let index = prefix_start + relative_index;
            let is_assigned = assigned
                .get_mut(index)
                .ok_or_else(|| anyhow!("model3 pair prefix range is out of bounds"))?;
            if *is_assigned {
                bail!("model3 pair prefix ranges overlap");
            }
            validate_token_id(prefix.w3, token_count, "model3 prefix.w3")?;
            if let Some(previous) = previous_w3
                && prefix.w3 <= previous
            {
                bail!("model3 prefix records are not sorted by w3");
            }
            previous_w3 = Some(prefix.w3);

            validate_prefix_edges(
                model.edges.as_slice(),
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model3 prefix",
            )?;

            *full_prefixes
                .get_mut(index)
                .context("model3 prefix assignment")? = vec![pair.w1, pair.w2, prefix.w3];
            *is_assigned = true;
        }
    }

    if assigned.iter().any(|assigned| !assigned) {
        bail!("some model3 prefixes are not covered by pair records");
    }

    Ok(full_prefixes)
}

fn validate_model2(model: &Model2Sections, token_count: u32) -> Result<()> {
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_w1 = None;

    for pair in &model.pairs {
        validate_token_id(pair.w1, token_count, "model2 pair.w1")?;
        if let Some(previous) = previous_w1
            && pair.w1 <= previous
        {
            bail!("model2 pair records are not strictly sorted");
        }
        previous_w1 = Some(pair.w1);

        let prefix_start = usize::try_from(pair.prefix_start)
            .map_err(|_error| anyhow!("model2 prefix start exceeds usize range"))?;
        let prefix_len = usize::try_from(pair.prefix_len)
            .map_err(|_error| anyhow!("model2 prefix len exceeds usize range"))?;
        if prefix_len == 0 {
            bail!("model2 pair prefix_len must be greater than zero");
        }
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or_else(|| anyhow!("model2 prefix range overflow"))?;
        if prefix_end > model.prefixes.len() {
            bail!("model2 pair prefix range is out of bounds");
        }

        let mut previous_w2 = None;
        let prefix_slice = model
            .prefixes
            .get(prefix_start..prefix_end)
            .context("model2 prefixes range")?;
        for (prefix, is_assigned) in prefix_slice.iter().zip(
            assigned
                .iter_mut()
                .skip(prefix_start)
                .take(prefix_end - prefix_start),
        ) {
            if *is_assigned {
                bail!("model2 pair prefix ranges overlap");
            }
            *is_assigned = true;
            if prefix.w1 != pair.w1 {
                bail!("model2 prefix.w1 does not match model2 pair.w1");
            }
            validate_token_id(prefix.w2, token_count, "model2 prefix.w2")?;
            if let Some(previous) = previous_w2
                && prefix.w2 <= previous
            {
                bail!("model2 prefix records are not sorted by w2 within pair group");
            }
            previous_w2 = Some(prefix.w2);

            validate_prefix_edges(
                model.edges.as_slice(),
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model2 prefix",
            )?;
        }
    }

    if assigned.iter().any(|assigned| !assigned) {
        bail!("some model2 prefixes are not covered by pair records");
    }

    Ok(())
}

fn validate_model1(model: &Model1Sections, token_count: u32) -> Result<()> {
    let mut previous_w1 = None;

    for prefix in &model.prefixes {
        validate_token_id(prefix.w1, token_count, "model1 prefix.w1")?;
        if let Some(previous) = previous_w1
            && prefix.w1 <= previous
        {
            bail!("model1 prefix records are not strictly sorted");
        }
        previous_w1 = Some(prefix.w1);

        validate_prefix_edges(
            model.edges.as_slice(),
            prefix.edge_start,
            prefix.edge_len,
            prefix.total,
            token_count,
            "model1 prefix",
        )?;
    }

    Ok(())
}

fn decode_starts(starts: &[StartRecord], model6_keys: &[Vec<u32>]) -> Result<Vec<SnapshotEntry>> {
    let mut seen = vec![false; model6_keys.len()];
    let mut previous_cumulative = 0_u64;
    let mut decoded = Vec::with_capacity(starts.len());

    for record in starts {
        let prefix_index = usize::try_from(record.prefix_id)
            .map_err(|_error| anyhow!("start prefix_id exceeds usize range"))?;
        let prefix = model6_keys
            .get(prefix_index)
            .ok_or_else(|| anyhow!("start prefix_id is out of range"))?;
        let seen_entry = seen
            .get_mut(prefix_index)
            .ok_or_else(|| anyhow!("start prefix_id is out of range"))?;
        if *seen_entry {
            bail!("duplicate start prefix_id is not allowed");
        }
        *seen_entry = true;
        if record.cumulative <= previous_cumulative {
            bail!("start cumulative must be strictly increasing");
        }

        let count = record
            .cumulative
            .checked_sub(previous_cumulative)
            .ok_or_else(|| anyhow!("start cumulative underflow"))?;
        previous_cumulative = record.cumulative;
        decoded.push(SnapshotEntry {
            prefix: prefix.clone(),
            count,
        });
    }

    Ok(decoded)
}

fn decode_model_entries<T>(
    prefixes: &[Vec<u32>],
    prefix_records: &[T],
    edges: &[EdgeRecord],
) -> Result<Vec<SnapshotModelEntry>>
where
    T: PrefixRecord,
{
    let mut entries = Vec::with_capacity(prefix_records.len());
    for (index, record) in prefix_records.iter().enumerate() {
        let prefix = prefixes
            .get(index)
            .ok_or_else(|| anyhow!("prefix index is out of bounds"))?;
        entries.push(SnapshotModelEntry {
            prefix: prefix.clone(),
            edges: decode_edges(
                edges,
                record.edge_start(),
                record.edge_len(),
                record.total(),
                record.context_label(),
            )?,
        });
    }
    Ok(entries)
}

fn decode_model2_entries(
    prefixes: &[Prefix2Record],
    edges: &[EdgeRecord],
) -> Result<Vec<SnapshotModelEntry>> {
    let mut entries = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        entries.push(SnapshotModelEntry {
            prefix: vec![prefix.w1, prefix.w2],
            edges: decode_edges(
                edges,
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                "model2 prefix",
            )?,
        });
    }
    Ok(entries)
}

fn decode_model1_entries(
    prefixes: &[Prefix1Record],
    edges: &[EdgeRecord],
) -> Result<Vec<SnapshotModelEntry>> {
    let mut entries = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        entries.push(SnapshotModelEntry {
            prefix: vec![prefix.w1],
            edges: decode_edges(
                edges,
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                "model1 prefix",
            )?,
        });
    }
    Ok(entries)
}

fn decode_edges(
    edges: &[EdgeRecord],
    edge_start: u32,
    edge_len: u32,
    total: u64,
    context: &str,
) -> Result<Vec<SnapshotEdge>> {
    let start =
        usize::try_from(edge_start).map_err(|_error| anyhow!("edge_start exceeds usize range"))?;
    let len =
        usize::try_from(edge_len).map_err(|_error| anyhow!("edge_len exceeds usize range"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow!("edge range overflow"))?;
    let edge_slice = edges
        .get(start..end)
        .ok_or_else(|| anyhow!("{context} edge range is out of bounds"))?;

    let mut previous_cumulative = 0_u64;
    let mut decoded = Vec::with_capacity(edge_slice.len());
    for edge in edge_slice {
        if edge.cumulative <= previous_cumulative {
            bail!("{context} cumulative must be strictly increasing");
        }
        let count = edge
            .cumulative
            .checked_sub(previous_cumulative)
            .ok_or_else(|| anyhow!("edge cumulative underflow"))?;
        previous_cumulative = edge.cumulative;
        decoded.push(SnapshotEdge {
            next: edge.next,
            count,
        });
    }

    if previous_cumulative != total {
        bail!("{context} total does not match last cumulative");
    }

    Ok(decoded)
}

fn validate_prefix_edges(
    edges: &[EdgeRecord],
    edge_start: u32,
    edge_len: u32,
    total: u64,
    token_count: u32,
    context: &str,
) -> Result<()> {
    let start =
        usize::try_from(edge_start).map_err(|_error| anyhow!("edge_start exceeds usize range"))?;
    let len =
        usize::try_from(edge_len).map_err(|_error| anyhow!("edge_len exceeds usize range"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow!("edge range overflow"))?;

    if end > edges.len() {
        bail!("{context} edge range is out of bounds");
    }
    if edge_len == 0 {
        if total != 0 {
            bail!("{context} total must be zero when edge_len is zero");
        }
        return Ok(());
    }

    let mut previous_next = None;
    let mut previous_cumulative = 0_u64;
    let edge_slice = edges.get(start..end).context("edges range")?;
    for edge in edge_slice {
        validate_token_id(edge.next, token_count, context)?;
        if let Some(previous) = previous_next
            && edge.next <= previous
        {
            bail!("{context} edges are not sorted by next");
        }
        previous_next = Some(edge.next);
        if edge.cumulative <= previous_cumulative {
            bail!("{context} cumulative must be strictly increasing");
        }
        previous_cumulative = edge.cumulative;
    }

    if previous_cumulative != total {
        bail!("{context} total does not match last cumulative");
    }

    Ok(())
}

fn validate_token_id(token_id: u32, token_count: u32, context: &str) -> Result<()> {
    if token_id >= token_count {
        bail!("{context}: token id {token_id} is out of range");
    }
    Ok(())
}

fn decode_vocab_blob(vocab_blob_bytes: &[u8], expected_size: usize, flags: u32) -> Result<Vec<u8>> {
    match vocab_blob_compression_flags(flags)? {
        0 => decode_vocab_blob_plain(vocab_blob_bytes, expected_size),
        FLAG_VOCAB_BLOB_RLE => decode_vocab_blob_rle(vocab_blob_bytes, expected_size),
        FLAG_VOCAB_BLOB_ZSTD => decode_vocab_blob_zstd(vocab_blob_bytes, expected_size),
        FLAG_VOCAB_BLOB_LZ4_FLEX => decode_vocab_blob_lz4_flex(vocab_blob_bytes, expected_size),
        _ => bail!("unsupported vocab blob compression flags"),
    }
}

fn decode_vocab_blob_plain(vocab_blob_bytes: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let stored_size = u64::try_from(vocab_blob_bytes.len())
        .map_err(|_error| anyhow!("vocab blob size exceeds u64 range"))?;
    let expected = u64::try_from(expected_size)
        .map_err(|_error| anyhow!("vocab blob size exceeds u64 range"))?;
    if expected > stored_size {
        bail!("vocab blob size exceeds stored section");
    }
    Ok(vocab_blob_bytes
        .get(..expected_size)
        .ok_or_else(|| anyhow!("vocab blob range is invalid"))?
        .to_vec())
}

fn decode_vocab_blob_rle(vocab_blob_bytes: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    validate_rle_expected_size(vocab_blob_bytes.len(), expected_size)?;

    let mut decoded = Vec::with_capacity(expected_size);
    let mut cursor = 0_usize;
    while decoded.len() < expected_size {
        let control = *vocab_blob_bytes
            .get(cursor)
            .ok_or_else(|| anyhow!("compressed vocab blob is truncated"))?;
        cursor += 1;
        if control < REPEAT_BASE {
            let literal_len = usize::from(control) + 1;
            let end = cursor
                .checked_add(literal_len)
                .ok_or_else(|| anyhow!("compressed vocab literal range overflow"))?;
            let chunk = vocab_blob_bytes
                .get(cursor..end)
                .ok_or_else(|| anyhow!("compressed vocab literal chunk is truncated"))?;
            append_chunk(
                &mut decoded,
                chunk,
                expected_size,
                "compressed vocab literal",
            )?;
            cursor = end;
        } else {
            let repeat_len = usize::from(control - REPEAT_BASE) + REPEAT_CHUNK_MIN;
            let value = *vocab_blob_bytes
                .get(cursor)
                .ok_or_else(|| anyhow!("compressed vocab repeat chunk is truncated"))?;
            cursor += 1;
            let next_size = decoded
                .len()
                .checked_add(repeat_len)
                .ok_or_else(|| anyhow!("compressed vocab size overflow"))?;
            if next_size > expected_size {
                bail!("compressed vocab repeat exceeds expected decoded size");
            }
            decoded.resize(next_size, value);
        }
    }

    if cursor != vocab_blob_bytes.len() {
        bail!("compressed vocab blob has trailing bytes");
    }

    Ok(decoded)
}

fn decode_vocab_blob_zstd(vocab_blob_bytes: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let decoded = zstd::bulk::decompress(vocab_blob_bytes, expected_size)?;
    if decoded.len() != expected_size {
        bail!("zstd vocab blob size does not match expected decoded size");
    }
    Ok(decoded)
}

fn decode_vocab_blob_lz4_flex(vocab_blob_bytes: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut decoded = vec![0; expected_size];
    let written = lz4_decompress_into(vocab_blob_bytes, decoded.as_mut_slice())?;
    if written != expected_size {
        bail!("lz4_flex vocab blob size does not match expected decoded size");
    }
    Ok(decoded)
}

fn validate_rle_expected_size(encoded_size: usize, expected_size: usize) -> Result<()> {
    let max_decoded_size = encoded_size
        .checked_mul(MAX_RLE_EXPANSION_PER_ENCODED_BYTE)
        .ok_or_else(|| anyhow!("compressed vocab expansion bound overflow"))?;
    if expected_size > max_decoded_size {
        bail!("compressed vocab decoded size exceeds supported expansion bound");
    }
    Ok(())
}

fn append_chunk(
    decoded: &mut Vec<u8>,
    chunk: &[u8],
    expected_size: usize,
    context: &str,
) -> Result<()> {
    let next_size = decoded
        .len()
        .checked_add(chunk.len())
        .ok_or_else(|| anyhow!("compressed vocab size overflow"))?;
    if next_size > expected_size {
        bail!("{context} exceeds expected decoded size");
    }
    decoded.extend_from_slice(chunk);
    Ok(())
}

fn compression_mode_from_flags(flags: u32) -> Result<StorageCompressionMode> {
    match vocab_blob_compression_flags(flags)? {
        0 => Ok(StorageCompressionMode::Uncompressed),
        FLAG_VOCAB_BLOB_RLE => Ok(StorageCompressionMode::Rle),
        FLAG_VOCAB_BLOB_ZSTD => Ok(StorageCompressionMode::Zstd),
        FLAG_VOCAB_BLOB_LZ4_FLEX => Ok(StorageCompressionMode::Lz4Flex),
        _ => bail!("unsupported vocab blob compression flags"),
    }
}

fn vocab_blob_compression_flags(flags: u32) -> Result<u32> {
    let compression_flags = flags & SUPPORTED_FLAGS;
    if compression_flags.count_ones() > 1 {
        bail!("multiple vocab blob compression flags are set");
    }
    Ok(compression_flags)
}

fn compute_checksum(bytes: &[u8]) -> u64 {
    let checksum_range = CHECKSUM_OFFSET..(CHECKSUM_OFFSET + std::mem::size_of::<u64>());
    let mut hash = FNV1A64_OFFSET_BASIS;
    for (index, byte) in bytes.iter().enumerate() {
        let normalized = if checksum_range.contains(&index) {
            0_u8
        } else {
            *byte
        };
        hash ^= u64::from(normalized);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn aligned_metadata_end() -> Result<usize> {
    let descriptor_bytes = SECTION_COUNT
        .checked_mul(DESCRIPTOR_SIZE)
        .ok_or_else(|| anyhow!("descriptor table size overflow"))?;
    Ok((HEADER_SIZE + descriptor_bytes).next_multiple_of(8))
}

fn section_range(offset: u64, size: u64, file_size: u64, context: &str) -> Result<Range<usize>> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| anyhow!("{context} range overflow"))?;
    if end > file_size {
        bail!("{context} exceeds file size");
    }
    Ok(
        usize::try_from(offset).map_err(|_error| anyhow!("{context} offset exceeds usize range"))?
            ..usize::try_from(end)
                .map_err(|_error| anyhow!("{context} end exceeds usize range"))?,
    )
}

fn validate_zero_padding(bytes: &[u8], start: usize, end: usize, context: &str) -> Result<()> {
    if start > end {
        bail!("padding range after {context} is invalid");
    }
    let slice = bytes
        .get(start..end)
        .ok_or_else(|| anyhow!("padding range after {context} is out of bounds"))?;
    if slice.iter().any(|byte| *byte != 0) {
        bail!("non-zero padding detected after {context}");
    }
    Ok(())
}

fn read_exact<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(count)
        .ok_or_else(|| anyhow!("cursor overflow"))?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("unexpected EOF while reading"))?;
    *cursor = end;
    Ok(slice)
}

fn read_u32_value(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let raw = read_exact(bytes, cursor, 4)?;
    let mut array = [0_u8; 4];
    array.copy_from_slice(raw);
    Ok(u32::from_le_bytes(array))
}

fn read_u64_value(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let raw = read_exact(bytes, cursor, 8)?;
    let mut array = [0_u8; 8];
    array.copy_from_slice(raw);
    Ok(u64::from_le_bytes(array))
}

trait PrefixRecord {
    fn edge_start(&self) -> u32;
    fn edge_len(&self) -> u32;
    fn total(&self) -> u64;
    fn context_label(&self) -> &'static str;
}

impl PrefixRecord for Prefix6Record {
    fn edge_start(&self) -> u32 {
        self.edge_start
    }
    fn edge_len(&self) -> u32 {
        self.edge_len
    }
    fn total(&self) -> u64 {
        self.total
    }
    fn context_label(&self) -> &'static str {
        "model6 prefix"
    }
}

impl PrefixRecord for Prefix5Record {
    fn edge_start(&self) -> u32 {
        self.edge_start
    }
    fn edge_len(&self) -> u32 {
        self.edge_len
    }
    fn total(&self) -> u64 {
        self.total
    }
    fn context_label(&self) -> &'static str {
        "model5 prefix"
    }
}

impl PrefixRecord for Prefix4Record {
    fn edge_start(&self) -> u32 {
        self.edge_start
    }
    fn edge_len(&self) -> u32 {
        self.edge_len
    }
    fn total(&self) -> u64 {
        self.total
    }
    fn context_label(&self) -> &'static str {
        "model4 prefix"
    }
}

impl PrefixRecord for Prefix3Record {
    fn edge_start(&self) -> u32 {
        self.edge_start
    }
    fn edge_len(&self) -> u32 {
        self.edge_len
    }
    fn total(&self) -> u64 {
        self.total
    }
    fn context_label(&self) -> &'static str {
        "model3 prefix"
    }
}
