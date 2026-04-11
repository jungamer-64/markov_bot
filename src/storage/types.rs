use std::{collections::HashMap, ops::Range};

use crate::markov::TokenId;

use super::DynError;

#[derive(Debug, Clone, Copy)]
pub(super) struct Header {
    pub(super) magic: [u8; 8],
    pub(super) version: u32,
    pub(super) flags: u32,
    pub(super) tokenizer_version: u32,
    pub(super) normalization_flags: u32,
    pub(super) section_count: u32,
    pub(super) file_size: u64,
    pub(super) checksum: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SectionDescriptor {
    pub(super) kind: u32,
    pub(super) flags: u32,
    pub(super) offset: u64,
    pub(super) size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum SectionKind {
    VocabOffsets = 1,
    VocabBlob = 2,
    Starts = 3,
    Model3Pairs = 4,
    Model3Prefixes = 5,
    Model3Edges = 6,
    Model2Pairs = 7,
    Model2Prefixes = 8,
    Model2Edges = 9,
    Model1Prefixes = 10,
    Model1Edges = 11,
}

impl SectionKind {
    pub(super) const ALL: [Self; super::SECTION_COUNT] = [
        Self::VocabOffsets,
        Self::VocabBlob,
        Self::Starts,
        Self::Model3Pairs,
        Self::Model3Prefixes,
        Self::Model3Edges,
        Self::Model2Pairs,
        Self::Model2Prefixes,
        Self::Model2Edges,
        Self::Model1Prefixes,
        Self::Model1Edges,
    ];

    pub(super) const fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::VocabOffsets),
            2 => Some(Self::VocabBlob),
            3 => Some(Self::Starts),
            4 => Some(Self::Model3Pairs),
            5 => Some(Self::Model3Prefixes),
            6 => Some(Self::Model3Edges),
            7 => Some(Self::Model2Pairs),
            8 => Some(Self::Model2Prefixes),
            9 => Some(Self::Model2Edges),
            10 => Some(Self::Model1Prefixes),
            11 => Some(Self::Model1Edges),
            _ => None,
        }
    }

    pub(super) const fn as_u32(self) -> u32 {
        self as u32
    }

    pub(super) const fn index(self) -> usize {
        match self {
            Self::VocabOffsets => 0,
            Self::VocabBlob => 1,
            Self::Starts => 2,
            Self::Model3Pairs => 3,
            Self::Model3Prefixes => 4,
            Self::Model3Edges => 5,
            Self::Model2Pairs => 6,
            Self::Model2Prefixes => 7,
            Self::Model2Edges => 8,
            Self::Model1Prefixes => 9,
            Self::Model1Edges => 10,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::VocabOffsets => "vocab offsets",
            Self::VocabBlob => "vocab blob",
            Self::Starts => "start records",
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
}

#[derive(Debug, Clone)]
pub(super) struct SectionEntry {
    pub(super) descriptor: SectionDescriptor,
    pub(super) range: Range<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct SectionTable {
    pub(super) entries: Vec<SectionEntry>,
}

impl SectionTable {
    pub(super) fn entry(&self, kind: SectionKind) -> Result<&SectionEntry, DynError> {
        self.entries
            .get(kind.index())
            .ok_or_else(|| format!("section table is missing {}", kind.label()).into())
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct StartRecord {
    pub(super) prefix_id: u32,
    pub(super) cumulative: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Pair3Record {
    pub(super) w1: u32,
    pub(super) w2: u32,
    pub(super) prefix_start: u32,
    pub(super) prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Pair2Record {
    pub(super) w1: u32,
    pub(super) prefix_start: u32,
    pub(super) prefix_len: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Prefix3Record {
    pub(super) w3: u32,
    pub(super) edge_start: u32,
    pub(super) edge_len: u32,
    pub(super) total: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Prefix2Record {
    pub(super) w1: u32,
    pub(super) w2: u32,
    pub(super) edge_start: u32,
    pub(super) edge_len: u32,
    pub(super) total: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Prefix1Record {
    pub(super) w1: u32,
    pub(super) edge_start: u32,
    pub(super) edge_len: u32,
    pub(super) total: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeRecord {
    pub(super) next: u32,
    pub(super) cumulative: u64,
}

#[derive(Debug, Clone)]
pub(super) struct VocabSections {
    pub(super) offsets: Vec<u64>,
    pub(super) blob: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(super) struct Model3Sections {
    pub(super) pairs: Vec<Pair3Record>,
    pub(super) prefixes: Vec<Prefix3Record>,
    pub(super) edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
pub(super) struct Model2Sections {
    pub(super) pairs: Vec<Pair2Record>,
    pub(super) prefixes: Vec<Prefix2Record>,
    pub(super) edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
pub(super) struct Model1Sections {
    pub(super) prefixes: Vec<Prefix1Record>,
    pub(super) edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
pub(super) struct StorageSections {
    pub(super) vocab: VocabSections,
    pub(super) starts: Vec<StartRecord>,
    pub(super) model3: Model3Sections,
    pub(super) model2: Model2Sections,
    pub(super) model1: Model1Sections,
}

pub(super) trait FixedRecord: Sized {
    const SIZE: u64;

    fn encode_into(&self, target: &mut Vec<u8>);
    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError>;
}

impl FixedRecord for StartRecord {
    const SIZE: u64 = super::START_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.prefix_id);
        write_u64(target, self.cumulative);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            prefix_id: read_u32(bytes, cursor)?,
            cumulative: read_u64(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair3Record {
    const SIZE: u64 = super::PAIR3_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.w1);
        write_u32(target, self.w2);
        write_u32(target, self.prefix_start);
        write_u32(target, self.prefix_len);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            w1: read_u32(bytes, cursor)?,
            w2: read_u32(bytes, cursor)?,
            prefix_start: read_u32(bytes, cursor)?,
            prefix_len: read_u32(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Pair2Record {
    const SIZE: u64 = super::PAIR2_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.w1);
        write_u32(target, self.prefix_start);
        write_u32(target, self.prefix_len);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            w1: read_u32(bytes, cursor)?,
            prefix_start: read_u32(bytes, cursor)?,
            prefix_len: read_u32(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix3Record {
    const SIZE: u64 = super::PREFIX3_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.w3);
        write_u32(target, self.edge_start);
        write_u32(target, self.edge_len);
        write_u64(target, self.total);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            w3: read_u32(bytes, cursor)?,
            edge_start: read_u32(bytes, cursor)?,
            edge_len: read_u32(bytes, cursor)?,
            total: read_u64(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix2Record {
    const SIZE: u64 = super::PREFIX2_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.w1);
        write_u32(target, self.w2);
        write_u32(target, self.edge_start);
        write_u32(target, self.edge_len);
        write_u64(target, self.total);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            w1: read_u32(bytes, cursor)?,
            w2: read_u32(bytes, cursor)?,
            edge_start: read_u32(bytes, cursor)?,
            edge_len: read_u32(bytes, cursor)?,
            total: read_u64(bytes, cursor)?,
        })
    }
}

impl FixedRecord for Prefix1Record {
    const SIZE: u64 = super::PREFIX1_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.w1);
        write_u32(target, self.edge_start);
        write_u32(target, self.edge_len);
        write_u64(target, self.total);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            w1: read_u32(bytes, cursor)?,
            edge_start: read_u32(bytes, cursor)?,
            edge_len: read_u32(bytes, cursor)?,
            total: read_u64(bytes, cursor)?,
        })
    }
}

impl FixedRecord for EdgeRecord {
    const SIZE: u64 = super::EDGE_RECORD_SIZE;

    fn encode_into(&self, target: &mut Vec<u8>) {
        write_u32(target, self.next);
        write_u64(target, self.cumulative);
    }

    fn decode_from(bytes: &[u8], cursor: &mut usize) -> Result<Self, DynError> {
        Ok(Self {
            next: read_u32(bytes, cursor)?,
            cumulative: read_u64(bytes, cursor)?,
        })
    }
}

fn read_exact<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8], DynError> {
    let end = cursor.checked_add(count).ok_or("cursor overflow")?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or("unexpected EOF while decoding record")?;
    *cursor = end;
    Ok(slice)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, DynError> {
    let raw = read_exact(bytes, cursor, 4)?;
    let mut array = [0_u8; 4];
    array.copy_from_slice(raw);
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, DynError> {
    let raw = read_exact(bytes, cursor, 8)?;
    let mut array = [0_u8; 8];
    array.copy_from_slice(raw);
    Ok(u64::from_le_bytes(array))
}

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(value.to_le_bytes().as_slice());
}

pub(super) type Model3PrefixIndex = HashMap<[TokenId; 3], u32>;
