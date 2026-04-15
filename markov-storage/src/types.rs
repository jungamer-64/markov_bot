use std::ops::Range;

use crate::{
    config::DynError,
    markov::{Count, Prefix, TokenId},
};

#[derive(Debug, Clone, Copy)]
pub(super) struct Header {
    pub(super) magic: [u8; 8],
    pub(super) version: u32,
    pub(super) flags: u32,
    pub(super) tokenizer_version: u32,
    pub(super) normalization_flags: u32,
    pub(super) ngram_order: u32,
    pub(super) section_count: u64,
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
    Model = 4,
}

impl SectionKind {
    pub(super) const fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::VocabOffsets),
            2 => Some(Self::VocabBlob),
            3 => Some(Self::Starts),
            4 => Some(Self::Model),
            _ => None,
        }
    }

    pub(super) const fn as_u32(self) -> u32 {
        match self {
            Self::VocabOffsets => 1,
            Self::VocabBlob => 2,
            Self::Starts => 3,
            Self::Model => 4,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::VocabOffsets => "vocab offsets",
            Self::VocabBlob => "vocab blob",
            Self::Starts => "start records",
            Self::Model => "model records",
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
    pub(super) fn unique_entry(&self, kind: SectionKind) -> Result<&SectionEntry, DynError> {
        let mut matches = self
            .entries
            .iter()
            .filter(|entry| SectionKind::from_u32(entry.descriptor.kind) == Some(kind));

        let entry = matches
            .next()
            .ok_or_else(|| format!("section table is missing {}", kind.label()))?;
        if matches.next().is_some() {
            return Err(format!("section table has duplicate {}", kind.label()).into());
        }

        Ok(entry)
    }

    pub(super) fn model_entries(&self) -> impl Iterator<Item = &SectionEntry> {
        self.entries.iter().filter(|entry| {
            SectionKind::from_u32(entry.descriptor.kind) == Some(SectionKind::Model)
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct StartRecord {
    pub(super) prefix: Prefix,
    pub(super) cumulative: Count,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeRecord {
    pub(super) next: TokenId,
    pub(super) cumulative: Count,
}

#[derive(Debug, Clone)]
pub(super) struct ModelRecord {
    pub(super) prefix: Prefix,
    pub(super) edge_start: u32,
    pub(super) edge_len: u32,
    pub(super) total: Count,
}

#[derive(Debug, Clone)]
pub(super) struct ModelSection {
    pub(super) order: usize,
    pub(super) records: Vec<ModelRecord>,
    pub(super) edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone)]
pub(super) struct VocabSections {
    pub(super) offsets: Vec<u64>,
    pub(super) blob: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(super) struct StorageSections {
    pub(super) ngram_order: usize,
    pub(super) vocab: VocabSections,
    pub(super) starts: Vec<StartRecord>,
    pub(super) models: Vec<ModelSection>,
}
