use std::collections::HashMap;

use crate::markov::TokenId;

#[derive(Debug, Clone, Copy)]
pub(super) struct Header {
    pub(super) magic: [u8; 8],
    pub(super) version: u32,
    pub(super) flags: u32,
    pub(super) tokenizer_version: u32,
    pub(super) normalization_flags: u32,
    pub(super) token_count: u32,
    pub(super) start_count: u32,
    pub(super) model3_pair_count: u32,
    pub(super) model3_prefix_count: u32,
    pub(super) model3_edge_count: u32,
    pub(super) model2_pair_count: u32,
    pub(super) model2_prefix_count: u32,
    pub(super) model2_edge_count: u32,
    pub(super) model1_prefix_count: u32,
    pub(super) model1_edge_count: u32,
    pub(super) vocab_offsets_offset: u64,
    pub(super) vocab_blob_offset: u64,
    pub(super) start_offset: u64,
    pub(super) model3_pair_offset: u64,
    pub(super) model3_prefix_offset: u64,
    pub(super) model3_edge_offset: u64,
    pub(super) model2_pair_offset: u64,
    pub(super) model2_prefix_offset: u64,
    pub(super) model2_edge_offset: u64,
    pub(super) model1_prefix_offset: u64,
    pub(super) model1_edge_offset: u64,
    pub(super) file_size: u64,
    pub(super) checksum: u64,
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

#[derive(Debug)]
pub(super) struct CompiledStorage {
    pub(super) vocab_offsets: Vec<u64>,
    pub(super) vocab_blob: Vec<u8>,
    pub(super) starts: Vec<StartRecord>,
    pub(super) model3_pairs: Vec<Pair3Record>,
    pub(super) model3_prefixes: Vec<Prefix3Record>,
    pub(super) model3_edges: Vec<EdgeRecord>,
    pub(super) model2_pairs: Vec<Pair2Record>,
    pub(super) model2_prefixes: Vec<Prefix2Record>,
    pub(super) model2_edges: Vec<EdgeRecord>,
    pub(super) model1_prefixes: Vec<Prefix1Record>,
    pub(super) model1_edges: Vec<EdgeRecord>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SectionCounts {
    pub(super) token: u32,
    pub(super) start: u32,
    pub(super) model3_pair: u32,
    pub(super) model3_prefix: u32,
    pub(super) model3_edge: u32,
    pub(super) model2_pair: u32,
    pub(super) model2_prefix: u32,
    pub(super) model2_edge: u32,
    pub(super) model1_prefix: u32,
    pub(super) model1_edge: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SectionSizes {
    pub(super) vocab_offsets: u64,
    pub(super) vocab_blob: u64,
    pub(super) starts: u64,
    pub(super) model3_pairs: u64,
    pub(super) model3_prefixes: u64,
    pub(super) model3_edges: u64,
    pub(super) model2_pairs: u64,
    pub(super) model2_prefixes: u64,
    pub(super) model2_edges: u64,
    pub(super) model1_prefixes: u64,
    pub(super) model1_edges: u64,
}

#[derive(Debug, Clone)]
pub(super) struct SectionRanges {
    pub(super) vocab_offsets: std::ops::Range<usize>,
    pub(super) vocab_blob_area: std::ops::Range<usize>,
    pub(super) starts: std::ops::Range<usize>,
    pub(super) model3_pairs: std::ops::Range<usize>,
    pub(super) model3_prefixes: std::ops::Range<usize>,
    pub(super) model3_edges: std::ops::Range<usize>,
    pub(super) model2_pairs: std::ops::Range<usize>,
    pub(super) model2_prefixes: std::ops::Range<usize>,
    pub(super) model2_edges: std::ops::Range<usize>,
    pub(super) model1_prefixes: std::ops::Range<usize>,
    pub(super) model1_edges: std::ops::Range<usize>,
}

#[derive(Debug)]
pub(super) struct ParsedStorage {
    pub(super) id_to_token: Vec<String>,
    pub(super) starts: Vec<StartRecord>,
    pub(super) model3_pairs: Vec<Pair3Record>,
    pub(super) model3_prefixes: Vec<Prefix3Record>,
    pub(super) model3_edges: Vec<EdgeRecord>,
    pub(super) model2_pairs: Vec<Pair2Record>,
    pub(super) model2_prefixes: Vec<Prefix2Record>,
    pub(super) model2_edges: Vec<EdgeRecord>,
    pub(super) model1_prefixes: Vec<Prefix1Record>,
    pub(super) model1_edges: Vec<EdgeRecord>,
}

pub(super) type Model3Build = (
    Vec<Pair3Record>,
    Vec<Prefix3Record>,
    Vec<EdgeRecord>,
    HashMap<[TokenId; 3], u32>,
);
