use super::super::{
    DynError, Header, ParsedStorage, SectionRanges, usize_from_u32, usize_from_u64,
    validate_special_tokens,
};

mod compression;
mod records;
mod vocab;

pub(super) fn parse_storage(
    bytes: &[u8],
    header: &Header,
    ranges: &SectionRanges,
) -> Result<ParsedStorage, DynError> {
    let vocab_offsets_len = usize_from_u32(
        header
            .token_count
            .checked_add(1)
            .ok_or("token_count overflow")?,
        "vocab offsets length",
    )?;

    let vocab_offsets = records::parse_u64_values(
        bytes[ranges.vocab_offsets.clone()].as_ref(),
        vocab_offsets_len,
        "vocab offsets",
    )?;
    vocab::validate_vocab_offsets(vocab_offsets.as_slice())?;

    let vocab_blob_size = *vocab_offsets.last().ok_or("vocab offsets are empty")?;
    let vocab_blob = compression::decode_vocab_blob(
        bytes[ranges.vocab_blob.clone()].as_ref(),
        usize_from_u64(vocab_blob_size, "vocab blob size")?,
        header.flags,
    )?;

    let id_to_token = vocab::decode_vocab(vocab_offsets.as_slice(), vocab_blob.as_slice())?;
    validate_special_tokens(id_to_token.as_slice())?;

    Ok(ParsedStorage {
        id_to_token,
        starts: records::parse_start_records(
            bytes[ranges.starts.clone()].as_ref(),
            usize_from_u32(header.start_count, "start count")?,
        )?,
        model3_pairs: records::parse_pair3_records(
            bytes[ranges.model3_pairs.clone()].as_ref(),
            usize_from_u32(header.model3_pair_count, "model3 pair count")?,
        )?,
        model3_prefixes: records::parse_prefix3_records(
            bytes[ranges.model3_prefixes.clone()].as_ref(),
            usize_from_u32(header.model3_prefix_count, "model3 prefix count")?,
        )?,
        model3_edges: records::parse_edge_records(
            bytes[ranges.model3_edges.clone()].as_ref(),
            usize_from_u32(header.model3_edge_count, "model3 edge count")?,
        )?,
        model2_pairs: records::parse_pair2_records(
            bytes[ranges.model2_pairs.clone()].as_ref(),
            usize_from_u32(header.model2_pair_count, "model2 pair count")?,
        )?,
        model2_prefixes: records::parse_prefix2_records(
            bytes[ranges.model2_prefixes.clone()].as_ref(),
            usize_from_u32(header.model2_prefix_count, "model2 prefix count")?,
        )?,
        model2_edges: records::parse_edge_records(
            bytes[ranges.model2_edges.clone()].as_ref(),
            usize_from_u32(header.model2_edge_count, "model2 edge count")?,
        )?,
        model1_prefixes: records::parse_prefix1_records(
            bytes[ranges.model1_prefixes.clone()].as_ref(),
            usize_from_u32(header.model1_prefix_count, "model1 prefix count")?,
        )?,
        model1_edges: records::parse_edge_records(
            bytes[ranges.model1_edges.clone()].as_ref(),
            usize_from_u32(header.model1_edge_count, "model1 edge count")?,
        )?,
    })
}
