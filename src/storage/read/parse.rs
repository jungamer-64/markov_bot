use super::super::{
    DynError, Header, Model1Sections, Model2Sections, Model3Sections, SectionKind, SectionTable,
    StorageSections, VocabSections, usize_from_u64,
};

pub(super) mod compression;
pub(super) mod records;
pub(super) mod vocab;

pub(super) fn parse_storage(
    bytes: &[u8],
    header: &Header,
    table: &SectionTable,
) -> Result<StorageSections, DynError> {
    let vocab_offsets = records::parse_u64_section(
        bytes[table.entry(SectionKind::VocabOffsets).range.clone()].as_ref(),
        SectionKind::VocabOffsets.label(),
    )?;
    vocab::validate_vocab_offsets(vocab_offsets.as_slice())?;

    let vocab_blob_size = *vocab_offsets.last().ok_or("vocab offsets are empty")?;
    let vocab_blob = compression::decode_vocab_blob(
        bytes[table.entry(SectionKind::VocabBlob).range.clone()].as_ref(),
        usize_from_u64(vocab_blob_size, "vocab blob size")?,
        header.flags,
    )?;

    Ok(StorageSections {
        vocab: VocabSections {
            offsets: vocab_offsets,
            blob: vocab_blob,
        },
        starts: records::parse_fixed_section(
            bytes[table.entry(SectionKind::Starts).range.clone()].as_ref(),
            SectionKind::Starts.label(),
        )?,
        model3: Model3Sections {
            pairs: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model3Pairs).range.clone()].as_ref(),
                SectionKind::Model3Pairs.label(),
            )?,
            prefixes: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model3Prefixes).range.clone()].as_ref(),
                SectionKind::Model3Prefixes.label(),
            )?,
            edges: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model3Edges).range.clone()].as_ref(),
                SectionKind::Model3Edges.label(),
            )?,
        },
        model2: Model2Sections {
            pairs: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model2Pairs).range.clone()].as_ref(),
                SectionKind::Model2Pairs.label(),
            )?,
            prefixes: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model2Prefixes).range.clone()].as_ref(),
                SectionKind::Model2Prefixes.label(),
            )?,
            edges: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model2Edges).range.clone()].as_ref(),
                SectionKind::Model2Edges.label(),
            )?,
        },
        model1: Model1Sections {
            prefixes: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model1Prefixes).range.clone()].as_ref(),
                SectionKind::Model1Prefixes.label(),
            )?,
            edges: records::parse_fixed_section(
                bytes[table.entry(SectionKind::Model1Edges).range.clone()].as_ref(),
                SectionKind::Model1Edges.label(),
            )?,
        },
    })
}
