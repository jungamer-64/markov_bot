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
        section_bytes(bytes, table, SectionKind::VocabOffsets)?,
        SectionKind::VocabOffsets.label(),
    )?;
    vocab::validate_vocab_offsets(vocab_offsets.as_slice())?;

    let vocab_blob_size = *vocab_offsets.last().ok_or("vocab offsets are empty")?;
    let vocab_blob = compression::decode_vocab_blob(
        section_bytes(bytes, table, SectionKind::VocabBlob)?,
        usize_from_u64(vocab_blob_size, "vocab blob size")?,
        header.flags,
    )?;

    Ok(StorageSections {
        vocab: VocabSections {
            offsets: vocab_offsets,
            blob: vocab_blob,
        },
        starts: records::parse_fixed_section(
            section_bytes(bytes, table, SectionKind::Starts)?,
            SectionKind::Starts.label(),
        )?,
        model3: Model3Sections {
            pairs: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model3Pairs)?,
                SectionKind::Model3Pairs.label(),
            )?,
            prefixes: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model3Prefixes)?,
                SectionKind::Model3Prefixes.label(),
            )?,
            edges: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model3Edges)?,
                SectionKind::Model3Edges.label(),
            )?,
        },
        model2: Model2Sections {
            pairs: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model2Pairs)?,
                SectionKind::Model2Pairs.label(),
            )?,
            prefixes: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model2Prefixes)?,
                SectionKind::Model2Prefixes.label(),
            )?,
            edges: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model2Edges)?,
                SectionKind::Model2Edges.label(),
            )?,
        },
        model1: Model1Sections {
            prefixes: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model1Prefixes)?,
                SectionKind::Model1Prefixes.label(),
            )?,
            edges: records::parse_fixed_section(
                section_bytes(bytes, table, SectionKind::Model1Edges)?,
                SectionKind::Model1Edges.label(),
            )?,
        },
    })
}

fn section_bytes<'a>(
    bytes: &'a [u8],
    table: &SectionTable,
    kind: SectionKind,
) -> Result<&'a [u8], DynError> {
    let range = table.entry(kind)?.range.clone();
    bytes
        .get(range)
        .ok_or_else(|| format!("{} range is out of bounds", kind.label()).into())
}
