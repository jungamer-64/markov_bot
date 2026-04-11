use tokio::fs;

use crate::config::DynError;
use crate::test_support::{ensure, ensure_ne};

use super::super::{
    CHECKSUM_OFFSET, NORMALIZATION_FLAGS, SECTION_COUNT_U32, SectionKind, TOKENIZER_VERSION,
    VERSION, load_chain,
};
use super::helpers::{
    FILE_SIZE_OFFSET, FLAGS_OFFSET, NORMALIZATION_FLAGS_OFFSET, SECTION_COUNT_OFFSET,
    TOKENIZER_VERSION_OFFSET, UNSUPPORTED_FLAG, VERSION_OFFSET, descriptor_kind_offset,
    descriptor_offset_offset, descriptor_size_offset, first_fixed_section_with_gap,
    first_padding_offset, read_u32_at, read_u64_at, rewrite_checksum, run_async_test, sample_chain,
    write_sample_file, write_u32_at, write_u64_at,
};

#[test]
fn rejects_invalid_magic() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("invalid_magic", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        let magic = bytes.first_mut().ok_or("magic byte should exist")?;
        *magic = b'X';
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "invalid magic should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_version_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("version_mismatch", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u32_at(&mut bytes, VERSION_OFFSET, VERSION.saturating_add(1))?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "version mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_flags_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("flags_mismatch", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u32_at(&mut bytes, FLAGS_OFFSET, UNSUPPORTED_FLAG)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "unsupported flags should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_tokenizer_or_normalization_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("preprocess_mismatch", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u32_at(
            &mut bytes,
            TOKENIZER_VERSION_OFFSET,
            TOKENIZER_VERSION.saturating_add(1),
        )?;
        fs::write(&file_path, &bytes).await?;
        ensure(
            load_chain(&file_path).await.is_err(),
            "tokenizer mismatch should be rejected",
        )?;

        write_u32_at(&mut bytes, TOKENIZER_VERSION_OFFSET, TOKENIZER_VERSION)?;
        write_u32_at(
            &mut bytes,
            NORMALIZATION_FLAGS_OFFSET,
            NORMALIZATION_FLAGS.saturating_add(1),
        )?;
        fs::write(&file_path, bytes).await?;
        ensure(
            load_chain(&file_path).await.is_err(),
            "normalization mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_section_count_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("section_count_mismatch", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u32_at(
            &mut bytes,
            SECTION_COUNT_OFFSET,
            SECTION_COUNT_U32.saturating_sub(1),
        )?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "section count mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_file_size_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("filesize_mismatch", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        let file_size = read_u64_at(&bytes, FILE_SIZE_OFFSET)?;
        write_u64_at(&mut bytes, FILE_SIZE_OFFSET, file_size.saturating_add(1))?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "file size mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_checksum_mismatch() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("checksum_non_zero", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u64_at(&mut bytes, CHECKSUM_OFFSET, 1)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "checksum mismatch should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn writes_non_zero_checksum() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("checksum_written", &chain).await?;
        let bytes = fs::read(&file_path).await?;

        let checksum = read_u64_at(&bytes, CHECKSUM_OFFSET)?;
        ensure_ne(&checksum, &0, "serialized checksum should be non-zero")?;
        Ok(())
    })
}

#[test]
fn rejects_unknown_section_kind() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("unknown_section_kind", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u32_at(
            &mut bytes,
            descriptor_kind_offset(SectionKind::Model2Pairs),
            99,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "unknown section kind should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_duplicate_section_kind() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("duplicate_section_kind", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;
        write_u32_at(
            &mut bytes,
            descriptor_kind_offset(SectionKind::Model2Pairs),
            SectionKind::Model3Edges.as_u32(),
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "duplicate section kinds should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_descriptor_order_violation() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("descriptor_order", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let left_offset = descriptor_kind_offset(SectionKind::Model2Pairs);
        let right_offset = descriptor_kind_offset(SectionKind::Model2Prefixes);
        let left_kind = read_u32_at(&bytes, left_offset)?;
        let right_kind = read_u32_at(&bytes, right_offset)?;
        write_u32_at(&mut bytes, left_offset, right_kind)?;
        write_u32_at(&mut bytes, right_offset, left_kind)?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "descriptor order violation should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_non_8_byte_aligned_section_offset() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("unaligned_offset", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let offset_location = descriptor_offset_offset(SectionKind::Model2Pairs);
        let offset = read_u64_at(&bytes, offset_location)?;
        write_u64_at(&mut bytes, offset_location, offset.saturating_add(1))?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "unaligned section offsets should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_overlapping_sections() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("section_overlap", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let overlap_target =
            read_u64_at(&bytes, descriptor_offset_offset(SectionKind::Model3Edges))?;
        write_u64_at(
            &mut bytes,
            descriptor_offset_offset(SectionKind::Model2Pairs),
            overlap_target,
        )?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "overlapping sections should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_non_zero_padding() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("padding_corrupt", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let Some(padding_offset) = first_padding_offset(&bytes)? else {
            return Err("sample file should contain padding".into());
        };
        let padding = bytes
            .get_mut(padding_offset)
            .ok_or("padding byte should exist")?;
        *padding = 1;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "non-zero padding should be rejected",
        )?;
        Ok(())
    })
}

#[test]
fn rejects_fixed_size_section_size_misalignment() -> Result<(), DynError> {
    run_async_test(async {
        let chain = sample_chain()?;
        let file_path = write_sample_file("section_size_misalignment", &chain).await?;
        let mut bytes = fs::read(&file_path).await?;

        let Some(kind) = first_fixed_section_with_gap(&bytes)? else {
            return Err("sample file should contain a gapped fixed section".into());
        };
        let size_offset = descriptor_size_offset(kind);
        let size = read_u64_at(&bytes, size_offset)?;
        write_u64_at(&mut bytes, size_offset, size.saturating_add(1))?;
        rewrite_checksum(&mut bytes)?;
        fs::write(&file_path, bytes).await?;

        ensure(
            load_chain(&file_path).await.is_err(),
            "misaligned fixed-size section should be rejected",
        )?;
        Ok(())
    })
}
