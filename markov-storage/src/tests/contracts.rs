use super::test_support::{ensure, rewrite_checksum, write_u64_at};
use crate::{Codec, LimitKind, StorageCompressionMode, StorageError, StorageLimits};
use markov_core::{Count, MarkovChain, NgramOrder};

// Specification vector: order 1, reserved vocabulary only, no starts or models.
// Header, descriptors and checksum are authored independently of the production writer.
fn empty_v8() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MKV3BIN\0");
    for value in [8_u32, 0, 1, 0, 1] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [4_u64, 200, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for (kind, flags, offset, size) in [
        (1_u32, 0_u32, 152_u64, 24_u64),
        (2, 0, 176, 10),
        (3, 0, 192, 4),
        (4, 1, 196, 8),
    ] {
        // Model bodies are not required to be aligned; only metadata is aligned.
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
    }
    bytes.resize(152, 0);
    for offset in [0_u64, 5, 10] {
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    bytes.extend_from_slice(b"<BOS><EOS>");
    bytes.resize(204, 0);
    let mut size_field = bytes.iter_mut().skip(36).take(8);
    for byte in 204_u64.to_le_bytes() {
        if let Some(slot) = size_field.next() {
            *slot = byte;
        }
    }
    let mut checksum = 14_695_981_039_346_656_037_u64;
    for byte in &bytes {
        checksum = (checksum ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211);
    }
    for (slot, byte) in bytes
        .iter_mut()
        .skip(44)
        .take(8)
        .zip(checksum.to_le_bytes())
    {
        *slot = byte;
    }
    bytes
}

#[test]
fn independent_empty_vector_and_limits() -> Result<(), StorageError> {
    let bytes = empty_v8();
    let codec = Codec::new(StorageLimits::new(204, 10)?);
    let chain = codec.decode_chain(&bytes, NgramOrder::new(1)?)?;
    ensure(chain.starts().is_empty(), "empty vector has no starts")?;
    ensure(
        chain.registry().tokens() == ["<BOS>", "<EOS>"],
        "reserved vocabulary",
    )?;
    let too_small = Codec::new(StorageLimits::new(203, 10)?);
    ensure(
        matches!(
            too_small.decode_snapshot(&bytes),
            Err(StorageError::Limit {
                kind: LimitKind::FileBytes,
                ..
            })
        ),
        "file limit before parsing",
    )?;
    let too_small = Codec::new(StorageLimits::new(204, 9)?);
    ensure(
        matches!(
            too_small.decode_snapshot(&bytes),
            Err(StorageError::Limit {
                kind: LimitKind::VocabBytes,
                ..
            })
        ),
        "vocabulary limit before decompression",
    )
}

#[test]
fn pruning_preserves_only_reachable_starts() -> Result<(), StorageError> {
    let codec = Codec::default();
    let mut chain = MarkovChain::new(NgramOrder::new(2)?)?;
    chain.train_tokens(&["one".into()])?;
    let bytes = codec.encode_chain(&chain, Count::new(2), StorageCompressionMode::Uncompressed)?;
    let restored = codec.decode_chain(&bytes, chain.order())?;
    ensure(
        restored.starts().is_empty()
            && restored
                .models()
                .iter()
                .all(std::collections::HashMap::is_empty),
        "fully pruned model must be empty",
    )?;
    chain.train_tokens(&["one".into()])?;
    let bytes = codec.encode_chain(&chain, Count::new(2), StorageCompressionMode::Uncompressed)?;
    ensure(
        !codec
            .decode_chain(&bytes, chain.order())?
            .starts()
            .is_empty(),
        "retained prefix keeps start",
    )
}

#[test]
fn output_obeys_same_limits_as_input() -> Result<(), StorageError> {
    let chain = MarkovChain::new(NgramOrder::new(1)?)?;
    let bytes = Codec::default().encode_chain(
        &chain,
        Count::new(1),
        StorageCompressionMode::Uncompressed,
    )?;
    let size =
        u64::try_from(bytes.len()).map_err(|error| StorageError::Format(error.to_string()))?;
    let exact = Codec::new(StorageLimits::new(size, 10)?);
    ensure(
        exact.encode_chain(&chain, Count::new(1), StorageCompressionMode::Uncompressed)? == bytes,
        "exact limit accepted",
    )?;
    ensure(
        matches!(
            Codec::new(StorageLimits::new(size - 1, 10)?).encode_chain(
                &chain,
                Count::new(1),
                StorageCompressionMode::Uncompressed
            ),
            Err(StorageError::Limit {
                kind: LimitKind::FileBytes,
                ..
            })
        ),
        "output rejects over-limit file",
    )
}

#[test]
fn rejects_oversized_metadata_before_allocation() -> Result<(), StorageError> {
    let mut bytes = empty_v8();
    super::test_support::write_u32_at(&mut bytes, 24, u32::MAX)?;
    write_u64_at(&mut bytes, 28, u64::from(u32::MAX) + 3)?;
    rewrite_checksum(&mut bytes)?;
    ensure(
        Codec::default().decode_snapshot(&bytes).is_err(),
        "huge metadata rejected without allocating from count",
    )
}

#[test]
fn corrupt_compressed_size_is_bounded() -> Result<(), StorageError> {
    let chain = MarkovChain::new(NgramOrder::new(1)?)?;
    for mode in [StorageCompressionMode::Rle, StorageCompressionMode::Zstd] {
        let mut bytes = Codec::default().encode_chain(&chain, Count::new(1), mode)?;
        let offset = super::test_support::section_body_offset(&bytes, 0)?;
        write_u64_at(&mut bytes, offset + 16, u64::MAX)?;
        rewrite_checksum(&mut bytes)?;
        ensure(
            matches!(
                Codec::default().decode_snapshot(&bytes),
                Err(StorageError::Limit {
                    kind: LimitKind::VocabBytes,
                    ..
                })
            ),
            "untrusted decoded size rejected before decompression",
        )?;
    }
    Ok(())
}

#[test]
fn v8_start_without_highest_transition_is_readable() -> Result<(), StorageError> {
    // A v8 start is a context, not a foreign key into the highest-order model.
    let mut bytes = empty_v8();
    bytes.truncate(192);
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.resize(216, 0);
    write_u64_at(&mut bytes, 36, 216)?;
    write_u64_at(&mut bytes, 116, 16)?;
    write_u64_at(&mut bytes, 132, 208)?;
    rewrite_checksum(&mut bytes)?;
    let chain = Codec::default().decode_chain(&bytes, NgramOrder::new(1)?)?;
    ensure(chain.starts().len() == 1, "v8 start context preserved")?;
    ensure(
        chain
            .models()
            .iter()
            .all(std::collections::HashMap::is_empty),
        "empty transitions preserved",
    )
}
