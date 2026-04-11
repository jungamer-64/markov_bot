use lz4_flex::block::compress as lz4_compress;

use super::super::super::{
    DynError, FLAG_VOCAB_BLOB_LZ4_FLEX, FLAG_VOCAB_BLOB_RLE, FLAG_VOCAB_BLOB_ZSTD, FLAGS,
    StorageCompressionMode,
};

const LITERAL_CHUNK_MAX: usize = 128;
const REPEAT_CHUNK_MIN: usize = 3;
const REPEAT_CHUNK_MAX: usize = 130;

pub(super) struct EncodedVocabBlob {
    pub(super) bytes: Vec<u8>,
    pub(super) flags: u32,
}

pub(super) fn encode_vocab_blob(
    vocab_blob: &[u8],
    compression_mode: StorageCompressionMode,
) -> Result<EncodedVocabBlob, DynError> {
    match compression_mode {
        StorageCompressionMode::Auto => encode_auto(vocab_blob),
        StorageCompressionMode::Uncompressed => Ok(EncodedVocabBlob {
            bytes: vocab_blob.to_vec(),
            flags: FLAGS,
        }),
        StorageCompressionMode::Rle => Ok(EncodedVocabBlob {
            bytes: encode_rle(vocab_blob)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_RLE,
        }),
        StorageCompressionMode::Zstd => Ok(EncodedVocabBlob {
            bytes: zstd::bulk::compress(vocab_blob, 0)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_ZSTD,
        }),
        StorageCompressionMode::Lz4Flex => Ok(EncodedVocabBlob {
            bytes: lz4_compress(vocab_blob),
            flags: FLAGS | FLAG_VOCAB_BLOB_LZ4_FLEX,
        }),
    }
}

fn encode_auto(vocab_blob: &[u8]) -> Result<EncodedVocabBlob, DynError> {
    if vocab_blob.is_empty() {
        return Ok(EncodedVocabBlob {
            bytes: Vec::new(),
            flags: FLAGS,
        });
    }

    let mut best = EncodedVocabBlob {
        bytes: vocab_blob.to_vec(),
        flags: FLAGS,
    };

    for candidate in [
        EncodedVocabBlob {
            bytes: encode_rle(vocab_blob)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_RLE,
        },
        EncodedVocabBlob {
            bytes: zstd::bulk::compress(vocab_blob, 0)?,
            flags: FLAGS | FLAG_VOCAB_BLOB_ZSTD,
        },
        EncodedVocabBlob {
            bytes: lz4_compress(vocab_blob),
            flags: FLAGS | FLAG_VOCAB_BLOB_LZ4_FLEX,
        },
    ] {
        if candidate.bytes.len() < best.bytes.len() {
            best = candidate;
        }
    }

    Ok(best)
}

fn encode_rle(input: &[u8]) -> Result<Vec<u8>, DynError> {
    let mut encoded = Vec::with_capacity(input.len());
    let mut cursor = 0_usize;

    while cursor < input.len() {
        let repeat_len = repeat_run_len(input, cursor);
        if repeat_len >= REPEAT_CHUNK_MIN {
            let value = *input.get(cursor).ok_or("rle cursor is out of bounds")?;
            push_repeat_chunk(&mut encoded, value, repeat_len)?;
            cursor += repeat_len;
            continue;
        }

        let literal_len = literal_run_len(input, cursor);
        let literal_end = cursor
            .checked_add(literal_len)
            .ok_or("rle literal length overflow")?;
        let literal = input
            .get(cursor..literal_end)
            .ok_or("rle literal range is out of bounds")?;
        push_literal_chunk(&mut encoded, literal)?;
        cursor += literal_len;
    }

    Ok(encoded)
}

fn repeat_run_len(input: &[u8], start: usize) -> usize {
    let Some(&value) = input.get(start) else {
        return 0;
    };
    let mut len = 1_usize;

    while start + len < input.len()
        && input.get(start + len).copied() == Some(value)
        && len < REPEAT_CHUNK_MAX
    {
        len += 1;
    }

    len
}

fn literal_run_len(input: &[u8], start: usize) -> usize {
    let mut len = 1_usize;

    while start + len < input.len() && len < LITERAL_CHUNK_MAX {
        if repeat_run_len(input, start + len) >= REPEAT_CHUNK_MIN {
            break;
        }
        len += 1;
    }

    len
}

fn push_repeat_chunk(encoded: &mut Vec<u8>, value: u8, repeat_len: usize) -> Result<(), DynError> {
    let repeat_control = repeat_len - REPEAT_CHUNK_MIN;
    let control =
        u8::try_from(repeat_control).map_err(|_error| "repeat chunk length is bounded")?;
    let control = control.saturating_add(128);
    encoded.push(control);
    encoded.push(value);
    Ok(())
}

fn push_literal_chunk(encoded: &mut Vec<u8>, bytes: &[u8]) -> Result<(), DynError> {
    let literal_control = bytes.len() - 1;
    let control =
        u8::try_from(literal_control).map_err(|_error| "literal chunk length is bounded")?;
    encoded.push(control);
    encoded.extend_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{REPEAT_CHUNK_MIN, encode_vocab_blob};
    use crate::config::DynError;
    use crate::storage::StorageCompressionMode;
    use crate::test_support::{ensure, ensure_eq, ensure_ne};

    #[test]
    fn auto_compresses_repeated_vocab_blob() -> Result<(), DynError> {
        let input = vec![b'a'; REPEAT_CHUNK_MIN + 40];
        let encoded = encode_vocab_blob(input.as_slice(), StorageCompressionMode::Auto)?;

        ensure_ne(
            &encoded.flags,
            &0,
            "auto mode should choose a compressed representation",
        )?;
        ensure(
            encoded.bytes.len() < input.len(),
            "auto mode should reduce encoded size for repeated data",
        )?;
        Ok(())
    }

    #[test]
    fn auto_keeps_uncompressed_when_not_beneficial() -> Result<(), DynError> {
        let input = b"abcdefg";
        let encoded = encode_vocab_blob(input, StorageCompressionMode::Auto)?;

        ensure_eq(
            &encoded.flags,
            &0,
            "auto mode should not set a compression flag when keeping the original bytes",
        )?;
        ensure_eq(
            &encoded.bytes.as_slice(),
            &input.as_slice(),
            "auto mode should preserve the original bytes when not compressing",
        )?;
        Ok(())
    }

    #[test]
    fn uncompressed_mode_never_sets_flag() -> Result<(), DynError> {
        let input = vec![b'a'; REPEAT_CHUNK_MIN + 40];
        let encoded = encode_vocab_blob(input.as_slice(), StorageCompressionMode::Uncompressed)?;

        ensure_eq(
            &encoded.flags,
            &0,
            "uncompressed mode should not set a compression flag",
        )?;
        ensure_eq(
            &encoded.bytes.as_slice(),
            &input.as_slice(),
            "uncompressed mode should preserve the original bytes",
        )?;
        Ok(())
    }

    #[test]
    fn rle_mode_sets_flag_even_when_larger() -> Result<(), DynError> {
        let input = b"abcdefg";
        let encoded = encode_vocab_blob(input, StorageCompressionMode::Rle)?;

        ensure_ne(&encoded.flags, &0, "rle mode should set a compression flag")?;
        Ok(())
    }

    #[test]
    fn zstd_mode_sets_flag() -> Result<(), DynError> {
        let input = b"abcdefg";
        let encoded = encode_vocab_blob(input, StorageCompressionMode::Zstd)?;

        ensure_ne(
            &encoded.flags,
            &0,
            "zstd mode should set a compression flag",
        )?;
        Ok(())
    }

    #[test]
    fn lz4_flex_mode_sets_flag() -> Result<(), DynError> {
        let input = b"abcdefg";
        let encoded = encode_vocab_blob(input, StorageCompressionMode::Lz4Flex)?;

        ensure_ne(&encoded.flags, &0, "lz4 mode should set a compression flag")?;
        Ok(())
    }
}
