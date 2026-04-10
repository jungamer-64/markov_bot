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
            bytes: encode_rle(vocab_blob),
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
            bytes: encode_rle(vocab_blob),
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

fn encode_rle(input: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(input.len());
    let mut cursor = 0_usize;

    while cursor < input.len() {
        let repeat_len = repeat_run_len(input, cursor);
        if repeat_len >= REPEAT_CHUNK_MIN {
            push_repeat_chunk(&mut encoded, input[cursor], repeat_len);
            cursor += repeat_len;
            continue;
        }

        let literal_len = literal_run_len(input, cursor);
        push_literal_chunk(&mut encoded, &input[cursor..cursor + literal_len]);
        cursor += literal_len;
    }

    encoded
}

fn repeat_run_len(input: &[u8], start: usize) -> usize {
    let value = input[start];
    let mut len = 1_usize;

    while start + len < input.len() && input[start + len] == value && len < REPEAT_CHUNK_MAX {
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

fn push_repeat_chunk(encoded: &mut Vec<u8>, value: u8, repeat_len: usize) {
    let control = u8::try_from(repeat_len - REPEAT_CHUNK_MIN)
        .expect("repeat chunk length is bounded")
        .saturating_add(128);
    encoded.push(control);
    encoded.push(value);
}

fn push_literal_chunk(encoded: &mut Vec<u8>, bytes: &[u8]) {
    let control = u8::try_from(bytes.len() - 1).expect("literal chunk length is bounded");
    encoded.push(control);
    encoded.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::{REPEAT_CHUNK_MIN, encode_vocab_blob};
    use crate::storage::StorageCompressionMode;

    #[test]
    fn auto_compresses_repeated_vocab_blob() {
        let input = vec![b'a'; REPEAT_CHUNK_MIN + 40];
        let encoded = encode_vocab_blob(input.as_slice(), StorageCompressionMode::Auto)
            .expect("auto encoding should succeed");

        assert_ne!(encoded.flags, 0);
        assert!(encoded.bytes.len() < input.len());
    }

    #[test]
    fn auto_keeps_uncompressed_when_not_beneficial() {
        let input = b"abcdefg";
        let encoded =
            encode_vocab_blob(input, StorageCompressionMode::Auto).expect("auto encoding succeeds");

        assert_eq!(encoded.flags, 0);
        assert_eq!(encoded.bytes, input);
    }

    #[test]
    fn uncompressed_mode_never_sets_flag() {
        let input = vec![b'a'; REPEAT_CHUNK_MIN + 40];
        let encoded = encode_vocab_blob(input.as_slice(), StorageCompressionMode::Uncompressed)
            .expect("uncompressed encoding succeeds");

        assert_eq!(encoded.flags, 0);
        assert_eq!(encoded.bytes, input);
    }

    #[test]
    fn rle_mode_sets_flag_even_when_larger() {
        let input = b"abcdefg";
        let encoded =
            encode_vocab_blob(input, StorageCompressionMode::Rle).expect("rle encoding succeeds");

        assert_ne!(encoded.flags, 0);
    }

    #[test]
    fn zstd_mode_sets_flag() {
        let input = b"abcdefg";
        let encoded =
            encode_vocab_blob(input, StorageCompressionMode::Zstd).expect("zstd encoding succeeds");

        assert_ne!(encoded.flags, 0);
    }

    #[test]
    fn lz4_flex_mode_sets_flag() {
        let input = b"abcdefg";
        let encoded = encode_vocab_blob(input, StorageCompressionMode::Lz4Flex)
            .expect("lz4_flex encoding succeeds");

        assert_ne!(encoded.flags, 0);
    }
}
