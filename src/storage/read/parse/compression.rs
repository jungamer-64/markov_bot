use super::super::super::{DynError, u64_from_usize};

const REPEAT_BASE: u8 = 128;
const REPEAT_CHUNK_MIN: usize = 3;
const REPEAT_CHUNK_MAX: usize = 130;
const MAX_RLE_EXPANSION_PER_ENCODED_BYTE: usize = REPEAT_CHUNK_MAX / 2;

pub(super) fn decode_vocab_blob(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
    flags: u32,
) -> Result<Vec<u8>, DynError> {
    if flags & super::super::super::FLAG_VOCAB_BLOB_RLE != 0 {
        decode_vocab_blob_rle(vocab_blob_bytes, expected_size)
    } else {
        decode_vocab_blob_plain(vocab_blob_bytes, expected_size)
    }
}

fn decode_vocab_blob_plain(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, DynError> {
    let stored_size = u64_from_usize(vocab_blob_bytes.len(), "vocab blob size")?;
    let expected = u64_from_usize(expected_size, "vocab blob size")?;
    if expected > stored_size {
        return Err("vocab blob size exceeds stored section".into());
    }

    let blob = vocab_blob_bytes
        .get(..expected_size)
        .ok_or("vocab blob range is invalid")?
        .to_vec();

    Ok(blob)
}

fn decode_vocab_blob_rle(
    vocab_blob_bytes: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, DynError> {
    validate_rle_expected_size(vocab_blob_bytes.len(), expected_size)?;

    let mut decoded = Vec::with_capacity(expected_size);
    let mut cursor = 0_usize;

    while decoded.len() < expected_size {
        let control = *vocab_blob_bytes
            .get(cursor)
            .ok_or("compressed vocab blob is truncated")?;
        cursor += 1;

        if control < REPEAT_BASE {
            decode_literal_chunk(
                vocab_blob_bytes,
                control,
                &mut cursor,
                &mut decoded,
                expected_size,
            )?;
        } else {
            decode_repeat_chunk(
                vocab_blob_bytes,
                control,
                &mut cursor,
                &mut decoded,
                expected_size,
            )?;
        }
    }

    if cursor != vocab_blob_bytes.len() {
        return Err("compressed vocab blob has trailing bytes".into());
    }

    Ok(decoded)
}

fn validate_rle_expected_size(encoded_size: usize, expected_size: usize) -> Result<(), DynError> {
    let max_decoded_size = encoded_size
        .checked_mul(MAX_RLE_EXPANSION_PER_ENCODED_BYTE)
        .ok_or("compressed vocab blob expansion bound overflow")?;
    if expected_size > max_decoded_size {
        return Err("compressed vocab blob decoded size exceeds supported expansion bound".into());
    }

    Ok(())
}

fn decode_literal_chunk(
    source: &[u8],
    control: u8,
    cursor: &mut usize,
    decoded: &mut Vec<u8>,
    expected_size: usize,
) -> Result<(), DynError> {
    let literal_len = usize::from(control) + 1;
    let end = cursor
        .checked_add(literal_len)
        .ok_or("compressed vocab literal range overflow")?;

    let chunk = source
        .get(*cursor..end)
        .ok_or("compressed vocab blob literal chunk is truncated")?;

    append_chunk(
        decoded,
        chunk,
        expected_size,
        "compressed vocab blob literal",
    )?;
    *cursor = end;

    Ok(())
}

fn decode_repeat_chunk(
    source: &[u8],
    control: u8,
    cursor: &mut usize,
    decoded: &mut Vec<u8>,
    expected_size: usize,
) -> Result<(), DynError> {
    let repeat_len = usize::from(control - REPEAT_BASE) + REPEAT_CHUNK_MIN;
    let value = *source
        .get(*cursor)
        .ok_or("compressed vocab blob repeat chunk is truncated")?;
    *cursor += 1;

    let next_size = decoded
        .len()
        .checked_add(repeat_len)
        .ok_or("compressed vocab blob size overflow")?;
    if next_size > expected_size {
        return Err("compressed vocab blob repeat exceeds expected decoded size".into());
    }

    decoded.resize(next_size, value);

    Ok(())
}

fn append_chunk(
    decoded: &mut Vec<u8>,
    chunk: &[u8],
    expected_size: usize,
    context: &str,
) -> Result<(), DynError> {
    let next_size = decoded
        .len()
        .checked_add(chunk.len())
        .ok_or("compressed vocab blob size overflow")?;
    if next_size > expected_size {
        return Err(format!("{context} exceeds expected decoded size").into());
    }

    decoded.extend_from_slice(chunk);
    Ok(())
}
