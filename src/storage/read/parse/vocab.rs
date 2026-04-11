use std::str;

use super::super::super::{DynError, usize_from_u64};

pub(in crate::storage::read) fn decode_vocab(
    offsets: &[u64],
    blob: &[u8],
) -> Result<Vec<String>, DynError> {
    if offsets.is_empty() {
        return Err("vocab offsets are empty".into());
    }

    let mut tokens = Vec::with_capacity(offsets.len().saturating_sub(1));
    for pair in offsets.windows(2) {
        let [start_offset, end_offset] = <&[u64; 2]>::try_from(pair)
            .map_err(|_error| "vocab offset pair must contain two values")?;
        let start = usize_from_u64(*start_offset, "vocab token start")?;
        let end = usize_from_u64(*end_offset, "vocab token end")?;
        let token_bytes = blob.get(start..end).ok_or("vocab token range is invalid")?;
        let token = str::from_utf8(token_bytes)
            .map_err(|_error| "vocab token is not valid UTF-8")?
            .to_owned();
        tokens.push(token);
    }

    Ok(tokens)
}

pub(super) fn validate_vocab_offsets(offsets: &[u64]) -> Result<(), DynError> {
    if offsets.first().copied() != Some(0) {
        return Err("vocab offsets must start with 0".into());
    }

    for pair in offsets.windows(2) {
        let [start_offset, end_offset] = <&[u64; 2]>::try_from(pair)
            .map_err(|_error| "vocab offset pair must contain two values")?;
        if start_offset > end_offset {
            return Err("vocab offsets must be non-decreasing".into());
        }
    }

    Ok(())
}
