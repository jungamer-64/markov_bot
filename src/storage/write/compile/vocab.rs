use super::super::super::{DynError, checked_add, u64_from_usize};

pub(super) fn build_vocab(tokens: &[String]) -> Result<(Vec<u64>, Vec<u8>), DynError> {
    let mut offsets = Vec::with_capacity(tokens.len().saturating_add(1));
    let mut blob = Vec::new();
    let mut position = 0_u64;

    offsets.push(0);

    for token in tokens {
        let token_bytes = token.as_bytes();
        blob.extend_from_slice(token_bytes);

        let token_len = u64_from_usize(token_bytes.len(), "token byte length")?;
        position = checked_add(position, token_len, "vocab blob size")?;
        offsets.push(position);
    }

    Ok((offsets, blob))
}
