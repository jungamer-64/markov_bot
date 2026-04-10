use super::super::super::{DynError, MarkovChain, u32_from_usize};

pub(super) fn validate_token_index(chain: &MarkovChain) -> Result<(), DynError> {
    for (index, token) in chain.id_to_token.iter().enumerate() {
        let token_id = u32_from_usize(index, "token index")?;

        let Some(stored_id) = chain.token_to_id.get(token).copied() else {
            return Err(format!("token '{token}' is missing in token_to_id").into());
        };

        if stored_id != token_id {
            return Err(format!("token '{token}' index mismatch").into());
        }
    }

    Ok(())
}
