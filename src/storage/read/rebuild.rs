use std::collections::HashMap;

use super::super::{Count, DynError, MarkovChain, ParsedStorage, TokenId, u32_from_usize};

mod decode;
mod validate;

type DecodedModels = (
    HashMap<[TokenId; 3], Count>,
    HashMap<[TokenId; 3], HashMap<TokenId, Count>>,
    HashMap<[TokenId; 2], HashMap<TokenId, Count>>,
    HashMap<TokenId, HashMap<TokenId, Count>>,
);

pub(super) fn rebuild_chain(parsed: ParsedStorage) -> Result<MarkovChain, DynError> {
    let token_count = u32_from_usize(parsed.id_to_token.len(), "token count")?;

    let model3_keys = validate_storage(&parsed, token_count)?;

    let token_to_id = build_token_index(parsed.id_to_token.as_slice())?;
    let (starts, model3, model2, model1) = decode_models(&parsed, model3_keys.as_slice())?;

    Ok(MarkovChain {
        token_to_id,
        id_to_token: parsed.id_to_token,
        model3,
        model2,
        model1,
        starts,
    })
}

fn validate_storage(
    parsed: &ParsedStorage,
    token_count: u32,
) -> Result<Vec<[TokenId; 3]>, DynError> {
    let model3_keys = validate::validate_and_build_model3_keys(
        parsed.model3_pairs.as_slice(),
        parsed.model3_prefixes.as_slice(),
        parsed.model3_edges.as_slice(),
        token_count,
    )?;
    validate::validate_model2(
        parsed.model2_pairs.as_slice(),
        parsed.model2_prefixes.as_slice(),
        parsed.model2_edges.as_slice(),
        token_count,
    )?;
    validate::validate_model1(
        parsed.model1_prefixes.as_slice(),
        parsed.model1_edges.as_slice(),
        token_count,
    )?;
    validate::validate_starts(parsed.starts.as_slice(), model3_keys.len())?;

    Ok(model3_keys)
}

fn decode_models(
    parsed: &ParsedStorage,
    model3_keys: &[[TokenId; 3]],
) -> Result<DecodedModels, DynError> {
    let starts = decode::decode_starts(parsed.starts.as_slice(), model3_keys)?;
    let model3 = decode::decode_model3(
        model3_keys,
        parsed.model3_prefixes.as_slice(),
        parsed.model3_edges.as_slice(),
    )?;
    let model2 = decode::decode_model2(
        parsed.model2_prefixes.as_slice(),
        parsed.model2_edges.as_slice(),
    )?;
    let model1 = decode::decode_model1(
        parsed.model1_prefixes.as_slice(),
        parsed.model1_edges.as_slice(),
    )?;

    Ok((starts, model3, model2, model1))
}

fn build_token_index(tokens: &[String]) -> Result<HashMap<String, u32>, DynError> {
    let mut index = HashMap::new();

    for (position, token) in tokens.iter().enumerate() {
        let token_id = u32_from_usize(position, "token id")?;

        if index.insert(token.clone(), token_id).is_some() {
            return Err(format!("duplicate token in vocab: {token}").into());
        }
    }

    Ok(index)
}
