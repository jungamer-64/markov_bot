use std::collections::HashMap;

use super::super::{
    Count, DynError, MarkovChain, Model1Sections, Model2Sections, Model3Sections, StartRecord,
    StorageSections, TokenId, u32_from_usize, validate_special_tokens,
};
use super::parse::vocab;

mod decode;
mod validate;

type DecodedModels = (
    HashMap<[TokenId; 3], Count>,
    HashMap<[TokenId; 3], HashMap<TokenId, Count>>,
    HashMap<[TokenId; 2], HashMap<TokenId, Count>>,
    HashMap<TokenId, HashMap<TokenId, Count>>,
);

pub(super) fn rebuild_chain(sections: StorageSections) -> Result<MarkovChain, DynError> {
    let StorageSections {
        vocab: vocab_sections,
        starts: start_records,
        model3: model3_sections,
        model2: model2_sections,
        model1: model1_sections,
    } = sections;

    let id_to_token = vocab::decode_vocab(
        vocab_sections.offsets.as_slice(),
        vocab_sections.blob.as_slice(),
    )?;
    validate_special_tokens(id_to_token.as_slice())?;

    let token_count = u32_from_usize(id_to_token.len(), "token count")?;
    let model3_keys = validate_storage(
        start_records.as_slice(),
        &model3_sections,
        &model2_sections,
        &model1_sections,
        token_count,
    )?;

    let token_to_id = build_token_index(id_to_token.as_slice())?;
    let (starts, model3, model2, model1) = decode_models(
        start_records.as_slice(),
        &model3_sections,
        &model2_sections,
        &model1_sections,
        model3_keys.as_slice(),
    )?;

    Ok(MarkovChain {
        token_to_id,
        id_to_token,
        model3,
        model2,
        model1,
        starts,
    })
}

fn validate_storage(
    starts: &[StartRecord],
    model3: &Model3Sections,
    model2: &Model2Sections,
    model1: &Model1Sections,
    token_count: u32,
) -> Result<Vec<[TokenId; 3]>, DynError> {
    let model3_keys = validate::validate_and_build_model3_keys(model3, token_count)?;
    validate::validate_model2(model2, token_count)?;
    validate::validate_model1(model1, token_count)?;
    validate::validate_starts(starts, model3_keys.len())?;

    Ok(model3_keys)
}

fn decode_models(
    starts: &[StartRecord],
    model3: &Model3Sections,
    model2: &Model2Sections,
    model1: &Model1Sections,
    model3_keys: &[[TokenId; 3]],
) -> Result<DecodedModels, DynError> {
    let starts = decode::decode_starts(starts, model3_keys)?;
    let model3 = decode::decode_model3(model3, model3_keys)?;
    let model2 = decode::decode_model2(model2)?;
    let model1 = decode::decode_model1(model1)?;

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
