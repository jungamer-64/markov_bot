use std::collections::HashMap;

use crate::markov::{Prefix3, Prefix4, Prefix5, Prefix6};

use super::super::{
    Count, DynError, MarkovChain, Model1Sections, Model2Sections, Model3Sections, Model4Sections,
    Model5Sections, Model6Sections, StartRecord, StorageSections, TokenId, u32_from_usize,
    validate_special_tokens,
};
use super::parse::vocab;

mod decode;
mod validate;

type ValidationKeys = (Vec<Prefix6>, Vec<Prefix5>, Vec<Prefix4>, Vec<Prefix3>);
type DecodedModels = (
    HashMap<Prefix6, Count>,
    HashMap<Prefix6, HashMap<TokenId, Count>>,
    HashMap<Prefix5, HashMap<TokenId, Count>>,
    HashMap<Prefix4, HashMap<TokenId, Count>>,
    HashMap<Prefix3, HashMap<TokenId, Count>>,
    HashMap<[TokenId; 2], HashMap<TokenId, Count>>,
    HashMap<TokenId, HashMap<TokenId, Count>>,
);

pub(super) fn rebuild_chain(sections: StorageSections) -> Result<MarkovChain, DynError> {
    let StorageSections {
        vocab: vocab_sections,
        starts: start_records,
        model6: model6_sections,
        model5: model5_sections,
        model4: model4_sections,
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
    let (model6_keys, model5_keys, model4_keys, model3_keys) = validate_storage(
        start_records.as_slice(),
        &model6_sections,
        &model5_sections,
        &model4_sections,
        &model3_sections,
        &model2_sections,
        &model1_sections,
        token_count,
    )?;

    let token_to_id = build_token_index(id_to_token.as_slice())?;
    let (starts, model6, model5, model4, model3, model2, model1) = decode_models(
        start_records.as_slice(),
        &model6_sections,
        &model5_sections,
        &model4_sections,
        &model3_sections,
        &model2_sections,
        &model1_sections,
        model6_keys.as_slice(),
        model5_keys.as_slice(),
        model4_keys.as_slice(),
        model3_keys.as_slice(),
    )?;

    Ok(MarkovChain {
        token_to_id,
        id_to_token,
        model6,
        model5,
        model4,
        model3,
        model2,
        model1,
        starts,
    })
}

fn validate_storage(
    starts: &[StartRecord],
    model6: &Model6Sections,
    model5: &Model5Sections,
    model4: &Model4Sections,
    model3: &Model3Sections,
    model2: &Model2Sections,
    model1: &Model1Sections,
    token_count: u32,
) -> Result<ValidationKeys, DynError> {
    let model6_keys = validate::validate_and_build_model6_keys(model6, token_count)?;
    let model5_keys = validate::validate_and_build_model5_keys(model5, token_count)?;
    let model4_keys = validate::validate_and_build_model4_keys(model4, token_count)?;
    let model3_keys = validate::validate_and_build_model3_keys(model3, token_count)?;
    validate::validate_model2(model2, token_count)?;
    validate::validate_model1(model1, token_count)?;
    validate::validate_starts(starts, model6_keys.len())?;

    Ok((model6_keys, model5_keys, model4_keys, model3_keys))
}

fn decode_models(
    starts: &[StartRecord],
    model6: &Model6Sections,
    model5: &Model5Sections,
    model4: &Model4Sections,
    model3: &Model3Sections,
    model2: &Model2Sections,
    model1: &Model1Sections,
    model6_keys: &[Prefix6],
    model5_keys: &[Prefix5],
    model4_keys: &[Prefix4],
    model3_keys: &[Prefix3],
) -> Result<DecodedModels, DynError> {
    let starts = decode::decode_starts(starts, model6_keys)?;
    let model6 = decode::decode_model6(model6, model6_keys)?;
    let model5 = decode::decode_model5(model5, model5_keys)?;
    let model4 = decode::decode_model4(model4, model4_keys)?;
    let model3 = decode::decode_model3(model3, model3_keys)?;
    let model2 = decode::decode_model2(model2)?;
    let model1 = decode::decode_model1(model1)?;

    Ok((starts, model6, model5, model4, model3, model2, model1))
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
