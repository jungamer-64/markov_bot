use super::super::{
    Count, DynError, MarkovChain, Model1Sections, Model2Sections, Model3Sections, StartRecord,
    StorageSections, VocabSections, validate_special_tokens,
};

mod index;
mod model;
mod vocab;

#[derive(Clone, Copy)]
struct CompilePolicy {
    min_edge_count: Count,
}

type CompiledModels = (
    Vec<StartRecord>,
    Model3Sections,
    Model2Sections,
    Model1Sections,
);

pub(super) fn compile_chain(
    chain: &MarkovChain,
    min_edge_count: Count,
) -> Result<StorageSections, DynError> {
    compile_chain_with_policy(chain, CompilePolicy { min_edge_count })
}

fn compile_chain_with_policy(
    chain: &MarkovChain,
    policy: CompilePolicy,
) -> Result<StorageSections, DynError> {
    if policy.min_edge_count == 0 {
        return Err("min_edge_count must be greater than zero".into());
    }

    validate_special_tokens(chain.id_to_token.as_slice())?;
    index::validate_token_index(chain)?;

    let token_count = super::super::u32_from_usize(chain.id_to_token.len(), "token count")?;
    let (offsets, blob) = vocab::build_vocab(chain.id_to_token.as_slice())?;
    let (starts, model3, model2, model1) = build_models(chain, token_count, policy)?;

    Ok(StorageSections {
        vocab: VocabSections { offsets, blob },
        starts,
        model3,
        model2,
        model1,
    })
}

fn build_models(
    chain: &MarkovChain,
    token_count: u32,
    policy: CompilePolicy,
) -> Result<CompiledModels, DynError> {
    let (model3, prefix_to_id) = model::build_model3(chain, token_count, policy.min_edge_count)?;
    let starts = model::build_starts(chain, &prefix_to_id)?;
    let model2 = model::build_model2(chain, token_count, policy.min_edge_count)?;
    let model1 = model::build_model1(chain, token_count, policy.min_edge_count)?;

    Ok((starts, model3, model2, model1))
}
