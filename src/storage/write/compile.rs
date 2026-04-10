use super::super::{
    CompiledStorage, DynError, MarkovChain, u32_from_usize, validate_special_tokens,
};

mod index;
mod model;
mod vocab;

type ModelSections = (
    Vec<super::super::Pair3Record>,
    Vec<super::super::Prefix3Record>,
    Vec<super::super::EdgeRecord>,
    Vec<super::super::StartRecord>,
    Vec<super::super::Pair2Record>,
    Vec<super::super::Prefix2Record>,
    Vec<super::super::EdgeRecord>,
    Vec<super::super::Prefix1Record>,
    Vec<super::super::EdgeRecord>,
);

pub(super) fn compile_chain(chain: &MarkovChain) -> Result<CompiledStorage, DynError> {
    validate_special_tokens(chain.id_to_token.as_slice())?;
    index::validate_token_index(chain)?;

    let token_count = u32_from_usize(chain.id_to_token.len(), "token count")?;
    let (vocab_offsets, vocab_blob) = vocab::build_vocab(chain.id_to_token.as_slice())?;
    let (
        model3_pairs,
        model3_prefixes,
        model3_edges,
        starts,
        model2_pairs,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    ) = build_models(chain, token_count)?;

    Ok(CompiledStorage {
        vocab_offsets,
        vocab_blob,
        starts,
        model3_pairs,
        model3_prefixes,
        model3_edges,
        model2_pairs,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    })
}

fn build_models(chain: &MarkovChain, token_count: u32) -> Result<ModelSections, DynError> {
    let (model3_pairs, model3_prefixes, model3_edges, prefix_to_id) =
        model::build_model3(chain, token_count)?;
    let starts = model::build_starts(chain, &prefix_to_id)?;
    let (model2_pairs, model2_prefixes, model2_edges) = model::build_model2(chain, token_count)?;
    let (model1_prefixes, model1_edges) = model::build_model1(chain, token_count)?;

    Ok((
        model3_pairs,
        model3_prefixes,
        model3_edges,
        starts,
        model2_pairs,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    ))
}
