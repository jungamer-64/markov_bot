use super::{CompiledStorage, Count, DynError, MarkovChain};

mod compile;
mod encode;

pub(super) fn compile_chain(
    chain: &MarkovChain,
    min_edge_count: Count,
) -> Result<CompiledStorage, DynError> {
    compile::compile_chain(chain, min_edge_count)
}

pub(super) fn encode_storage(compiled: &CompiledStorage) -> Result<Vec<u8>, DynError> {
    encode::encode_storage(compiled)
}
