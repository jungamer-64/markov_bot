use super::{CompiledStorage, DynError, MarkovChain};

mod compile;
mod encode;

pub(super) fn compile_chain(chain: &MarkovChain) -> Result<CompiledStorage, DynError> {
    compile::compile_chain(chain)
}

pub(super) fn encode_storage(compiled: &CompiledStorage) -> Result<Vec<u8>, DynError> {
    encode::encode_storage(compiled)
}
