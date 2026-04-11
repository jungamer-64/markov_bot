use super::{Count, DynError, MarkovChain, StorageCompressionMode, StorageSections};

mod compile;
mod encode;

pub(super) fn compile_chain(
    chain: &MarkovChain,
    min_edge_count: Count,
) -> Result<StorageSections, DynError> {
    compile::compile_chain(chain, min_edge_count)
}

pub(super) fn encode_storage(
    sections: &StorageSections,
    compression_mode: StorageCompressionMode,
) -> Result<Vec<u8>, DynError> {
    encode::encode_storage(sections, compression_mode)
}
