use std::collections::HashMap;

use super::super::{
    CompiledStorage, Count, DynError, EdgeRecord, MarkovChain, Model3Build, Pair3Record,
    Prefix1Record, Prefix2Record, Prefix3Record, StartRecord, TokenId, checked_add, u32_from_usize,
    u64_from_usize, validate_special_tokens, validate_token_id,
};

pub(super) fn compile_chain(chain: &MarkovChain) -> Result<CompiledStorage, DynError> {
    validate_special_tokens(chain.id_to_token.as_slice())?;
    validate_token_index(chain)?;

    let token_count = u32_from_usize(chain.id_to_token.len(), "token count")?;
    let (vocab_offsets, vocab_blob) = build_vocab(chain.id_to_token.as_slice())?;

    let (model3_pairs, model3_prefixes, model3_edges, prefix_to_id) =
        build_model3(chain, token_count)?;
    let starts = build_starts(chain, &prefix_to_id)?;
    let (model2_prefixes, model2_edges) = build_model2(chain, token_count)?;
    let (model1_prefixes, model1_edges) = build_model1(chain, token_count)?;

    Ok(CompiledStorage {
        vocab_offsets,
        vocab_blob,
        starts,
        model3_pairs,
        model3_prefixes,
        model3_edges,
        model2_prefixes,
        model2_edges,
        model1_prefixes,
        model1_edges,
    })
}

fn validate_token_index(chain: &MarkovChain) -> Result<(), DynError> {
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

fn build_vocab(tokens: &[String]) -> Result<(Vec<u64>, Vec<u8>), DynError> {
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

fn build_model3(chain: &MarkovChain, token_count: u32) -> Result<Model3Build, DynError> {
    let mut entries = chain
        .model3
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();
    let mut prefix_to_id = HashMap::new();

    let mut index = 0_usize;
    while index < entries.len() {
        let [w1, w2, _] = entries[index].0;
        validate_token_id(w1, token_count, "model3 pair.w1")?;
        validate_token_id(w2, token_count, "model3 pair.w2")?;

        let prefix_start = u32_from_usize(prefix_records.len(), "model3 prefix start")?;
        let mut prefix_len = 0_u32;

        while index < entries.len() && entries[index].0[0] == w1 && entries[index].0[1] == w2 {
            let prefix = entries[index].0;
            let w3 = prefix[2];
            validate_token_id(w3, token_count, "model3 prefix.w3")?;

            let prefix_id = u32_from_usize(prefix_records.len(), "model3 prefix id")?;
            prefix_to_id.insert(prefix, prefix_id);

            let (edge_start, edge_len, total) = append_edges(
                entries[index].1,
                &mut edge_records,
                token_count,
                "model3 edges",
            )?;

            prefix_records.push(Prefix3Record {
                w3,
                edge_start,
                edge_len,
                total,
            });

            prefix_len = prefix_len
                .checked_add(1)
                .ok_or("model3 prefix length overflow")?;
            index += 1;
        }

        pair_records.push(Pair3Record {
            w1,
            w2,
            prefix_start,
            prefix_len,
        });
    }

    Ok((pair_records, prefix_records, edge_records, prefix_to_id))
}

fn build_model2(
    chain: &MarkovChain,
    token_count: u32,
) -> Result<(Vec<Prefix2Record>, Vec<EdgeRecord>), DynError> {
    let mut entries = chain
        .model2
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for (prefix, edges) in entries {
        validate_token_id(prefix[0], token_count, "model2 prefix.w1")?;
        validate_token_id(prefix[1], token_count, "model2 prefix.w2")?;

        let (edge_start, edge_len, total) =
            append_edges(edges, &mut edge_records, token_count, "model2 edges")?;

        prefix_records.push(Prefix2Record {
            w1: prefix[0],
            w2: prefix[1],
            edge_start,
            edge_len,
            total,
        });
    }

    Ok((prefix_records, edge_records))
}

fn build_model1(
    chain: &MarkovChain,
    token_count: u32,
) -> Result<(Vec<Prefix1Record>, Vec<EdgeRecord>), DynError> {
    let mut entries = chain
        .model1
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for (prefix, edges) in entries {
        validate_token_id(prefix, token_count, "model1 prefix.w1")?;

        let (edge_start, edge_len, total) =
            append_edges(edges, &mut edge_records, token_count, "model1 edges")?;

        prefix_records.push(Prefix1Record {
            w1: prefix,
            edge_start,
            edge_len,
            total,
        });
    }

    Ok((prefix_records, edge_records))
}

fn build_starts(
    chain: &MarkovChain,
    prefix_to_id: &HashMap<[TokenId; 3], u32>,
) -> Result<Vec<StartRecord>, DynError> {
    let mut entries = chain
        .starts
        .iter()
        .map(|(prefix, count)| (*prefix, *count))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut records = Vec::new();
    let mut cumulative = 0_u32;

    for (prefix, count) in entries {
        if count == 0 {
            continue;
        }

        let Some(prefix_id) = prefix_to_id.get(&prefix).copied() else {
            return Err("start prefix is missing from model3 prefixes".into());
        };

        let count = u32::try_from(count).map_err(|_| "start count exceeds u32 range")?;
        cumulative = cumulative
            .checked_add(count)
            .ok_or("start cumulative overflow")?;

        records.push(StartRecord {
            prefix_id,
            cumulative,
        });
    }

    Ok(records)
}

fn append_edges(
    source: &HashMap<TokenId, Count>,
    edges: &mut Vec<EdgeRecord>,
    token_count: u32,
    context: &str,
) -> Result<(u32, u32, u32), DynError> {
    let edge_start = u32_from_usize(edges.len(), "edge start")?;

    let mut sorted_edges = source
        .iter()
        .map(|(next, count)| (*next, *count))
        .collect::<Vec<_>>();
    sorted_edges.sort_unstable_by_key(|(next, _)| *next);

    let mut cumulative = 0_u32;

    for (next, count) in sorted_edges {
        if count == 0 {
            continue;
        }

        validate_token_id(next, token_count, context)?;

        let weight =
            u32::try_from(count).map_err(|_| format!("{context} count exceeds u32 range"))?;
        cumulative = cumulative
            .checked_add(weight)
            .ok_or_else(|| format!("{context} cumulative overflow"))?;

        edges.push(EdgeRecord { next, cumulative });
    }

    let edge_end = u32_from_usize(edges.len(), "edge end")?;
    let edge_len = edge_end
        .checked_sub(edge_start)
        .ok_or("edge length underflow")?;

    Ok((edge_start, edge_len, cumulative))
}
