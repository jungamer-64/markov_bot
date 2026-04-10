use std::collections::HashMap;

use super::super::super::{
    Count, DynError, EdgeRecord, MarkovChain, Model3Build, Pair2Record, Pair3Record,
    Prefix1Record, Prefix2Record, Prefix3Record, StartRecord, TokenId, u32_from_usize,
    validate_token_id,
};

type Model3Entry<'a> = ([TokenId; 3], &'a HashMap<TokenId, Count>);
type Model2Build = (Vec<Pair2Record>, Vec<Prefix2Record>, Vec<EdgeRecord>);

pub(super) fn build_model3(chain: &MarkovChain, token_count: u32) -> Result<Model3Build, DynError> {
    let mut entries = chain
        .model3
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<Model3Entry<'_>>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();
    let mut prefix_to_id = HashMap::new();

    for group in entries.chunk_by(|left, right| left.0[0] == right.0[0] && left.0[1] == right.0[1])
    {
        let (w1, w2) = validate_model3_pair_group(group, token_count)?;
        let (prefix_start, prefix_len) = append_model3_group(
            group,
            token_count,
            &mut prefix_records,
            &mut edge_records,
            &mut prefix_to_id,
        )?;

        pair_records.push(Pair3Record {
            w1,
            w2,
            prefix_start,
            prefix_len,
        });
    }

    Ok((pair_records, prefix_records, edge_records, prefix_to_id))
}

pub(super) fn build_model2(
    chain: &MarkovChain,
    token_count: u32,
) -> Result<Model2Build, DynError> {
    let mut entries = chain
        .model2
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for group in entries.chunk_by(|left, right| left.0[0] == right.0[0]) {
        let w1 = group[0].0[0];
        validate_token_id(w1, token_count, "model2 pair.w1")?;

        let prefix_start = u32_from_usize(prefix_records.len(), "model2 prefix start")?;

        for (prefix, edges) in group.iter().copied() {
            validate_token_id(prefix[1], token_count, "model2 prefix.w2")?;

            let (edge_start, edge_len, total) =
                append_edges(edges, &mut edge_records, token_count, "model2 edges")?;

            prefix_records.push(Prefix2Record {
                w1,
                w2: prefix[1],
                edge_start,
                edge_len,
                total,
            });
        }

        let prefix_len = u32_from_usize(group.len(), "model2 prefix length")?;
        pair_records.push(Pair2Record {
            w1,
            prefix_start,
            prefix_len,
        });
    }

    Ok((pair_records, prefix_records, edge_records))
}

pub(super) fn build_model1(
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

pub(super) fn build_starts(
    chain: &MarkovChain,
    prefix_to_id: &HashMap<[TokenId; 3], u32>,
) -> Result<Vec<StartRecord>, DynError> {
    let mut entries = chain
        .starts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(prefix, count)| (*prefix, *count))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut records = Vec::new();
    let mut cumulative = 0_u64;

    for (prefix, count) in entries {
        let Some(prefix_id) = prefix_to_id.get(&prefix).copied() else {
            return Err("start prefix is missing from model3 prefixes".into());
        };

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

fn validate_model3_pair_group(
    group: &[Model3Entry<'_>],
    token_count: u32,
) -> Result<(u32, u32), DynError> {
    let [w1, w2, _] = group[0].0;
    validate_token_id(w1, token_count, "model3 pair.w1")?;
    validate_token_id(w2, token_count, "model3 pair.w2")?;
    Ok((w1, w2))
}

fn append_model3_group(
    group: &[Model3Entry<'_>],
    token_count: u32,
    prefix_records: &mut Vec<Prefix3Record>,
    edge_records: &mut Vec<EdgeRecord>,
    prefix_to_id: &mut HashMap<[TokenId; 3], u32>,
) -> Result<(u32, u32), DynError> {
    let prefix_start = u32_from_usize(prefix_records.len(), "model3 prefix start")?;

    for (prefix, source_edges) in group.iter().copied() {
        append_model3_prefix(
            prefix,
            source_edges,
            token_count,
            prefix_records,
            edge_records,
            prefix_to_id,
        )?;
    }

    let prefix_len = u32_from_usize(group.len(), "model3 prefix length")?;
    Ok((prefix_start, prefix_len))
}

fn append_model3_prefix(
    prefix: [TokenId; 3],
    source_edges: &HashMap<TokenId, Count>,
    token_count: u32,
    prefix_records: &mut Vec<Prefix3Record>,
    edge_records: &mut Vec<EdgeRecord>,
    prefix_to_id: &mut HashMap<[TokenId; 3], u32>,
) -> Result<(), DynError> {
    let w3 = prefix[2];
    validate_token_id(w3, token_count, "model3 prefix.w3")?;

    let prefix_id = u32_from_usize(prefix_records.len(), "model3 prefix id")?;
    prefix_to_id.insert(prefix, prefix_id);

    let (edge_start, edge_len, total) =
        append_edges(source_edges, edge_records, token_count, "model3 edges")?;

    prefix_records.push(Prefix3Record {
        w3,
        edge_start,
        edge_len,
        total,
    });

    Ok(())
}

fn append_edges(
    source: &HashMap<TokenId, Count>,
    edges: &mut Vec<EdgeRecord>,
    token_count: u32,
    context: &str,
) -> Result<(u32, u32, u64), DynError> {
    let edge_start = u32_from_usize(edges.len(), "edge start")?;
    let sorted_edges = sorted_non_zero_edges(source);
    let cumulative = append_sorted_edges(sorted_edges.as_slice(), edges, token_count, context)?;
    let edge_len = compute_edge_len(edge_start, edges.len())?;

    Ok((edge_start, edge_len, cumulative))
}

fn sorted_non_zero_edges(source: &HashMap<TokenId, Count>) -> Vec<(TokenId, Count)> {
    let mut sorted_edges = source
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(next, count)| (*next, *count))
        .collect::<Vec<_>>();
    sorted_edges.sort_unstable_by_key(|(next, _)| *next);
    sorted_edges
}

fn append_sorted_edges(
    sorted_edges: &[(TokenId, Count)],
    edges: &mut Vec<EdgeRecord>,
    token_count: u32,
    context: &str,
) -> Result<u64, DynError> {
    let mut cumulative = 0_u64;

    for (next, count) in sorted_edges.iter().copied() {
        validate_token_id(next, token_count, context)?;

        cumulative = cumulative
            .checked_add(count)
            .ok_or_else(|| format!("{context} cumulative overflow"))?;

        edges.push(EdgeRecord { next, cumulative });
    }

    Ok(cumulative)
}

fn compute_edge_len(edge_start: u32, edge_count: usize) -> Result<u32, DynError> {
    let edge_end = u32_from_usize(edge_count, "edge end")?;
    let edge_len = edge_end
        .checked_sub(edge_start)
        .ok_or("edge length underflow")?;

    Ok(edge_len)
}
