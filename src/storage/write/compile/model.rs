use std::collections::HashMap;

use super::super::super::{
    Count, DynError, EdgeRecord, MarkovChain, Model1Sections, Model2Sections, Model3PrefixIndex,
    Model3Sections, Pair2Record, Pair3Record, Prefix1Record, Prefix2Record, Prefix3Record,
    StartRecord, TokenId, u32_from_usize, validate_token_id,
};

type Model3Entry<'a> = ([TokenId; 3], &'a HashMap<TokenId, Count>);

pub(super) fn build_model3(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<(Model3Sections, Model3PrefixIndex), DynError> {
    let mut entries = chain
        .model3
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<Model3Entry<'_>>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();
    let mut prefix_to_id = Model3PrefixIndex::new();

    for group in entries.chunk_by(|left, right| left.0[0] == right.0[0] && left.0[1] == right.0[1])
    {
        let (w1, w2) = validate_model3_pair_group(group, token_count)?;
        let (prefix_start, prefix_len) = append_model3_group(
            group,
            token_count,
            min_edge_count,
            &mut prefix_records,
            &mut edge_records,
            &mut prefix_to_id,
        )?;

        if prefix_len == 0 {
            continue;
        }

        pair_records.push(Pair3Record {
            w1,
            w2,
            prefix_start,
            prefix_len,
        });
    }

    Ok((
        Model3Sections {
            pairs: pair_records,
            prefixes: prefix_records,
            edges: edge_records,
        },
        prefix_to_id,
    ))
}

pub(super) fn build_model2(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Model2Sections, DynError> {
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
        let mut retained_prefix_len = 0_u32;

        for (prefix, edges) in group.iter().copied() {
            validate_token_id(prefix[1], token_count, "model2 prefix.w2")?;

            let (edge_start, edge_len, total) = append_edges(
                edges,
                &mut edge_records,
                token_count,
                min_edge_count,
                "model2 edges",
            )?;

            if edge_len == 0 {
                continue;
            }

            prefix_records.push(Prefix2Record {
                w1,
                w2: prefix[1],
                edge_start,
                edge_len,
                total,
            });

            retained_prefix_len = retained_prefix_len
                .checked_add(1)
                .ok_or("model2 prefix length overflow")?;
        }

        if retained_prefix_len == 0 {
            continue;
        }

        pair_records.push(Pair2Record {
            w1,
            prefix_start,
            prefix_len: retained_prefix_len,
        });
    }

    Ok(Model2Sections {
        pairs: pair_records,
        prefixes: prefix_records,
        edges: edge_records,
    })
}

pub(super) fn build_model1(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Model1Sections, DynError> {
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

        let (edge_start, edge_len, total) = append_edges(
            edges,
            &mut edge_records,
            token_count,
            min_edge_count,
            "model1 edges",
        )?;

        if edge_len == 0 {
            continue;
        }

        prefix_records.push(Prefix1Record {
            w1: prefix,
            edge_start,
            edge_len,
            total,
        });
    }

    Ok(Model1Sections {
        prefixes: prefix_records,
        edges: edge_records,
    })
}

pub(super) fn build_starts(
    chain: &MarkovChain,
    prefix_to_id: &Model3PrefixIndex,
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
            // NOTE: model3 prefix-level pruning may remove a start prefix entirely.
            // In that case we drop only that start entry and keep remaining starts.
            continue;
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
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix3Record>,
    edge_records: &mut Vec<EdgeRecord>,
    prefix_to_id: &mut Model3PrefixIndex,
) -> Result<(u32, u32), DynError> {
    let prefix_start = u32_from_usize(prefix_records.len(), "model3 prefix start")?;
    let mut retained_prefix_len = 0_u32;

    for (prefix, source_edges) in group.iter().copied() {
        let appended = append_model3_prefix(
            prefix,
            source_edges,
            token_count,
            min_edge_count,
            prefix_records,
            edge_records,
            prefix_to_id,
        )?;

        if appended {
            retained_prefix_len = retained_prefix_len
                .checked_add(1)
                .ok_or("model3 prefix length overflow")?;
        }
    }

    Ok((prefix_start, retained_prefix_len))
}

fn append_model3_prefix(
    prefix: [TokenId; 3],
    source_edges: &HashMap<TokenId, Count>,
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix3Record>,
    edge_records: &mut Vec<EdgeRecord>,
    prefix_to_id: &mut Model3PrefixIndex,
) -> Result<bool, DynError> {
    let w3 = prefix[2];
    validate_token_id(w3, token_count, "model3 prefix.w3")?;

    let (edge_start, edge_len, total) = append_edges(
        source_edges,
        edge_records,
        token_count,
        min_edge_count,
        "model3 edges",
    )?;

    if edge_len == 0 {
        return Ok(false);
    }

    let prefix_id = u32_from_usize(prefix_records.len(), "model3 prefix id")?;
    prefix_to_id.insert(prefix, prefix_id);

    prefix_records.push(Prefix3Record {
        w3,
        edge_start,
        edge_len,
        total,
    });

    Ok(true)
}

fn append_edges(
    source: &HashMap<TokenId, Count>,
    edges: &mut Vec<EdgeRecord>,
    token_count: u32,
    min_edge_count: Count,
    context: &str,
) -> Result<(u32, u32, u64), DynError> {
    let edge_start = u32_from_usize(edges.len(), "edge start")?;
    let sorted_edges = sorted_pruned_edges(source, min_edge_count);
    let cumulative = append_sorted_edges(sorted_edges.as_slice(), edges, token_count, context)?;
    let edge_len = compute_edge_len(edge_start, edges.len())?;

    Ok((edge_start, edge_len, cumulative))
}

fn sorted_pruned_edges(
    source: &HashMap<TokenId, Count>,
    min_edge_count: Count,
) -> Vec<(TokenId, Count)> {
    let mut sorted_edges = source
        .iter()
        .filter(|(_, count)| **count >= min_edge_count)
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::markov::MarkovChain;

    use super::{
        EdgeRecord, append_edges, build_model2, build_model3, build_starts, sorted_pruned_edges,
    };

    #[test]
    fn prunes_edges_below_min_count() {
        let mut source = HashMap::new();
        source.insert(3_u32, 1_u64);
        source.insert(1_u32, 4_u64);
        source.insert(2_u32, 2_u64);

        let filtered = sorted_pruned_edges(&source, 2);

        assert_eq!(filtered, vec![(1_u32, 4_u64), (2_u32, 2_u64)]);
    }

    #[test]
    fn append_edges_recalculates_cumulative_after_pruning() {
        let mut source = HashMap::new();
        source.insert(10_u32, 1_u64);
        source.insert(20_u32, 3_u64);
        source.insert(30_u32, 2_u64);

        let mut edges = Vec::<EdgeRecord>::new();
        let (_start, len, total) =
            append_edges(&source, &mut edges, 100, 2, "model edges").expect("append should work");

        assert_eq!(len, 2);
        assert_eq!(total, 5);
        assert_eq!(edges[0].next, 20);
        assert_eq!(edges[0].cumulative, 3);
        assert_eq!(edges[1].next, 30);
        assert_eq!(edges[1].cumulative, 5);
    }

    #[test]
    fn model3_pair_prefix_len_counts_only_retained_prefixes() {
        let mut chain = minimal_chain();

        chain.model3.insert([2, 3, 4], HashMap::from([(5, 1_u64)]));
        chain.model3.insert([2, 3, 5], HashMap::from([(6, 2_u64)]));

        let (model3, prefix_to_id) =
            build_model3(&chain, 7, 2).expect("build_model3 should succeed");

        assert_eq!(model3.pairs.len(), 1);
        assert_eq!(model3.pairs[0].prefix_start, 0);
        assert_eq!(model3.pairs[0].prefix_len, 1);
        assert_eq!(model3.prefixes.len(), 1);
        assert_eq!(prefix_to_id.len(), 1);
        assert!(prefix_to_id.contains_key(&[2, 3, 5]));
        assert!(!prefix_to_id.contains_key(&[2, 3, 4]));
    }

    #[test]
    fn model2_pair_prefix_len_counts_only_retained_prefixes() {
        let mut chain = minimal_chain();

        chain.model2.insert([2, 4], HashMap::from([(6, 1_u64)]));
        chain.model2.insert([2, 5], HashMap::from([(6, 2_u64)]));

        let model2 = build_model2(&chain, 7, 2).expect("build_model2 should succeed");

        assert_eq!(model2.pairs.len(), 1);
        assert_eq!(model2.pairs[0].w1, 2);
        assert_eq!(model2.pairs[0].prefix_start, 0);
        assert_eq!(model2.pairs[0].prefix_len, 1);
        assert_eq!(model2.prefixes.len(), 1);
        assert_eq!(model2.prefixes[0].w2, 5);
    }

    #[test]
    fn starts_skip_prefixes_missing_after_pruning() {
        let mut chain = minimal_chain();
        chain.starts.insert([2, 3, 4], 3);
        chain.starts.insert([2, 3, 5], 2);

        let prefix_to_id = HashMap::from([([2, 3, 5], 0_u32)]);

        let starts = build_starts(&chain, &prefix_to_id).expect("build_starts should succeed");

        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].prefix_id, 0);
        assert_eq!(starts[0].cumulative, 2);
    }

    fn minimal_chain() -> MarkovChain {
        let mut chain = MarkovChain::default();
        for (token_id, token) in [
            (2_u32, "t2"),
            (3_u32, "t3"),
            (4_u32, "t4"),
            (5_u32, "t5"),
            (6_u32, "t6"),
        ] {
            chain.token_to_id.insert(token.to_owned(), token_id);
            chain.id_to_token.push(token.to_owned());
        }

        chain
    }
}
