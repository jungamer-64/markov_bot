use std::collections::HashMap;

use super::super::super::{
    Count, DynError, EdgeRecord, MarkovChain, Model1Sections, Model2Sections, Model3Sections,
    Model4Sections, Model5Sections, Model6PrefixIndex, Model6Sections, Pair2Record, Pair3Record,
    Pair4Record, Pair5Record, Pair6Record, Prefix1Record, Prefix2Record, Prefix3Record,
    Prefix4Record, Prefix5Record, Prefix6Record, StartRecord, TokenId, u32_from_usize,
    validate_token_id,
};

type Model6Entry<'a> = ([TokenId; 6], &'a HashMap<TokenId, Count>);
type Model5Entry<'a> = ([TokenId; 5], &'a HashMap<TokenId, Count>);
type Model4Entry<'a> = ([TokenId; 4], &'a HashMap<TokenId, Count>);
type Model3Entry<'a> = ([TokenId; 3], &'a HashMap<TokenId, Count>);

pub(super) fn build_model6(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<(Model6Sections, Model6PrefixIndex), DynError> {
    let mut entries = chain
        .model6
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<Model6Entry<'_>>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();
    let mut prefix_to_id = Model6PrefixIndex::new();

    for group in entries.chunk_by(|left, right| left.0[..5] == right.0[..5]) {
        let (w1, w2, w3, w4, w5) = validate_model6_pair_group(group, token_count)?;
        let (prefix_start, prefix_len) = append_model6_group(
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

        pair_records.push(Pair6Record {
            w1,
            w2,
            w3,
            w4,
            w5,
            prefix_start,
            prefix_len,
        });
    }

    Ok((
        Model6Sections {
            pairs: pair_records,
            prefixes: prefix_records,
            edges: edge_records,
        },
        prefix_to_id,
    ))
}

pub(super) fn build_model5(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Model5Sections, DynError> {
    let mut entries = chain
        .model5
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<Model5Entry<'_>>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for group in entries.chunk_by(|left, right| left.0[..4] == right.0[..4]) {
        let (w1, w2, w3, w4) = validate_model5_pair_group(group, token_count)?;
        let (prefix_start, prefix_len) = append_model5_group(
            group,
            token_count,
            min_edge_count,
            &mut prefix_records,
            &mut edge_records,
        )?;

        if prefix_len == 0 {
            continue;
        }

        pair_records.push(Pair5Record {
            w1,
            w2,
            w3,
            w4,
            prefix_start,
            prefix_len,
        });
    }

    Ok(Model5Sections {
        pairs: pair_records,
        prefixes: prefix_records,
        edges: edge_records,
    })
}

pub(super) fn build_model4(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Model4Sections, DynError> {
    let mut entries = chain
        .model4
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<Model4Entry<'_>>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for group in entries.chunk_by(|left, right| left.0[..3] == right.0[..3]) {
        let (w1, w2, w3) = validate_model4_pair_group(group, token_count)?;
        let (prefix_start, prefix_len) = append_model4_group(
            group,
            token_count,
            min_edge_count,
            &mut prefix_records,
            &mut edge_records,
        )?;

        if prefix_len == 0 {
            continue;
        }

        pair_records.push(Pair4Record {
            w1,
            w2,
            w3,
            prefix_start,
            prefix_len,
        });
    }

    Ok(Model4Sections {
        pairs: pair_records,
        prefixes: prefix_records,
        edges: edge_records,
    })
}

pub(super) fn build_model3(
    chain: &MarkovChain,
    token_count: u32,
    min_edge_count: Count,
) -> Result<Model3Sections, DynError> {
    let mut entries = chain
        .model3
        .iter()
        .map(|(prefix, edges)| (*prefix, edges))
        .collect::<Vec<Model3Entry<'_>>>();
    entries.sort_unstable_by_key(|(prefix, _)| *prefix);

    let mut pair_records = Vec::new();
    let mut prefix_records = Vec::new();
    let mut edge_records = Vec::new();

    for group in entries.chunk_by(|left, right| left.0[..2] == right.0[..2]) {
        let (w1, w2) = validate_model3_pair_group(group, token_count)?;
        let (prefix_start, prefix_len) = append_model3_group(
            group,
            token_count,
            min_edge_count,
            &mut prefix_records,
            &mut edge_records,
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

    Ok(Model3Sections {
        pairs: pair_records,
        prefixes: prefix_records,
        edges: edge_records,
    })
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
        let (first_prefix, _) = group
            .first()
            .copied()
            .ok_or("model2 pair group must contain at least one prefix")?;
        let [w1, _w2] = first_prefix;
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
    prefix_to_id: &Model6PrefixIndex,
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
            // NOTE: model6 prefix-level pruning may remove a start prefix entirely.
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

fn validate_model6_pair_group(
    group: &[Model6Entry<'_>],
    token_count: u32,
) -> Result<(u32, u32, u32, u32, u32), DynError> {
    let (prefix, _) = group
        .first()
        .copied()
        .ok_or("model6 pair group must contain at least one prefix")?;
    let [w1, w2, w3, w4, w5, _w6] = prefix;
    validate_token_id(w1, token_count, "model6 pair.w1")?;
    validate_token_id(w2, token_count, "model6 pair.w2")?;
    validate_token_id(w3, token_count, "model6 pair.w3")?;
    validate_token_id(w4, token_count, "model6 pair.w4")?;
    validate_token_id(w5, token_count, "model6 pair.w5")?;
    Ok((w1, w2, w3, w4, w5))
}

fn append_model6_group(
    group: &[Model6Entry<'_>],
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix6Record>,
    edge_records: &mut Vec<EdgeRecord>,
    prefix_to_id: &mut Model6PrefixIndex,
) -> Result<(u32, u32), DynError> {
    let prefix_start = u32_from_usize(prefix_records.len(), "model6 prefix start")?;
    let mut retained_prefix_len = 0_u32;

    for (prefix, source_edges) in group.iter().copied() {
        let appended = append_model6_prefix(
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
                .ok_or("model6 prefix length overflow")?;
        }
    }

    Ok((prefix_start, retained_prefix_len))
}

fn append_model6_prefix(
    prefix: [TokenId; 6],
    source_edges: &HashMap<TokenId, Count>,
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix6Record>,
    edge_records: &mut Vec<EdgeRecord>,
    prefix_to_id: &mut Model6PrefixIndex,
) -> Result<bool, DynError> {
    let w6 = prefix[5];
    validate_token_id(w6, token_count, "model6 prefix.w6")?;

    let (edge_start, edge_len, total) = append_edges(
        source_edges,
        edge_records,
        token_count,
        min_edge_count,
        "model6 edges",
    )?;

    if edge_len == 0 {
        return Ok(false);
    }

    let prefix_id = u32_from_usize(prefix_records.len(), "model6 prefix id")?;
    prefix_to_id.insert(prefix, prefix_id);

    prefix_records.push(Prefix6Record {
        w6,
        edge_start,
        edge_len,
        total,
    });

    Ok(true)
}

fn validate_model5_pair_group(
    group: &[Model5Entry<'_>],
    token_count: u32,
) -> Result<(u32, u32, u32, u32), DynError> {
    let (prefix, _) = group
        .first()
        .copied()
        .ok_or("model5 pair group must contain at least one prefix")?;
    let [w1, w2, w3, w4, _w5] = prefix;
    validate_token_id(w1, token_count, "model5 pair.w1")?;
    validate_token_id(w2, token_count, "model5 pair.w2")?;
    validate_token_id(w3, token_count, "model5 pair.w3")?;
    validate_token_id(w4, token_count, "model5 pair.w4")?;
    Ok((w1, w2, w3, w4))
}

fn append_model5_group(
    group: &[Model5Entry<'_>],
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix5Record>,
    edge_records: &mut Vec<EdgeRecord>,
) -> Result<(u32, u32), DynError> {
    let prefix_start = u32_from_usize(prefix_records.len(), "model5 prefix start")?;
    let mut retained_prefix_len = 0_u32;

    for (prefix, source_edges) in group.iter().copied() {
        let appended = append_model5_prefix(
            prefix,
            source_edges,
            token_count,
            min_edge_count,
            prefix_records,
            edge_records,
        )?;

        if appended {
            retained_prefix_len = retained_prefix_len
                .checked_add(1)
                .ok_or("model5 prefix length overflow")?;
        }
    }

    Ok((prefix_start, retained_prefix_len))
}

fn append_model5_prefix(
    prefix: [TokenId; 5],
    source_edges: &HashMap<TokenId, Count>,
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix5Record>,
    edge_records: &mut Vec<EdgeRecord>,
) -> Result<bool, DynError> {
    let w5 = prefix[4];
    validate_token_id(w5, token_count, "model5 prefix.w5")?;

    let (edge_start, edge_len, total) = append_edges(
        source_edges,
        edge_records,
        token_count,
        min_edge_count,
        "model5 edges",
    )?;

    if edge_len == 0 {
        return Ok(false);
    }

    prefix_records.push(Prefix5Record {
        w5,
        edge_start,
        edge_len,
        total,
    });

    Ok(true)
}

fn validate_model4_pair_group(
    group: &[Model4Entry<'_>],
    token_count: u32,
) -> Result<(u32, u32, u32), DynError> {
    let (prefix, _) = group
        .first()
        .copied()
        .ok_or("model4 pair group must contain at least one prefix")?;
    let [w1, w2, w3, _w4] = prefix;
    validate_token_id(w1, token_count, "model4 pair.w1")?;
    validate_token_id(w2, token_count, "model4 pair.w2")?;
    validate_token_id(w3, token_count, "model4 pair.w3")?;
    Ok((w1, w2, w3))
}

fn append_model4_group(
    group: &[Model4Entry<'_>],
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix4Record>,
    edge_records: &mut Vec<EdgeRecord>,
) -> Result<(u32, u32), DynError> {
    let prefix_start = u32_from_usize(prefix_records.len(), "model4 prefix start")?;
    let mut retained_prefix_len = 0_u32;

    for (prefix, source_edges) in group.iter().copied() {
        let appended = append_model4_prefix(
            prefix,
            source_edges,
            token_count,
            min_edge_count,
            prefix_records,
            edge_records,
        )?;

        if appended {
            retained_prefix_len = retained_prefix_len
                .checked_add(1)
                .ok_or("model4 prefix length overflow")?;
        }
    }

    Ok((prefix_start, retained_prefix_len))
}

fn append_model4_prefix(
    prefix: [TokenId; 4],
    source_edges: &HashMap<TokenId, Count>,
    token_count: u32,
    min_edge_count: Count,
    prefix_records: &mut Vec<Prefix4Record>,
    edge_records: &mut Vec<EdgeRecord>,
) -> Result<bool, DynError> {
    let w4 = prefix[3];
    validate_token_id(w4, token_count, "model4 prefix.w4")?;

    let (edge_start, edge_len, total) = append_edges(
        source_edges,
        edge_records,
        token_count,
        min_edge_count,
        "model4 edges",
    )?;

    if edge_len == 0 {
        return Ok(false);
    }

    prefix_records.push(Prefix4Record {
        w4,
        edge_start,
        edge_len,
        total,
    });

    Ok(true)
}

fn validate_model3_pair_group(
    group: &[Model3Entry<'_>],
    token_count: u32,
) -> Result<(u32, u32), DynError> {
    let (prefix, _) = group
        .first()
        .copied()
        .ok_or("model3 pair group must contain at least one prefix")?;
    let [w1, w2, _w3] = prefix;
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

    use crate::config::DynError;
    use crate::markov::MarkovChain;
    use crate::test_support::{ensure, ensure_eq};

    use super::{
        EdgeRecord, append_edges, build_model2, build_model3, build_model5, build_model6,
        build_starts, sorted_pruned_edges,
    };

    #[test]
    fn prunes_edges_below_min_count() -> Result<(), DynError> {
        let mut source = HashMap::new();
        source.insert(3_u32, 1_u64);
        source.insert(1_u32, 4_u64);
        source.insert(2_u32, 2_u64);

        let filtered = sorted_pruned_edges(&source, 2);

        ensure_eq(
            &filtered,
            &vec![(1_u32, 4_u64), (2_u32, 2_u64)],
            "edges below min count should be pruned and the rest sorted",
        )?;
        Ok(())
    }

    #[test]
    fn append_edges_recalculates_cumulative_after_pruning() -> Result<(), DynError> {
        let mut source = HashMap::new();
        source.insert(10_u32, 1_u64);
        source.insert(20_u32, 3_u64);
        source.insert(30_u32, 2_u64);

        let mut edges = Vec::<EdgeRecord>::new();
        let (_start, len, total) = append_edges(&source, &mut edges, 100, 2, "model edges")?;

        ensure_eq(&len, &2, "two retained edges should remain after pruning")?;
        ensure_eq(&total, &5, "cumulative total should reflect retained edges")?;
        let first = edges.first().ok_or("first retained edge should exist")?;
        let second = edges.get(1).ok_or("second retained edge should exist")?;
        ensure_eq(
            &first.next,
            &20,
            "first retained edge should be sorted by next token",
        )?;
        ensure_eq(
            &first.cumulative,
            &3,
            "first retained edge cumulative count should be recomputed",
        )?;
        ensure_eq(
            &second.next,
            &30,
            "second retained edge should be sorted by next token",
        )?;
        ensure_eq(
            &second.cumulative,
            &5,
            "second retained edge cumulative count should be recomputed",
        )?;
        Ok(())
    }

    #[test]
    fn model6_pair_prefix_len_counts_only_retained_prefixes() -> Result<(), DynError> {
        let mut chain = minimal_chain();

        chain
            .model6
            .insert([2, 3, 4, 5, 6, 7], HashMap::from([(8, 1_u64)]));
        chain
            .model6
            .insert([2, 3, 4, 5, 6, 8], HashMap::from([(9, 2_u64)]));

        let (model6, prefix_to_id) = build_model6(&chain, 10, 2)?;

        ensure_eq(
            &model6.pairs.len(),
            &1,
            "only one model6 pair should remain",
        )?;
        let pair = model6
            .pairs
            .first()
            .ok_or("retained model6 pair should exist")?;
        ensure_eq(
            &pair.prefix_start,
            &0,
            "retained model6 pair should start at prefix 0",
        )?;
        ensure_eq(
            &pair.prefix_len,
            &1,
            "retained model6 pair should expose one prefix",
        )?;
        ensure_eq(
            &model6.prefixes.len(),
            &1,
            "only one model6 prefix should remain",
        )?;
        ensure_eq(
            &prefix_to_id.len(),
            &1,
            "one model6 prefix id should be assigned",
        )?;
        ensure(
            prefix_to_id.contains_key(&[2, 3, 4, 5, 6, 8]),
            "retained model6 prefix must stay addressable",
        )?;
        ensure(
            !prefix_to_id.contains_key(&[2, 3, 4, 5, 6, 7]),
            "pruned model6 prefix must be removed",
        )?;
        Ok(())
    }

    #[test]
    fn model5_pair_prefix_len_counts_only_retained_prefixes() -> Result<(), DynError> {
        let mut chain = minimal_chain();

        chain
            .model5
            .insert([2, 3, 4, 5, 7], HashMap::from([(8, 1_u64)]));
        chain
            .model5
            .insert([2, 3, 4, 5, 8], HashMap::from([(9, 2_u64)]));

        let model5 = build_model5(&chain, 10, 2)?;

        ensure_eq(
            &model5.pairs.len(),
            &1,
            "only one model5 pair should remain",
        )?;
        let pair = model5
            .pairs
            .first()
            .ok_or("retained model5 pair should exist")?;
        ensure_eq(
            &pair.prefix_start,
            &0,
            "retained model5 pair should start at prefix 0",
        )?;
        ensure_eq(
            &pair.prefix_len,
            &1,
            "retained model5 pair should expose one prefix",
        )?;
        ensure_eq(
            &model5.prefixes.len(),
            &1,
            "only one model5 prefix should remain",
        )?;
        let prefix = model5
            .prefixes
            .first()
            .ok_or("retained model5 prefix should exist")?;
        ensure_eq(
            &prefix.w5,
            &8,
            "retained model5 prefix should keep the surviving token",
        )?;
        Ok(())
    }

    #[test]
    fn model3_pair_prefix_len_counts_only_retained_prefixes() -> Result<(), DynError> {
        let mut chain = minimal_chain();

        chain.model3.insert([2, 3, 4], HashMap::from([(5, 1_u64)]));
        chain.model3.insert([2, 3, 5], HashMap::from([(6, 2_u64)]));

        let model3 = build_model3(&chain, 10, 2)?;

        ensure_eq(
            &model3.pairs.len(),
            &1,
            "only one model3 pair should remain",
        )?;
        let pair = model3
            .pairs
            .first()
            .ok_or("retained model3 pair should exist")?;
        ensure_eq(
            &pair.prefix_start,
            &0,
            "retained model3 pair should start at prefix 0",
        )?;
        ensure_eq(
            &pair.prefix_len,
            &1,
            "retained model3 pair should expose one prefix",
        )?;
        ensure_eq(
            &model3.prefixes.len(),
            &1,
            "only one model3 prefix should remain",
        )?;
        let prefix = model3
            .prefixes
            .first()
            .ok_or("retained model3 prefix should exist")?;
        ensure_eq(
            &prefix.w3,
            &5,
            "retained model3 prefix should keep the surviving token",
        )?;
        Ok(())
    }

    #[test]
    fn model2_pair_prefix_len_counts_only_retained_prefixes() -> Result<(), DynError> {
        let mut chain = minimal_chain();

        chain.model2.insert([2, 4], HashMap::from([(6, 1_u64)]));
        chain.model2.insert([2, 5], HashMap::from([(6, 2_u64)]));

        let model2 = build_model2(&chain, 10, 2)?;

        ensure_eq(
            &model2.pairs.len(),
            &1,
            "only one model2 pair should remain",
        )?;
        let pair = model2
            .pairs
            .first()
            .ok_or("retained model2 pair should exist")?;
        ensure_eq(
            &pair.w1,
            &2,
            "retained model2 pair should keep its first token",
        )?;
        ensure_eq(
            &pair.prefix_start,
            &0,
            "retained model2 pair should start at prefix 0",
        )?;
        ensure_eq(
            &pair.prefix_len,
            &1,
            "retained model2 pair should expose one prefix",
        )?;
        ensure_eq(
            &model2.prefixes.len(),
            &1,
            "only one model2 prefix should remain",
        )?;
        let prefix = model2
            .prefixes
            .first()
            .ok_or("retained model2 prefix should exist")?;
        ensure_eq(
            &prefix.w2,
            &5,
            "retained model2 prefix should keep the surviving token",
        )?;
        Ok(())
    }

    #[test]
    fn starts_skip_prefixes_missing_after_pruning() -> Result<(), DynError> {
        let mut chain = minimal_chain();
        chain.starts.insert([2, 3, 4, 5, 6, 7], 3);
        chain.starts.insert([2, 3, 4, 5, 6, 8], 2);

        let prefix_to_id = HashMap::from([([2, 3, 4, 5, 6, 8], 0_u32)]);

        let starts = build_starts(&chain, &prefix_to_id)?;

        ensure_eq(&starts.len(), &1, "only one start record should remain")?;
        let record = starts.first().ok_or("retained start record should exist")?;
        ensure_eq(
            &record.prefix_id,
            &0,
            "remaining start record should target prefix 0",
        )?;
        ensure_eq(
            &record.cumulative,
            &2,
            "remaining start record cumulative count should match retained weight",
        )?;
        Ok(())
    }

    fn minimal_chain() -> MarkovChain {
        let mut chain = MarkovChain::default();
        for (token_id, token) in [
            (2_u32, "t2"),
            (3_u32, "t3"),
            (4_u32, "t4"),
            (5_u32, "t5"),
            (6_u32, "t6"),
            (7_u32, "t7"),
            (8_u32, "t8"),
            (9_u32, "t9"),
        ] {
            chain.token_to_id.insert(token.to_owned(), token_id);
            chain.id_to_token.push(token.to_owned());
        }

        chain
    }
}
