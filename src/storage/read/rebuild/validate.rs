use super::super::super::{
    DynError, EdgeRecord, Model1Sections, Model2Sections, Model3Sections, StartRecord, TokenId,
    usize_from_u32, validate_token_id,
};

pub(super) fn validate_and_build_model3_keys(
    model: &Model3Sections,
    token_count: u32,
) -> Result<Vec<[TokenId; 3]>, DynError> {
    let mut full_prefixes = vec![[0_u32; 3]; model.prefixes.len()];
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_pair = None;

    for pair in &model.pairs {
        validate_token_id(pair.w1, token_count, "model3 pair.w1")?;
        validate_token_id(pair.w2, token_count, "model3 pair.w2")?;

        let current_pair = (pair.w1, pair.w2);
        if let Some(previous) = previous_pair
            && current_pair <= previous
        {
            return Err("model3 pair records are not strictly sorted".into());
        }
        previous_pair = Some(current_pair);

        let prefix_start = usize_from_u32(pair.prefix_start, "model3 prefix start")?;
        let prefix_len = usize_from_u32(pair.prefix_len, "model3 prefix len")?;
        let prefix_end = prefix_start
            .checked_add(prefix_len)
            .ok_or("model3 prefix range overflow")?;

        if prefix_end > model.prefixes.len() {
            return Err("model3 pair prefix range is out of bounds".into());
        }

        let mut previous_w3 = None;
        for index in prefix_start..prefix_end {
            if assigned[index] {
                return Err("model3 pair prefix ranges overlap".into());
            }

            let prefix = model.prefixes[index];
            validate_token_id(prefix.w3, token_count, "model3 prefix.w3")?;

            if let Some(previous) = previous_w3
                && prefix.w3 <= previous
            {
                return Err("model3 prefix records are not sorted by w3".into());
            }
            previous_w3 = Some(prefix.w3);

            validate_prefix_edges(
                model.edges.as_slice(),
                prefix.edge_start,
                prefix.edge_len,
                prefix.total,
                token_count,
                "model3 prefix",
            )?;

            full_prefixes[index] = [pair.w1, pair.w2, prefix.w3];
            assigned[index] = true;
        }
    }

    if assigned.iter().any(|is_assigned| !*is_assigned) {
        return Err("some model3 prefixes are not covered by pair records".into());
    }

    Ok(full_prefixes)
}

pub(super) fn validate_model2(model: &Model2Sections, token_count: u32) -> Result<(), DynError> {
    let mut assigned = vec![false; model.prefixes.len()];
    let mut previous_w1 = None;

    for pair in &model.pairs {
        validate_model2_pair_order(pair.w1, token_count, &mut previous_w1)?;

        let (start, end) =
            model2_pair_range(pair.prefix_start, pair.prefix_len, model.prefixes.len())?;
        validate_model2_pair_group(pair.w1, model, token_count, start, end, &mut assigned)?;
    }

    if assigned.iter().any(|is_assigned| !*is_assigned) {
        return Err("some model2 prefixes are not covered by pair records".into());
    }

    Ok(())
}

pub(super) fn validate_model1(model: &Model1Sections, token_count: u32) -> Result<(), DynError> {
    let mut previous_w1 = None;

    for prefix in &model.prefixes {
        validate_token_id(prefix.w1, token_count, "model1 prefix.w1")?;

        if let Some(previous) = previous_w1
            && prefix.w1 <= previous
        {
            return Err("model1 prefix records are not strictly sorted".into());
        }
        previous_w1 = Some(prefix.w1);

        validate_prefix_edges(
            model.edges.as_slice(),
            prefix.edge_start,
            prefix.edge_len,
            prefix.total,
            token_count,
            "model1 prefix",
        )?;
    }

    Ok(())
}

pub(super) fn validate_starts(
    starts: &[StartRecord],
    model3_prefix_count: usize,
) -> Result<(), DynError> {
    let mut previous_cumulative = 0_u64;
    let mut seen = vec![false; model3_prefix_count];

    for record in starts {
        let prefix_id = usize_from_u32(record.prefix_id, "start prefix_id")?;
        if prefix_id >= model3_prefix_count {
            return Err("start prefix_id is out of range".into());
        }
        if seen[prefix_id] {
            return Err("duplicate start prefix_id is not allowed".into());
        }
        seen[prefix_id] = true;

        if record.cumulative <= previous_cumulative {
            return Err("start cumulative must be strictly increasing".into());
        }

        previous_cumulative = record.cumulative;
    }

    Ok(())
}

fn validate_model2_pair_order(
    w1: u32,
    token_count: u32,
    previous_w1: &mut Option<u32>,
) -> Result<(), DynError> {
    validate_token_id(w1, token_count, "model2 pair.w1")?;

    if let Some(previous) = *previous_w1
        && w1 <= previous
    {
        return Err("model2 pair records are not strictly sorted".into());
    }
    *previous_w1 = Some(w1);

    Ok(())
}

fn model2_pair_range(
    prefix_start: u32,
    prefix_len: u32,
    prefix_count: usize,
) -> Result<(usize, usize), DynError> {
    let start = usize_from_u32(prefix_start, "model2 prefix start")?;
    let len = usize_from_u32(prefix_len, "model2 prefix len")?;
    if len == 0 {
        return Err("model2 pair prefix_len must be greater than zero".into());
    }

    let end = start
        .checked_add(len)
        .ok_or("model2 prefix range overflow")?;
    if end > prefix_count {
        return Err("model2 pair prefix range is out of bounds".into());
    }

    Ok((start, end))
}

fn validate_model2_pair_group(
    w1: u32,
    model: &Model2Sections,
    token_count: u32,
    start: usize,
    end: usize,
    assigned: &mut [bool],
) -> Result<(), DynError> {
    let mut previous_w2 = None;

    for (index, is_assigned) in assigned.iter_mut().enumerate().take(end).skip(start) {
        if *is_assigned {
            return Err("model2 pair prefix ranges overlap".into());
        }
        *is_assigned = true;

        let prefix = model.prefixes[index];
        if prefix.w1 != w1 {
            return Err("model2 prefix.w1 does not match model2 pair.w1".into());
        }

        validate_token_id(prefix.w2, token_count, "model2 prefix.w2")?;
        if let Some(previous) = previous_w2
            && prefix.w2 <= previous
        {
            return Err("model2 prefix records are not sorted by w2 within pair group".into());
        }
        previous_w2 = Some(prefix.w2);

        validate_prefix_edges(
            model.edges.as_slice(),
            prefix.edge_start,
            prefix.edge_len,
            prefix.total,
            token_count,
            "model2 prefix",
        )?;
    }

    Ok(())
}

fn validate_prefix_edges(
    edges: &[EdgeRecord],
    edge_start: u32,
    edge_len: u32,
    total: u64,
    token_count: u32,
    context: &str,
) -> Result<(), DynError> {
    let start = usize_from_u32(edge_start, "edge_start")?;
    let len = usize_from_u32(edge_len, "edge_len")?;
    let end = start.checked_add(len).ok_or("edge range overflow")?;

    if end > edges.len() {
        return Err(format!("{context} edge range is out of bounds").into());
    }

    if edge_len == 0 {
        if total != 0 {
            return Err(format!("{context} total must be zero when edge_len is zero").into());
        }

        return Ok(());
    }

    let edge_slice = &edges[start..end];
    let mut previous_next = None;
    let mut previous_cumulative = 0_u64;

    for edge in edge_slice {
        validate_token_id(edge.next, token_count, context)?;

        if let Some(previous) = previous_next
            && edge.next <= previous
        {
            return Err(format!("{context} edges are not sorted by next").into());
        }
        previous_next = Some(edge.next);

        if edge.cumulative <= previous_cumulative {
            return Err(format!("{context} cumulative must be strictly increasing").into());
        }
        previous_cumulative = edge.cumulative;
    }

    if previous_cumulative != total {
        return Err(format!("{context} total does not match last cumulative").into());
    }

    Ok(())
}
