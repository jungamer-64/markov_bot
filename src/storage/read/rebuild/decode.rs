use std::collections::HashMap;

use super::super::super::{
    Count, DynError, EdgeRecord, Model1Sections, Model2Sections, Model3Sections, StartRecord,
    TokenId, usize_from_u32,
};

pub(super) fn decode_starts(
    starts: &[StartRecord],
    model3_keys: &[[TokenId; 3]],
) -> Result<HashMap<[TokenId; 3], Count>, DynError> {
    let mut decoded = HashMap::new();
    let mut previous = 0_u64;

    for record in starts {
        let delta = record
            .cumulative
            .checked_sub(previous)
            .ok_or("start cumulative underflow")?;
        previous = record.cumulative;

        let prefix_index = usize_from_u32(record.prefix_id, "start prefix_id")?;
        let prefix = *model3_keys
            .get(prefix_index)
            .ok_or("start prefix_id is out of bounds")?;

        let entry = decoded.entry(prefix).or_insert(0_u64);
        *entry = (*entry)
            .checked_add(delta)
            .ok_or("start count overflow while decoding")?;
    }

    Ok(decoded)
}

pub(super) fn decode_model3(
    model: &Model3Sections,
    model3_keys: &[[TokenId; 3]],
) -> Result<HashMap<[TokenId; 3], HashMap<TokenId, Count>>, DynError> {
    let mut decoded = HashMap::new();

    for (index, prefix) in model.prefixes.iter().enumerate() {
        let key = *model3_keys
            .get(index)
            .ok_or("model3 prefix index is out of bounds")?;
        let edge_map = decode_edge_map(model.edges.as_slice(), prefix.edge_start, prefix.edge_len)?;
        decoded.insert(key, edge_map);
    }

    Ok(decoded)
}

pub(super) fn decode_model2(
    model: &Model2Sections,
) -> Result<HashMap<[TokenId; 2], HashMap<TokenId, Count>>, DynError> {
    let mut decoded = HashMap::new();

    for prefix in &model.prefixes {
        let key = [prefix.w1, prefix.w2];
        let edge_map = decode_edge_map(model.edges.as_slice(), prefix.edge_start, prefix.edge_len)?;
        decoded.insert(key, edge_map);
    }

    Ok(decoded)
}

pub(super) fn decode_model1(
    model: &Model1Sections,
) -> Result<HashMap<TokenId, HashMap<TokenId, Count>>, DynError> {
    let mut decoded = HashMap::new();

    for prefix in &model.prefixes {
        let edge_map = decode_edge_map(model.edges.as_slice(), prefix.edge_start, prefix.edge_len)?;
        decoded.insert(prefix.w1, edge_map);
    }

    Ok(decoded)
}

fn decode_edge_map(
    edges: &[EdgeRecord],
    edge_start: u32,
    edge_len: u32,
) -> Result<HashMap<TokenId, Count>, DynError> {
    let start = usize_from_u32(edge_start, "edge_start")?;
    let len = usize_from_u32(edge_len, "edge_len")?;
    let end = start.checked_add(len).ok_or("edge range overflow")?;
    let edge_slice = edges.get(start..end).ok_or("edge range is out of bounds")?;

    let mut map = HashMap::new();
    let mut previous = 0_u64;

    for edge in edge_slice {
        let delta = edge
            .cumulative
            .checked_sub(previous)
            .ok_or("edge cumulative underflow")?;
        previous = edge.cumulative;
        map.insert(edge.next, delta);
    }

    Ok(map)
}
