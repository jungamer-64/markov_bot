use super::super::super::{
    DynError, EDGE_RECORD_SIZE, EdgeRecord, PAIR3_RECORD_SIZE, PREFIX1_RECORD_SIZE,
    PREFIX2_RECORD_SIZE, PREFIX3_RECORD_SIZE, Pair3Record, Prefix1Record, Prefix2Record,
    Prefix3Record, START_RECORD_SIZE, StartRecord, bytes_for_len, usize_from_u64,
};
use super::super::{read_u32_value, read_u64_value};

pub(super) fn parse_u64_values(
    bytes: &[u8],
    count: usize,
    context: &str,
) -> Result<Vec<u64>, DynError> {
    let expected = bytes_for_len(count, 8, context)?;
    let expected = usize_from_u64(expected, context)?;
    if bytes.len() != expected {
        return Err(format!(
            "{context} length mismatch: expected {expected}, got {}",
            bytes.len()
        )
        .into());
    }

    let mut values = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        values.push(read_u64_value(bytes, &mut cursor)?);
    }

    Ok(values)
}

pub(super) fn parse_start_records(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<StartRecord>, DynError> {
    let expected = bytes_for_len(count, START_RECORD_SIZE, "start records")?;
    let expected = usize_from_u64(expected, "start records")?;
    if bytes.len() != expected {
        return Err("start records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(StartRecord {
            prefix_id: read_u32_value(bytes, &mut cursor)?,
            cumulative: read_u64_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

pub(super) fn parse_pair3_records(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<Pair3Record>, DynError> {
    let expected = bytes_for_len(count, PAIR3_RECORD_SIZE, "model3 pair records")?;
    let expected = usize_from_u64(expected, "model3 pair records")?;
    if bytes.len() != expected {
        return Err("model3 pair records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Pair3Record {
            w1: read_u32_value(bytes, &mut cursor)?,
            w2: read_u32_value(bytes, &mut cursor)?,
            prefix_start: read_u32_value(bytes, &mut cursor)?,
            prefix_len: read_u32_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

pub(super) fn parse_prefix3_records(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<Prefix3Record>, DynError> {
    let expected = bytes_for_len(count, PREFIX3_RECORD_SIZE, "model3 prefix records")?;
    let expected = usize_from_u64(expected, "model3 prefix records")?;
    if bytes.len() != expected {
        return Err("model3 prefix records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Prefix3Record {
            w3: read_u32_value(bytes, &mut cursor)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: read_u64_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

pub(super) fn parse_prefix2_records(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<Prefix2Record>, DynError> {
    let expected = bytes_for_len(count, PREFIX2_RECORD_SIZE, "model2 prefix records")?;
    let expected = usize_from_u64(expected, "model2 prefix records")?;
    if bytes.len() != expected {
        return Err("model2 prefix records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Prefix2Record {
            w1: read_u32_value(bytes, &mut cursor)?,
            w2: read_u32_value(bytes, &mut cursor)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: read_u64_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

pub(super) fn parse_prefix1_records(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<Prefix1Record>, DynError> {
    let expected = bytes_for_len(count, PREFIX1_RECORD_SIZE, "model1 prefix records")?;
    let expected = usize_from_u64(expected, "model1 prefix records")?;
    if bytes.len() != expected {
        return Err("model1 prefix records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(Prefix1Record {
            w1: read_u32_value(bytes, &mut cursor)?,
            edge_start: read_u32_value(bytes, &mut cursor)?,
            edge_len: read_u32_value(bytes, &mut cursor)?,
            total: read_u64_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}

pub(super) fn parse_edge_records(bytes: &[u8], count: usize) -> Result<Vec<EdgeRecord>, DynError> {
    let expected = bytes_for_len(count, EDGE_RECORD_SIZE, "edge records")?;
    let expected = usize_from_u64(expected, "edge records")?;
    if bytes.len() != expected {
        return Err("edge records size mismatch".into());
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for _ in 0..count {
        records.push(EdgeRecord {
            next: read_u32_value(bytes, &mut cursor)?,
            cumulative: read_u64_value(bytes, &mut cursor)?,
        });
    }

    Ok(records)
}
