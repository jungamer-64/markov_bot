use std::collections::HashMap;

use rand::{Rng, RngExt};

use super::{
    options::{EosPolicy, Temperature},
    token::{Count, EOS_ID, Prefix, TokenId},
};

#[derive(Debug)]
struct AliasTable<K> {
    keys: Vec<K>,
    probabilities: Vec<f64>,
    aliases: Vec<usize>,
}

pub(crate) fn choose_weighted_prefix<R: Rng + ?Sized>(
    starts: &HashMap<Prefix, Count>,
    rng: &mut R,
    temperature: Temperature,
) -> Option<Prefix> {
    let mut entries = starts
        .iter()
        .filter_map(|(prefix, count)| (count.get() > 0).then_some((prefix.clone(), *count)))
        .collect::<Vec<_>>();

    entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    choose_weighted_key(entries, rng, temperature)
}

pub(crate) fn choose_weighted_token<R: Rng + ?Sized>(
    edges: &HashMap<TokenId, Count>,
    rng: &mut R,
    temperature: Temperature,
    policy: EosPolicy,
) -> Option<TokenId> {
    let mut entries = edges
        .iter()
        .filter_map(|(token, count)| {
            if count.get() == 0 || (policy == EosPolicy::Forbidden && *token == EOS_ID) {
                return None;
            }

            Some((*token, *count))
        })
        .collect::<Vec<_>>();

    entries.sort_unstable_by_key(|(token, _)| *token);
    choose_weighted_key(entries, rng, temperature)
}

fn choose_weighted_key<K, R: Rng + ?Sized>(
    entries: Vec<(K, Count)>,
    rng: &mut R,
    temperature: Temperature,
) -> Option<K> {
    if (temperature.get() - Temperature::DEFAULT.get()).abs() <= f64::EPSILON {
        return choose_weighted_key_default(entries, rng);
    }

    choose_weighted_key_with_temperature(entries, rng, temperature)
}

fn choose_weighted_key_default<K, R: Rng + ?Sized>(
    entries: Vec<(K, Count)>,
    rng: &mut R,
) -> Option<K> {
    let weighted_entries = entries
        .into_iter()
        .filter_map(|(key, count)| {
            default_sampling_weight(count).map(|weight| (key, weight))
        })
        .collect::<Vec<_>>();

    let alias_table = AliasTable::build(weighted_entries)?;
    alias_table.sample(rng)
}

fn choose_weighted_key_with_temperature<K, R: Rng + ?Sized>(
    entries: Vec<(K, Count)>,
    rng: &mut R,
    temperature: Temperature,
) -> Option<K> {
    let weighted_entries = build_temperature_weights(entries, temperature)?;
    let alias_table = AliasTable::build(weighted_entries)?;
    alias_table.sample(rng)
}

fn build_temperature_weights<K>(
    entries: Vec<(K, Count)>,
    temperature: Temperature,
) -> Option<Vec<(K, f64)>> {
    let exponent = 1.0_f64 / temperature.get();
    let mut weighted_entries = Vec::with_capacity(entries.len());

    for (key, count) in entries {
        let Some(weight) = scaled_temperature_weight(count, exponent) else {
            continue;
        };

        weighted_entries.push((key, weight));
    }

    (!weighted_entries.is_empty()).then_some(weighted_entries)
}

fn scaled_temperature_weight(count: Count, exponent: f64) -> Option<f64> {
    let scaled = default_sampling_weight(count)?.powf(exponent);
    (scaled.is_finite() && scaled > 0.0).then_some(scaled)
}

fn default_sampling_weight(count: Count) -> Option<f64> {
    if count.get() == 0 {
        return None;
    }

    let bounded = u32::try_from(count.get()).unwrap_or(u32::MAX);
    Some(f64::from(bounded))
}

impl<K> AliasTable<K> {
    fn build(entries: Vec<(K, f64)>) -> Option<Self> {
        let (keys, mut scaled) = normalized_weights(entries)?;

        let table_len = keys.len();
        let mut probabilities = vec![0.0_f64; table_len];
        let mut aliases = (0..table_len).collect::<Vec<_>>();
        let (mut small, mut large) = partition_indices(scaled.as_slice());

        fill_alias_tables(
            probabilities.as_mut_slice(),
            aliases.as_mut_slice(),
            scaled.as_mut_slice(),
            &mut small,
            &mut large,
        );

        finalize_alias_entries(probabilities.as_mut_slice(), aliases.as_mut_slice(), small);
        finalize_alias_entries(probabilities.as_mut_slice(), aliases.as_mut_slice(), large);

        Some(Self {
            keys,
            probabilities,
            aliases,
        })
    }

    fn sample<R: Rng + ?Sized>(self, rng: &mut R) -> Option<K> {
        if self.keys.is_empty() {
            return None;
        }

        let keys_len = self.keys.len();
        if keys_len == 1 {
            return self.keys.into_iter().next();
        }

        let column = rng.random_range(0..keys_len);
        let threshold = rng.random_range(0.0_f64..1.0_f64);
        let probability = *self.probabilities.get(column)?;
        let picked = if threshold < probability {
            column
        } else {
            *self.aliases.get(column)?
        };

        let mut keys = self.keys;
        if picked < keys.len() {
            Some(keys.swap_remove(picked))
        } else {
            None
        }
    }
}

fn normalized_weights<K>(entries: Vec<(K, f64)>) -> Option<(Vec<K>, Vec<f64>)> {
    let (keys, mut weights, total_weight) = collect_positive_weights(entries);
    if keys.is_empty() || total_weight <= 0.0 {
        return None;
    }

    let table_len_u32 = u32::try_from(keys.len()).ok()?;
    let normalization = f64::from(table_len_u32) / total_weight;
    for weight in &mut weights {
        *weight *= normalization;
    }

    Some((keys, weights))
}

fn collect_positive_weights<K>(entries: Vec<(K, f64)>) -> (Vec<K>, Vec<f64>, f64) {
    let mut keys = Vec::with_capacity(entries.len());
    let mut weights = Vec::with_capacity(entries.len());
    let mut total_weight = 0.0_f64;

    for (key, weight) in entries {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }

        keys.push(key);
        weights.push(weight);
        total_weight += weight;
    }

    (keys, weights, total_weight)
}

fn partition_indices(values: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut small = Vec::new();
    let mut large = Vec::new();

    for (index, value) in values.iter().enumerate() {
        if *value < 1.0 {
            small.push(index);
        } else {
            large.push(index);
        }
    }

    (small, large)
}

fn fill_alias_tables(
    probabilities: &mut [f64],
    aliases: &mut [usize],
    scaled: &mut [f64],
    small: &mut Vec<usize>,
    large: &mut Vec<usize>,
) {
    while let (Some(small_index), Some(large_index)) = (small.pop(), large.pop()) {
        let Some(small_scaled) = scaled.get(small_index).copied() else {
            continue;
        };
        let Some(probability) = probabilities.get_mut(small_index) else {
            continue;
        };
        *probability = small_scaled.clamp(0.0, 1.0);
        let assigned_probability = *probability;

        let Some(alias) = aliases.get_mut(small_index) else {
            continue;
        };
        *alias = large_index;

        let Some(large_scaled) = scaled.get(large_index).copied() else {
            continue;
        };

        let updated = large_scaled + assigned_probability - 1.0;
        let Some(slot) = scaled.get_mut(large_index) else {
            continue;
        };
        *slot = updated;

        if updated < 1.0 {
            small.push(large_index);
        } else {
            large.push(large_index);
        }
    }
}

fn finalize_alias_entries(probabilities: &mut [f64], aliases: &mut [usize], pending: Vec<usize>) {
    for index in pending {
        let Some(probability) = probabilities.get_mut(index) else {
            continue;
        };
        *probability = 1.0;

        let Some(alias) = aliases.get_mut(index) else {
            continue;
        };
        *alias = index;
    }
}
