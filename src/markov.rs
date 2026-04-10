use std::{collections::HashMap, hash::Hash};

use rand::{Rng, RngExt};

use crate::config::DynError;

pub type TokenId = u32;
pub type Count = u64;

pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

pub const BOS_ID: TokenId = 0;
pub const EOS_ID: TokenId = 1;
pub const DEFAULT_GENERATION_TEMPERATURE: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    pub max_words: usize,
    pub temperature: f64,
    pub min_words_before_eos: usize,
}

impl GenerationOptions {
    pub const fn new(max_words: usize, temperature: f64, min_words_before_eos: usize) -> Self {
        Self {
            max_words,
            temperature,
            min_words_before_eos,
        }
    }

    fn is_valid(self) -> bool {
        self.max_words > 0
            && self.temperature.is_finite()
            && self.temperature > 0.0
            && self.min_words_before_eos <= self.max_words
    }
}

#[derive(Debug, Clone)]
pub struct MarkovChain {
    pub(crate) token_to_id: HashMap<String, TokenId>,
    pub(crate) id_to_token: Vec<String>,
    pub(crate) model3: HashMap<[TokenId; 3], HashMap<TokenId, Count>>,
    pub(crate) model2: HashMap<[TokenId; 2], HashMap<TokenId, Count>>,
    pub(crate) model1: HashMap<TokenId, HashMap<TokenId, Count>>,
    pub(crate) starts: HashMap<[TokenId; 3], Count>,
}

impl Default for MarkovChain {
    fn default() -> Self {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(BOS_TOKEN.to_owned(), BOS_ID);
        token_to_id.insert(EOS_TOKEN.to_owned(), EOS_ID);

        Self {
            token_to_id,
            id_to_token: vec![BOS_TOKEN.to_owned(), EOS_TOKEN.to_owned()],
            model3: HashMap::new(),
            model2: HashMap::new(),
            model1: HashMap::new(),
            starts: HashMap::new(),
        }
    }
}

impl MarkovChain {
    pub fn train_tokens(&mut self, tokens: &[String]) -> Result<(), DynError> {
        if tokens.is_empty() {
            return Ok(());
        }

        let mut sentence = Vec::with_capacity(tokens.len().saturating_add(4));
        sentence.extend([BOS_ID, BOS_ID, BOS_ID]);

        for token in tokens {
            let token_id = self.intern_token(token)?;
            sentence.push(token_id);
        }

        sentence.push(EOS_ID);

        let start_prefix = [sentence[0], sentence[1], sentence[2]];
        increment_count(&mut self.starts, start_prefix);

        for window in sentence.windows(4) {
            let prefix3 = [window[0], window[1], window[2]];
            let prefix2 = [window[1], window[2]];
            let prefix1 = window[2];
            let next = window[3];

            increment_nested_count(&mut self.model3, prefix3, next);
            increment_nested_count(&mut self.model2, prefix2, next);
            increment_nested_count(&mut self.model1, prefix1, next);
        }

        Ok(())
    }

    #[cfg(test)]
    pub fn generate_sentence<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        max_words: usize,
    ) -> Option<String> {
        self.generate_sentence_with_options(
            rng,
            GenerationOptions::new(max_words, DEFAULT_GENERATION_TEMPERATURE, 0),
        )
    }

    pub fn generate_sentence_with_options<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
    ) -> Option<String> {
        if !self.can_generate(options) {
            return None;
        }

        let mut context = choose_weighted_prefix(&self.starts, rng)?;
        let mut generated = Vec::new();

        self.collect_generated_tokens(rng, options, &mut context, &mut generated)?;

        (!generated.is_empty()).then_some(generated.join(""))
    }

    fn can_generate(&self, options: GenerationOptions) -> bool {
        options.is_valid() && !self.starts.is_empty()
    }

    fn collect_generated_tokens<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
        context: &mut [TokenId; 3],
        generated: &mut Vec<String>,
    ) -> Option<()> {
        for _ in 0..options.max_words {
            let allow_eos = generated.len() >= options.min_words_before_eos;
            let next = self.choose_next_token(*context, rng, options.temperature, allow_eos);

            if next == EOS_ID {
                break;
            }

            *context = [context[1], context[2], next];
            self.push_generated_token(generated, next)?;
        }

        Some(())
    }

    fn push_generated_token(&self, generated: &mut Vec<String>, next: TokenId) -> Option<()> {
        if next == BOS_ID {
            return Some(());
        }

        let token_index = usize::try_from(next).ok()?;
        let token = self.id_to_token.get(token_index)?.clone();
        generated.push(token);

        Some(())
    }

    fn intern_token(&mut self, token: &str) -> Result<TokenId, DynError> {
        if let Some(token_id) = self.token_to_id.get(token).copied() {
            return Ok(token_id);
        }

        let next_id = u32::try_from(self.id_to_token.len())
            .map_err(|_| "token vocabulary exceeds u32 range")?;
        let owned = token.to_owned();

        self.id_to_token.push(owned.clone());
        self.token_to_id.insert(owned, next_id);

        Ok(next_id)
    }

    fn choose_next_token<R: Rng + ?Sized>(
        &self,
        context: [TokenId; 3],
        rng: &mut R,
        temperature: f64,
        allow_eos: bool,
    ) -> TokenId {
        if let Some(next) = self.choose_with_backoff(context, rng, temperature, allow_eos) {
            return next;
        }

        if allow_eos {
            return EOS_ID;
        }

        self.choose_global_non_eos(rng, temperature)
            .unwrap_or(EOS_ID)
    }

    fn choose_with_backoff<R: Rng + ?Sized>(
        &self,
        context: [TokenId; 3],
        rng: &mut R,
        temperature: f64,
        allow_eos: bool,
    ) -> Option<TokenId> {
        if let Some(edges) = self.model3.get(&context)
            && let Some(next) = choose_weighted_token(edges, rng, temperature, allow_eos)
        {
            return Some(next);
        }

        let suffix2 = [context[1], context[2]];
        if let Some(edges) = self.model2.get(&suffix2)
            && let Some(next) = choose_weighted_token(edges, rng, temperature, allow_eos)
        {
            return Some(next);
        }

        if let Some(edges) = self.model1.get(&context[2])
            && let Some(next) = choose_weighted_token(edges, rng, temperature, allow_eos)
        {
            return Some(next);
        }

        None
    }

    fn choose_global_non_eos<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        temperature: f64,
    ) -> Option<TokenId> {
        let mut totals = HashMap::<TokenId, Count>::new();

        for edges in self.model1.values() {
            for (token, count) in edges {
                if *token == EOS_ID || *count == 0 {
                    continue;
                }

                let total = totals.entry(*token).or_insert(0);
                *total = total.saturating_add(*count);
            }
        }

        choose_weighted_token(&totals, rng, temperature, false)
    }
}

fn increment_count<K>(map: &mut HashMap<K, Count>, key: K)
where
    K: Eq + Hash,
{
    let count = map.entry(key).or_insert(0);
    *count = count.saturating_add(1);
}

fn increment_nested_count<K>(
    map: &mut HashMap<K, HashMap<TokenId, Count>>,
    prefix: K,
    next: TokenId,
) where
    K: Eq + Hash,
{
    let candidates = map.entry(prefix).or_default();
    increment_count(candidates, next);
}

fn choose_weighted_prefix<R: Rng + ?Sized>(
    starts: &HashMap<[TokenId; 3], Count>,
    rng: &mut R,
) -> Option<[TokenId; 3]> {
    let mut entries = starts
        .iter()
        .filter_map(|(prefix, count)| (*count > 0).then_some((*prefix, *count)))
        .collect::<Vec<_>>();

    entries.sort_unstable_by_key(|(prefix, _)| *prefix);
    choose_weighted_key(entries.as_slice(), rng, DEFAULT_GENERATION_TEMPERATURE)
}

fn choose_weighted_token<R: Rng + ?Sized>(
    edges: &HashMap<TokenId, Count>,
    rng: &mut R,
    temperature: f64,
    allow_eos: bool,
) -> Option<TokenId> {
    let mut entries = edges
        .iter()
        .filter_map(|(token, count)| {
            if *count == 0 || (!allow_eos && *token == EOS_ID) {
                return None;
            }

            Some((*token, *count))
        })
        .collect::<Vec<_>>();

    entries.sort_unstable_by_key(|(token, _)| *token);
    choose_weighted_key(entries.as_slice(), rng, temperature)
}

fn choose_weighted_key<K: Copy, R: Rng + ?Sized>(
    entries: &[(K, Count)],
    rng: &mut R,
    temperature: f64,
) -> Option<K> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return None;
    }

    if (temperature - DEFAULT_GENERATION_TEMPERATURE).abs() <= f64::EPSILON {
        return choose_weighted_key_default(entries, rng);
    }

    choose_weighted_key_with_temperature(entries, rng, temperature)
}

fn choose_weighted_key_default<K: Copy, R: Rng + ?Sized>(
    entries: &[(K, Count)],
    rng: &mut R,
) -> Option<K> {
    let total = entries
        .iter()
        .map(|(_, count)| *count)
        .fold(0_u64, u64::saturating_add);

    if total == 0 {
        return None;
    }

    let mut threshold = rng.random_range(1..=total);

    for (key, count) in entries {
        if threshold <= *count {
            return Some(*key);
        }

        threshold -= *count;
    }

    None
}

fn choose_weighted_key_with_temperature<K: Copy, R: Rng + ?Sized>(
    entries: &[(K, Count)],
    rng: &mut R,
    temperature: f64,
) -> Option<K> {
    let (weighted_entries, total_weight) = build_temperature_weights(entries, temperature)?;

    let threshold = rng.random_range(0.0_f64..total_weight);
    let mut cumulative = 0.0_f64;

    for (key, weight) in weighted_entries {
        cumulative += weight;
        if threshold < cumulative {
            return Some(key);
        }
    }

    entries.last().map(|(key, _)| *key)
}

fn build_temperature_weights<K: Copy>(
    entries: &[(K, Count)],
    temperature: f64,
) -> Option<(Vec<(K, f64)>, f64)> {
    let exponent = 1.0_f64 / temperature;
    let mut weighted_entries = Vec::with_capacity(entries.len());
    let mut total_weight = 0.0_f64;

    for (key, count) in entries {
        let Some(weight) = scaled_temperature_weight(*count, exponent) else {
            continue;
        };

        total_weight += weight;
        weighted_entries.push((*key, weight));
    }

    (total_weight > 0.0).then_some((weighted_entries, total_weight))
}

fn scaled_temperature_weight(count: Count, exponent: f64) -> Option<f64> {
    if count == 0 {
        return None;
    }

    let count_u32 = u32::try_from(count).unwrap_or(u32::MAX);
    let scaled = f64::from(count_u32).powf(exponent);

    (scaled.is_finite() && scaled > 0.0).then_some(scaled)
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::{GenerationOptions, MarkovChain};

    #[test]
    fn trains_and_generates_sentence_without_spaces() {
        let mut chain = MarkovChain::default();
        chain
            .train_tokens(&tokens(["今日", "は", "良い", "天気", "です", "ね"]))
            .expect("training should succeed");
        chain
            .train_tokens(&tokens(["今日", "は", "少し", "寒い", "です", "ね"]))
            .expect("training should succeed");

        let mut rng = StdRng::seed_from_u64(42);
        let sentence = chain
            .generate_sentence(&mut rng, 20)
            .expect("sentence should be generated");

        assert!(!sentence.contains("<BOS>"));
        assert!(!sentence.contains("<EOS>"));
        assert!(!sentence.chars().any(char::is_whitespace));
    }

    #[test]
    fn learns_single_token_sentence() {
        let mut chain = MarkovChain::default();
        chain
            .train_tokens(&tokens(["草"]))
            .expect("training should succeed");

        let mut rng = StdRng::seed_from_u64(1);
        let sentence = chain
            .generate_sentence(&mut rng, 20)
            .expect("sentence should be generated");

        assert_eq!(sentence, "草");
    }

    #[test]
    fn ignores_empty_tokens() {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&[]).expect("training should succeed");

        assert!(chain.starts.is_empty());
    }

    #[test]
    fn temperature_changes_sampling_bias() {
        let mut chain = MarkovChain::default();

        for _ in 0..120 {
            chain
                .train_tokens(&tokens(["a"]))
                .expect("training should succeed");
        }
        for _ in 0..8 {
            chain
                .train_tokens(&tokens(["b"]))
                .expect("training should succeed");
        }

        let low_temp = 0.35;
        let high_temp = 2.2;
        let sample_count = 200;

        let mut low_rng = StdRng::seed_from_u64(11);
        let mut high_rng = StdRng::seed_from_u64(11);

        let low_b = sample_b_frequency(&chain, &mut low_rng, sample_count, low_temp);
        let high_b = sample_b_frequency(&chain, &mut high_rng, sample_count, high_temp);

        assert!(high_b > low_b);
    }

    #[test]
    fn eos_can_be_deferred_until_min_words() {
        let mut chain = MarkovChain::default();
        chain
            .train_tokens(&tokens(["x"]))
            .expect("training should succeed");

        let mut rng = StdRng::seed_from_u64(99);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(6, 1.0, 2));

        assert_eq!(sentence, Some("xx".to_owned()));
    }

    #[test]
    fn rejects_invalid_generation_options() {
        let mut chain = MarkovChain::default();
        chain
            .train_tokens(&tokens(["x"]))
            .expect("training should succeed");

        let mut rng = StdRng::seed_from_u64(13);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(5, 0.0, 0));

        assert!(sentence.is_none());
    }

    fn sample_b_frequency(
        chain: &MarkovChain,
        rng: &mut StdRng,
        sample_count: usize,
        temperature: f64,
    ) -> usize {
        let mut hits = 0_usize;

        for _ in 0..sample_count {
            let sentence = chain
                .generate_sentence_with_options(rng, GenerationOptions::new(1, temperature, 0));
            if sentence.as_deref() == Some("b") {
                hits += 1;
            }
        }

        hits
    }

    fn tokens<const N: usize>(items: [&str; N]) -> Vec<String> {
        items.into_iter().map(ToOwned::to_owned).collect()
    }
}
