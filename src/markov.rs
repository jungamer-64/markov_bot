use std::{collections::HashMap, hash::Hash};

use rand::Rng;

use crate::config::DynError;

mod sampling;

pub(crate) type TokenId = u32;
pub(crate) type Count = u64;

pub(crate) const BOS_TOKEN: &str = "<BOS>";
pub(crate) const EOS_TOKEN: &str = "<EOS>";

pub(crate) const BOS_ID: TokenId = 0;
pub(crate) const EOS_ID: TokenId = 1;
pub(crate) const DEFAULT_GENERATION_TEMPERATURE: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
pub(crate) struct GenerationOptions {
    pub max_words: usize,
    pub temperature: f64,
    pub min_words_before_eos: usize,
}

impl GenerationOptions {
    pub(crate) const fn new(
        max_words: usize,
        temperature: f64,
        min_words_before_eos: usize,
    ) -> Self {
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
pub(crate) struct MarkovChain {
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
    pub(crate) fn train_tokens(&mut self, tokens: &[String]) -> Result<(), DynError> {
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

        let first_token = *sentence
            .get(3)
            .ok_or("trained sentence is missing first token")?;
        let start_prefix = [
            *sentence
                .get(1)
                .ok_or("trained sentence is missing second token")?,
            *sentence
                .get(2)
                .ok_or("trained sentence is missing third token")?,
            first_token,
        ];
        increment_count(&mut self.starts, start_prefix);

        for window in sentence.windows(4) {
            let [w1, w2, w3, next] = <&[TokenId; 4]>::try_from(window)
                .map_err(|_error| "training window must contain exactly four tokens")?;
            let prefix3 = [*w1, *w2, *w3];
            let prefix2 = [*w2, *w3];
            let prefix1 = *w3;
            let next = *next;

            increment_nested_count(&mut self.model3, prefix3, next);
            increment_nested_count(&mut self.model2, prefix2, next);
            increment_nested_count(&mut self.model1, prefix1, next);
        }

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn generate_sentence<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        max_words: usize,
    ) -> Option<String> {
        self.generate_sentence_with_options(
            rng,
            GenerationOptions::new(max_words, DEFAULT_GENERATION_TEMPERATURE, 0),
        )
    }

    pub(crate) fn generate_sentence_with_options<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
    ) -> Option<String> {
        if !self.can_generate(options) {
            return None;
        }

        let mut context = sampling::choose_weighted_prefix(&self.starts, rng, options.temperature)?;
        let mut generated = Vec::new();

        self.seed_generated_tokens_from_context(context, options.max_words, &mut generated)?;

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
        while generated.len() < options.max_words {
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

    fn seed_generated_tokens_from_context(
        &self,
        context: [TokenId; 3],
        max_words: usize,
        generated: &mut Vec<String>,
    ) -> Option<()> {
        for token in context {
            if generated.len() >= max_words {
                break;
            }

            if token == BOS_ID {
                continue;
            }

            if token == EOS_ID {
                break;
            }

            self.push_generated_token(generated, token)?;
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
            .map_err(|_error| "token vocabulary exceeds u32 range")?;
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
            && let Some(next) = sampling::choose_weighted_token(edges, rng, temperature, allow_eos)
        {
            return Some(next);
        }

        let suffix2 = [context[1], context[2]];
        if let Some(edges) = self.model2.get(&suffix2)
            && let Some(next) = sampling::choose_weighted_token(edges, rng, temperature, allow_eos)
        {
            return Some(next);
        }

        if let Some(edges) = self.model1.get(&context[2])
            && let Some(next) = sampling::choose_weighted_token(edges, rng, temperature, allow_eos)
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

        sampling::choose_weighted_token(&totals, rng, temperature, false)
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::config::DynError;
    use crate::test_support::{ensure, ensure_eq};
    use rand::{SeedableRng, rngs::StdRng};

    use super::{BOS_ID, EOS_ID, GenerationOptions, MarkovChain};

    #[test]
    fn trains_and_generates_sentence_without_spaces() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["今日", "は", "良い", "天気", "です", "ね"]))?;
        chain.train_tokens(&tokens(["今日", "は", "少し", "寒い", "です", "ね"]))?;

        let mut rng = StdRng::seed_from_u64(42);
        let Some(sentence) = chain.generate_sentence(&mut rng, 20) else {
            return Err("sentence should be generated".into());
        };

        ensure(
            !sentence.contains("<BOS>"),
            "generated sentence must not contain <BOS>",
        )?;
        ensure(
            !sentence.contains("<EOS>"),
            "generated sentence must not contain <EOS>",
        )?;
        ensure(
            !sentence.chars().any(char::is_whitespace),
            "generated sentence must not contain whitespace",
        )?;
        Ok(())
    }

    #[test]
    fn learns_single_token_sentence() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["草"]))?;

        let mut rng = StdRng::seed_from_u64(1);
        let Some(sentence) = chain.generate_sentence(&mut rng, 20) else {
            return Err("sentence should be generated".into());
        };

        ensure_eq(
            &sentence,
            &"草",
            "single-token sentence should be preserved",
        )?;
        Ok(())
    }

    #[test]
    fn ignores_empty_tokens() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&[])?;

        ensure(
            chain.starts.is_empty(),
            "empty training input must not create starts",
        )?;
        Ok(())
    }

    #[test]
    fn temperature_changes_sampling_bias() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();

        for _ in 0..120 {
            chain.train_tokens(&tokens(["a"]))?;
        }
        for _ in 0..8 {
            chain.train_tokens(&tokens(["b"]))?;
        }

        let low_temp = 0.35;
        let high_temp = 2.2;
        let sample_count = 200;

        let mut low_rng = StdRng::seed_from_u64(11);
        let mut high_rng = StdRng::seed_from_u64(11);

        let low_b = sample_b_frequency(&chain, &mut low_rng, sample_count, low_temp);
        let high_b = sample_b_frequency(&chain, &mut high_rng, sample_count, high_temp);

        ensure(
            high_b > low_b,
            "higher temperature should increase sampling frequency of the rarer token",
        )?;
        Ok(())
    }

    #[test]
    fn eos_can_be_deferred_until_min_words() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["x"]))?;

        let mut rng = StdRng::seed_from_u64(99);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(6, 1.0, 2));

        ensure_eq(
            &sentence,
            &Some("xx".to_owned()),
            "minimum words before EOS should defer termination",
        )?;
        Ok(())
    }

    #[test]
    fn rejects_invalid_generation_options() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["x"]))?;

        let mut rng = StdRng::seed_from_u64(13);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(5, 0.0, 0));

        ensure(
            sentence.is_none(),
            "invalid generation options must return None",
        )?;
        Ok(())
    }

    #[test]
    fn stores_start_prefix_using_first_token() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["a", "b"]))?;

        let Some(a_id) = chain.token_to_id.get("a").copied() else {
            return Err("token id for 'a' should exist".into());
        };

        ensure_eq(
            &chain.starts.get(&[BOS_ID, BOS_ID, a_id]),
            &Some(&1),
            "start prefix should use the first generated token",
        )?;
        ensure(
            !chain.starts.contains_key(&[BOS_ID, BOS_ID, BOS_ID]),
            "pure BOS prefix must not be stored as a start",
        )?;
        Ok(())
    }

    #[test]
    fn emits_seeded_start_token_before_first_transition() -> Result<(), DynError> {
        let mut chain = MarkovChain::default();

        chain.token_to_id.insert("a".to_owned(), 2);
        chain.id_to_token.push("a".to_owned());

        chain.starts.insert([BOS_ID, BOS_ID, 2], 1);
        chain
            .model3
            .insert([BOS_ID, BOS_ID, 2], HashMap::from([(EOS_ID, 1)]));
        chain
            .model2
            .insert([BOS_ID, 2], HashMap::from([(EOS_ID, 1)]));
        chain.model1.insert(2, HashMap::from([(EOS_ID, 1)]));

        let mut rng = StdRng::seed_from_u64(123);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(1, 1.0, 0));

        ensure_eq(
            &sentence,
            &Some("a".to_owned()),
            "seeded start token should be emitted before the first transition",
        )?;
        Ok(())
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
