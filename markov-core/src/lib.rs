use std::{collections::HashMap, hash::Hash};

use rand::Rng;
use thiserror::Error;

mod sampling;
#[cfg(test)]
mod test_support;

pub type TokenId = u32;
pub type Count = u64;
pub type Prefix = Vec<TokenId>;

pub const DEFAULT_NGRAM_ORDER: usize = 6;

pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

pub const BOS_ID: TokenId = 0;
pub const EOS_ID: TokenId = 1;
pub const DEFAULT_GENERATION_TEMPERATURE: f64 = 1.0;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MarkovError {
    #[error("{0}")]
    Invalid(String),
}

impl From<&str> for MarkovError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

impl From<String> for MarkovError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

pub fn validate_ngram_order(ngram_order: usize, context: &str) -> Result<(), MarkovError> {
    if ngram_order == 0 {
        return Err(format!("{context} must be >= 1").into());
    }

    u32::try_from(ngram_order).map_err(|_error| format!("{context} must be <= {}", u32::MAX))?;

    Ok(())
}

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
    pub ngram_order: usize,
    pub token_to_id: HashMap<String, TokenId>,
    pub id_to_token: Vec<String>,
    pub models: Vec<HashMap<Prefix, HashMap<TokenId, Count>>>,
    pub starts: HashMap<Prefix, Count>,
}

impl MarkovChain {
    pub fn new(ngram_order: usize) -> Result<Self, MarkovError> {
        validate_ngram_order(ngram_order, "ngram_order")?;

        let mut token_to_id = HashMap::new();
        token_to_id.insert(BOS_TOKEN.to_owned(), BOS_ID);
        token_to_id.insert(EOS_TOKEN.to_owned(), EOS_ID);

        Ok(Self {
            ngram_order,
            token_to_id,
            id_to_token: vec![BOS_TOKEN.to_owned(), EOS_TOKEN.to_owned()],
            models: (0..ngram_order).map(|_| HashMap::new()).collect(),
            starts: HashMap::new(),
        })
    }

    pub fn train_tokens(&mut self, tokens: &[String]) -> Result<(), MarkovError> {
        if tokens.is_empty() {
            return Ok(());
        }

        let mut sentence = Vec::with_capacity(tokens.len().saturating_add(self.ngram_order + 1));
        sentence.extend(std::iter::repeat_n(BOS_ID, self.ngram_order));

        for token in tokens {
            let token_id = self.intern_token(token)?;
            sentence.push(token_id);
        }

        sentence.push(EOS_ID);

        let start_end = self
            .ngram_order
            .checked_add(1)
            .ok_or("start prefix bound overflow")?;
        let start_prefix = sentence
            .get(1..start_end)
            .ok_or("trained sentence is missing start prefix")?
            .to_vec();
        increment_count(&mut self.starts, start_prefix);

        let window_size = self
            .ngram_order
            .checked_add(1)
            .ok_or("training window size overflow")?;

        for window in sentence.windows(window_size) {
            let next = *window
                .last()
                .ok_or("training window is unexpectedly empty")?;

            for order in 1..=self.ngram_order {
                let prefix_start = self.ngram_order - order;
                let prefix = window
                    .get(prefix_start..self.ngram_order)
                    .ok_or("training prefix range is invalid")?
                    .to_vec();
                let model = self
                    .models
                    .get_mut(order - 1)
                    .ok_or("training model index is out of bounds")?;
                increment_nested_count(model, prefix, next);
            }
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

        let mut context = sampling::choose_weighted_prefix(&self.starts, rng, options.temperature)?;
        let mut generated = Vec::new();

        self.seed_generated_tokens_from_context(
            context.as_slice(),
            options.max_words,
            &mut generated,
        )?;
        self.collect_generated_tokens(rng, options, context.as_mut_slice(), &mut generated)?;

        (!generated.is_empty()).then_some(generated.join(""))
    }

    fn can_generate(&self, options: GenerationOptions) -> bool {
        options.is_valid()
            && !self.starts.is_empty()
            && self.models.len() == self.ngram_order
            && self.ngram_order > 0
    }

    fn collect_generated_tokens<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
        context: &mut [TokenId],
        generated: &mut Vec<String>,
    ) -> Option<()> {
        while generated.len() < options.max_words {
            let allow_eos = generated.len() >= options.min_words_before_eos;
            let next = self.choose_next_token(context, rng, options.temperature, allow_eos);

            if next == EOS_ID {
                break;
            }

            Self::advance_context(context, next)?;
            self.push_generated_token(generated, next)?;
        }

        Some(())
    }

    fn seed_generated_tokens_from_context(
        &self,
        context: &[TokenId],
        max_words: usize,
        generated: &mut Vec<String>,
    ) -> Option<()> {
        for token in context {
            if generated.len() >= max_words {
                break;
            }

            if *token == BOS_ID {
                continue;
            }

            if *token == EOS_ID {
                break;
            }

            self.push_generated_token(generated, *token)?;
        }

        Some(())
    }

    fn advance_context(context: &mut [TokenId], next: TokenId) -> Option<()> {
        if context.is_empty() {
            return None;
        }

        context.rotate_left(1);
        let last = context.last_mut()?;
        *last = next;
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

    fn intern_token(&mut self, token: &str) -> Result<TokenId, MarkovError> {
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
        context: &[TokenId],
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
        context: &[TokenId],
        rng: &mut R,
        temperature: f64,
        allow_eos: bool,
    ) -> Option<TokenId> {
        for order in (1..=self.ngram_order).rev() {
            let Some(prefix_start) = context.len().checked_sub(order) else {
                continue;
            };
            let Some(prefix) = context.get(prefix_start..) else {
                continue;
            };
            let Some(model) = self.models.get(order - 1) else {
                continue;
            };
            if let Some(edges) = model.get(prefix)
                && let Some(next) =
                    sampling::choose_weighted_token(edges, rng, temperature, allow_eos)
            {
                return Some(next);
            }
        }

        None
    }

    fn choose_global_non_eos<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        temperature: f64,
    ) -> Option<TokenId> {
        let mut totals = HashMap::<TokenId, Count>::new();

        for edges in self.models.first()?.values() {
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

    use crate::MarkovError;
    use crate::test_support::{ensure, ensure_eq};
    use rand::{SeedableRng, rngs::StdRng};

    use super::{
        BOS_ID, Count, DEFAULT_NGRAM_ORDER, EOS_ID, GenerationOptions, MarkovChain, Prefix, TokenId,
    };

    #[test]
    fn trains_and_generates_sentence_without_spaces() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;
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
    fn learns_single_token_sentence() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;
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
    fn ignores_empty_tokens() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;
        chain.train_tokens(&[])?;

        ensure(
            chain.starts.is_empty(),
            "empty training input must not create starts",
        )?;
        Ok(())
    }

    #[test]
    fn temperature_changes_sampling_bias() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;

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
    fn eos_can_be_deferred_until_min_words() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;
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
    fn rejects_invalid_generation_options() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;
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
    fn stores_start_prefix_using_first_token() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;
        chain.train_tokens(&tokens(["a", "b"]))?;

        let Some(a_id) = chain.token_to_id.get("a").copied() else {
            return Err("token id for 'a' should exist".into());
        };

        let expected_prefix = [vec![BOS_ID; DEFAULT_NGRAM_ORDER - 1], vec![a_id]].concat();
        ensure_eq(
            &chain.starts.get(&expected_prefix),
            &Some(&1),
            "start prefix should use the first generated token",
        )?;
        ensure(
            !chain
                .starts
                .contains_key(&vec![BOS_ID; DEFAULT_NGRAM_ORDER]),
            "pure BOS prefix must not be stored as a start",
        )?;
        Ok(())
    }

    #[test]
    fn emits_seeded_start_token_before_first_transition() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;

        add_token(&mut chain, 2, "a");

        let start = [vec![BOS_ID; DEFAULT_NGRAM_ORDER - 1], vec![2]].concat();
        chain.starts.insert(start.clone(), 1);

        for order in 1..=DEFAULT_NGRAM_ORDER {
            let prefix_start = start
                .len()
                .checked_sub(order)
                .ok_or("seed prefix start underflow")?;
            let prefix = start
                .get(prefix_start..)
                .ok_or("seed prefix slice is invalid")?
                .to_vec();
            insert_model(&mut chain, order, prefix, HashMap::from([(EOS_ID, 1)]))?;
        }

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

    #[test]
    fn backoff_walks_all_levels_from_seven_to_one() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(7)?;

        for (token_id, token) in [
            (2, "a"),
            (3, "b"),
            (4, "c"),
            (5, "d"),
            (6, "e"),
            (7, "f"),
            (8, "g"),
        ] {
            add_token(&mut chain, token_id, token);
        }

        let start = vec![BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, 2];
        chain.starts.insert(start.clone(), 1);
        insert_model(&mut chain, 7, start, HashMap::from([(3, 1)]))?;
        insert_model(
            &mut chain,
            6,
            vec![BOS_ID, BOS_ID, BOS_ID, BOS_ID, 2, 3],
            HashMap::from([(4, 1)]),
        )?;
        insert_model(
            &mut chain,
            5,
            vec![BOS_ID, BOS_ID, 2, 3, 4],
            HashMap::from([(5, 1)]),
        )?;
        insert_model(&mut chain, 4, vec![2, 3, 4, 5], HashMap::from([(6, 1)]))?;
        insert_model(&mut chain, 3, vec![4, 5, 6], HashMap::from([(7, 1)]))?;
        insert_model(&mut chain, 2, vec![6, 7], HashMap::from([(8, 1)]))?;
        insert_model(&mut chain, 1, vec![8], HashMap::from([(EOS_ID, 1)]))?;

        let mut rng = StdRng::seed_from_u64(321);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(7, 1.0, 0));

        ensure_eq(
            &sentence,
            &Some("abcdefg".to_owned()),
            "generation should fall back from model7 down to model1 in order",
        )?;
        Ok(())
    }

    #[test]
    fn trains_only_up_to_configured_ngram_order() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(3)?;
        chain.train_tokens(&tokens(["a", "b", "c", "d"]))?;

        ensure(chain.models.len() == 3, "model count should match order")?;
        ensure(
            !model(&chain, 3)?.is_empty(),
            "model3 should be trained for order=3",
        )?;
        ensure(
            !model(&chain, 2)?.is_empty(),
            "model2 should be trained for order=3",
        )?;
        ensure(
            !model(&chain, 1)?.is_empty(),
            "model1 should be trained for order=3",
        )?;
        Ok(())
    }

    #[test]
    fn trains_requested_model_count_for_order_seven() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(7)?;
        chain.train_tokens(&tokens(["a", "b", "c", "d", "e", "f", "g"]))?;

        ensure(chain.models.len() == 7, "model count should match order=7")?;
        ensure(
            chain.models.iter().all(|model| !model.is_empty()),
            "all models up to order=7 should be populated",
        )?;
        Ok(())
    }

    #[test]
    fn start_seed_uses_configured_prefix() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(2)?;

        for (token_id, token) in [(2, "a"), (3, "b"), (4, "c")] {
            add_token(&mut chain, token_id, token);
        }

        chain.starts.insert(vec![3, 4], 1);
        insert_model(&mut chain, 2, vec![3, 4], HashMap::from([(EOS_ID, 1)]))?;
        insert_model(&mut chain, 1, vec![4], HashMap::from([(EOS_ID, 1)]))?;

        let mut rng = StdRng::seed_from_u64(17);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(5, 1.0, 0));

        ensure_eq(
            &sentence,
            &Some("bc".to_owned()),
            "start seeding should emit the configured prefix",
        )?;
        Ok(())
    }

    #[test]
    fn backoff_starts_from_configured_ngram_order() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(3)?;

        for (token_id, token) in [(2, "a"), (3, "b"), (4, "c")] {
            add_token(&mut chain, token_id, token);
        }

        let start = vec![BOS_ID, BOS_ID, 2];
        chain.starts.insert(start.clone(), 1);
        insert_model(&mut chain, 3, start, HashMap::from([(3, 1)]))?;
        insert_model(&mut chain, 2, vec![2, 3], HashMap::from([(4, 1)]))?;
        insert_model(&mut chain, 1, vec![4], HashMap::from([(EOS_ID, 1)]))?;

        let mut rng = StdRng::seed_from_u64(27);
        let sentence =
            chain.generate_sentence_with_options(&mut rng, GenerationOptions::new(6, 1.0, 0));

        ensure_eq(
            &sentence,
            &Some("abc".to_owned()),
            "generation should back off only from the configured order down to model1",
        )?;
        Ok(())
    }

    fn add_token(chain: &mut MarkovChain, token_id: u32, token: &str) {
        chain.token_to_id.insert(token.to_owned(), token_id);
        chain.id_to_token.push(token.to_owned());
    }

    fn insert_model(
        chain: &mut MarkovChain,
        order: usize,
        prefix: Prefix,
        edges: HashMap<TokenId, Count>,
    ) -> Result<(), MarkovError> {
        let model = chain
            .models
            .get_mut(order.saturating_sub(1))
            .ok_or_else(|| format!("model{order} should exist"))?;
        model.insert(prefix, edges);
        Ok(())
    }

    fn model(
        chain: &MarkovChain,
        order: usize,
    ) -> Result<&HashMap<Prefix, HashMap<TokenId, Count>>, MarkovError> {
        chain
            .models
            .get(order.saturating_sub(1))
            .ok_or_else(|| format!("model{order} should exist").into())
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
