use std::{collections::HashMap, fmt, hash::Hash};

use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod sampling;
#[cfg(test)]
pub mod test_support;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenId(pub u32);

impl From<u32> for TokenId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<TokenId> for u32 {
    fn from(value: TokenId) -> Self {
        value.0
    }
}

impl From<TokenId> for usize {
    fn from(value: TokenId) -> Self {
        // SAFETY: u32 is always representable as usize on 32/64-bit systems.
        // On 16-bit systems this would fail, but we don't support them.
        Self::try_from(value.0).unwrap_or(0)
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Count(pub u64);

impl From<u64> for Count {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<Count> for u64 {
    fn from(value: Count) -> Self {
        value.0
    }
}

impl Count {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn saturating_add(self, other: u64) -> Self {
        Self(self.0.saturating_add(other))
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Prefix(Vec<TokenId>);

impl From<Vec<TokenId>> for Prefix {
    fn from(value: Vec<TokenId>) -> Self {
        Self(value)
    }
}

impl Prefix {
    #[must_use]
    pub const fn new(tokens: Vec<TokenId>) -> Self {
        Self(tokens)
    }

    #[must_use]
    pub fn as_slice(&self) -> &[TokenId] {
        &self.0
    }

    pub fn push(&mut self, token: TokenId) {
        self.0.push(token);
    }

    pub fn rotate_left(&mut self, n: usize) {
        self.0.rotate_left(n);
    }

    #[must_use]
    pub fn last_mut(&mut self) -> Option<&mut TokenId> {
        self.0.last_mut()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

pub const DEFAULT_NGRAM_ORDER: usize = 6;

pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

pub const BOS_ID: TokenId = TokenId(0);
pub const EOS_ID: TokenId = TokenId(1);
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

/// # Errors
/// Returns `MarkovError::Invalid` if `ngram_order` is 0 or greater than `u32::MAX`.
pub fn validate_ngram_order(ngram_order: usize, context: &str) -> Result<(), MarkovError> {
    if ngram_order == 0 {
        return Err(format!("{context} must be >= 1").into());
    }

    u32::try_from(ngram_order).map_err(|_error| format!("{context} must be <= {}", u32::MAX))?;

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EosPolicy {
    Forbidden,
    Allowed,
}

impl From<bool> for EosPolicy {
    fn from(value: bool) -> Self {
        if value {
            Self::Allowed
        } else {
            Self::Forbidden
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    pub max_words: usize,
    pub temperature: f64,
    pub min_words_before_eos: usize,
}

impl GenerationOptions {
    /// # Errors
    /// Returns `MarkovError::Invalid` if the options are invalid.
    pub fn new(
        max_words: usize,
        temperature: f64,
        min_words_before_eos: usize,
    ) -> Result<Self, MarkovError> {
        let options = Self {
            max_words,
            temperature,
            min_words_before_eos,
        };

        if options.max_words == 0 {
            return Err("max_words must be > 0".into());
        }
        if !options.temperature.is_finite() || options.temperature <= 0.0 {
            return Err("temperature must be > 0 and finite".into());
        }
        if options.min_words_before_eos > options.max_words {
            return Err("min_words_before_eos must be <= max_words".into());
        }

        Ok(options)
    }
}

#[derive(Debug, Clone)]
pub struct MarkovChain {
    ngram_order: usize,
    token_to_id: HashMap<String, TokenId>,
    id_to_token: Vec<String>,
    models: Vec<HashMap<Prefix, HashMap<TokenId, Count>>>,
    starts: HashMap<Prefix, Count>,
}

impl MarkovChain {
    /// # Errors
    /// Returns `MarkovError::Invalid` if `ngram_order` is 0 or greater than `u32::MAX`.
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

    /// Creates a Markov chain from raw components.
    ///
    /// # Errors
    /// Returns `MarkovError::Invalid` if the components are inconsistent.
    pub fn from_parts(
        ngram_order: usize,
        token_to_id: HashMap<String, TokenId>,
        id_to_token: Vec<String>,
        models: Vec<HashMap<Prefix, HashMap<TokenId, Count>>>,
        starts: HashMap<Prefix, Count>,
    ) -> Result<Self, MarkovError> {
        validate_ngram_order(ngram_order, "ngram_order")?;
        if models.len() != ngram_order {
            return Err(format!(
                "model count mismatch: expected {ngram_order}, got {}",
                models.len()
            )
            .into());
        }

        Ok(Self {
            ngram_order,
            token_to_id,
            id_to_token,
            models,
            starts,
        })
    }

    #[must_use]
    pub const fn ngram_order(&self) -> usize {
        self.ngram_order
    }

    #[must_use]
    pub const fn token_to_id(&self) -> &HashMap<String, TokenId> {
        &self.token_to_id
    }

    #[must_use]
    pub fn id_to_token(&self) -> &[String] {
        &self.id_to_token
    }

    #[must_use]
    pub fn models(&self) -> &[HashMap<Prefix, HashMap<TokenId, Count>>] {
        &self.models
    }

    #[must_use]
    pub const fn starts(&self) -> &HashMap<Prefix, Count> {
        &self.starts
    }

    /// # Errors
    /// Returns `MarkovError::Invalid` if internal bounds or start prefix logic overflows.
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
        increment_count(&mut self.starts, Prefix(start_prefix));

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
                increment_nested_count(model, Prefix(prefix), next);
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
        let options = GenerationOptions::new(max_words, DEFAULT_GENERATION_TEMPERATURE, 0).ok()?;
        self.generate_sentence_with_options(rng, options)
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
        self.collect_generated_tokens(rng, options, &mut context, &mut generated)?;

        (!generated.is_empty()).then_some(generated.join(""))
    }

    fn can_generate(&self, options: GenerationOptions) -> bool {
        !self.starts.is_empty()
            && self.models.len() == self.ngram_order
            && self.ngram_order > 0
            && options.max_words > 0
    }

    fn collect_generated_tokens<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
        context: &mut Prefix,
        generated: &mut Vec<String>,
    ) -> Option<()> {
        while generated.len() < options.max_words {
            let policy = if generated.len() >= options.min_words_before_eos {
                EosPolicy::Allowed
            } else {
                EosPolicy::Forbidden
            };
            let next = self.choose_next_token(context.as_slice(), rng, options.temperature, policy);

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

    fn advance_context(context: &mut Prefix, next: TokenId) -> Option<()> {
        if context.as_slice().is_empty() {
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

        let token_index = usize::from(next);
        let token = self.id_to_token.get(token_index)?.clone();
        generated.push(token);

        Some(())
    }

    fn intern_token(&mut self, token: &str) -> Result<TokenId, MarkovError> {
        if let Some(token_id) = self.token_to_id.get(token).copied() {
            return Ok(token_id);
        }

        let next_id = TokenId(u32::try_from(self.id_to_token.len())
            .map_err(|_error| "token vocabulary exceeds u32 range")?);
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
        policy: EosPolicy,
    ) -> TokenId {
        if let Some(next) = self.choose_with_backoff(context, rng, temperature, policy) {
            return next;
        }

        if policy == EosPolicy::Allowed {
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
        policy: EosPolicy,
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
            if let Some(edges) = model.get(&Prefix(prefix.to_vec()))
                && let Some(next) =
                    sampling::choose_weighted_token(edges, rng, temperature, policy)
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
                if *token == EOS_ID || *count == Count::ZERO {
                    continue;
                }

                let total = totals.entry(*token).or_insert(Count::ZERO);
                *total = total.saturating_add(count.0);
            }
        }

        sampling::choose_weighted_token(&totals, rng, temperature, EosPolicy::Forbidden)
    }
}

fn increment_count<K>(map: &mut HashMap<K, Count>, key: K)
where
    K: Eq + Hash,
{
    let count = map.entry(key).or_insert(Count::ZERO);
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
        let options = GenerationOptions::new(6, 1.0, 2)?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

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

        let options = GenerationOptions::new(5, 0.0, 0);
        ensure(
            options.is_err(),
            "invalid generation options must return Err",
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

        let expected_prefix = Prefix([vec![BOS_ID; DEFAULT_NGRAM_ORDER - 1], vec![a_id]].concat());
        ensure_eq(
            &chain.starts.get(&expected_prefix),
            &Some(&Count(1)),
            "start prefix should use the first generated token",
        )?;
        ensure(
            !chain
                .starts
                .contains_key(&Prefix(vec![BOS_ID; DEFAULT_NGRAM_ORDER])),
            "pure BOS prefix must not be stored as a start",
        )?;
        Ok(())
    }

    #[test]
    fn emits_seeded_start_token_before_first_transition() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(DEFAULT_NGRAM_ORDER)?;

        add_token(&mut chain, TokenId(2), "a");

        let start = Prefix([vec![BOS_ID; DEFAULT_NGRAM_ORDER - 1], vec![TokenId(2)]].concat());
        chain.starts.insert(start.clone(), Count(1));

        for order in 1..=DEFAULT_NGRAM_ORDER {
            let prefix_start = start.0
                .len()
                .checked_sub(order)
                .ok_or("seed prefix start underflow")?;
            let prefix = start.0
                .get(prefix_start..)
                .ok_or("seed prefix slice is invalid")?
                .to_vec();
            insert_model(&mut chain, order, Prefix(prefix), HashMap::from([(EOS_ID, Count(1))]))?;
        }

        let mut rng = StdRng::seed_from_u64(123);
        let options = GenerationOptions::new(1, 1.0, 0)?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

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
            add_token(&mut chain, TokenId(token_id), token);
        }

        let start = Prefix(vec![BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, BOS_ID, TokenId(2)]);
        chain.starts.insert(start.clone(), Count(1));
        insert_model(&mut chain, 7, start, HashMap::from([(TokenId(3), Count(1))]))?;
        insert_model(
            &mut chain,
            6,
            Prefix(vec![BOS_ID, BOS_ID, BOS_ID, BOS_ID, TokenId(2), TokenId(3)]),
            HashMap::from([(TokenId(4), Count(1))]),
        )?;
        insert_model(
            &mut chain,
            5,
            Prefix(vec![BOS_ID, BOS_ID, TokenId(2), TokenId(3), TokenId(4)]),
            HashMap::from([(TokenId(5), Count(1))]),
        )?;
        insert_model(&mut chain, 4, Prefix(vec![TokenId(2), TokenId(3), TokenId(4), TokenId(5)]), HashMap::from([(TokenId(6), Count(1))]))?;
        insert_model(&mut chain, 3, Prefix(vec![TokenId(4), TokenId(5), TokenId(6)]), HashMap::from([(TokenId(7), Count(1))]))?;
        insert_model(&mut chain, 2, Prefix(vec![TokenId(6), TokenId(7)]), HashMap::from([(TokenId(8), Count(1))]))?;
        insert_model(&mut chain, 1, Prefix(vec![TokenId(8)]), HashMap::from([(EOS_ID, Count(1))]))?;

        let mut rng = StdRng::seed_from_u64(321);
        let options = GenerationOptions::new(7, 1.0, 0)?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

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
            add_token(&mut chain, TokenId(token_id), token);
        }

        chain.starts.insert(Prefix(vec![TokenId(3), TokenId(4)]), Count(1));
        insert_model(&mut chain, 2, Prefix(vec![TokenId(3), TokenId(4)]), HashMap::from([(EOS_ID, Count(1))]))?;
        insert_model(&mut chain, 1, Prefix(vec![TokenId(4)]), HashMap::from([(EOS_ID, Count(1))]))?;

        let mut rng = StdRng::seed_from_u64(17);
        let options = GenerationOptions::new(5, 1.0, 0)?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

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
            add_token(&mut chain, TokenId(token_id), token);
        }

        let start = Prefix(vec![BOS_ID, BOS_ID, TokenId(2)]);
        chain.starts.insert(start.clone(), Count(1));
        insert_model(&mut chain, 3, start, HashMap::from([(TokenId(3), Count(1))]))?;
        insert_model(&mut chain, 2, Prefix(vec![TokenId(2), TokenId(3)]), HashMap::from([(TokenId(4), Count(1))]))?;
        insert_model(&mut chain, 1, Prefix(vec![TokenId(4)]), HashMap::from([(EOS_ID, Count(1))]))?;

        let mut rng = StdRng::seed_from_u64(27);
        let options = GenerationOptions::new(6, 1.0, 0)?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

        ensure_eq(
            &sentence,
            &Some("abc".to_owned()),
            "generation should back off only from the configured order down to model1",
        )?;
        Ok(())
    }

    fn add_token(chain: &mut MarkovChain, token_id: TokenId, token: &str) {
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
            let options = GenerationOptions::new(1, temperature, 0).unwrap();
            let sentence = chain
                .generate_sentence_with_options(rng, options);
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
