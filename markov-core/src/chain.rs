use std::collections::HashMap;

use rand::Rng;

use crate::{
    BOS_ID, BOS_TOKEN, Count, EOS_ID, EOS_TOKEN, MarkovError, NgramOrder, Prefix, TokenId,
    options::{EosPolicy, GenerationOptions, Temperature},
    sampling,
};

#[derive(Debug, Clone)]
pub struct MarkovChain {
    order: NgramOrder,
    token_to_id: HashMap<String, TokenId>,
    id_to_token: Vec<String>,
    models: Vec<HashMap<Prefix, HashMap<TokenId, Count>>>,
    starts: HashMap<Prefix, Count>,
}

impl MarkovChain {
    /// # Errors
    /// Returns `MarkovError::InvalidNgramOrder` if `order` is 0.
    pub fn new(order: NgramOrder) -> Result<Self, MarkovError> {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(BOS_TOKEN.to_owned(), BOS_ID);
        token_to_id.insert(EOS_TOKEN.to_owned(), EOS_ID);

        let mut id_to_token = vec![String::new(); 2];
        let bos_slot = id_to_token
            .get_mut(BOS_ID.get() as usize)
            .ok_or_else(|| MarkovError::Boundary("BOS_ID is out of bounds".into()))?;
        *bos_slot = BOS_TOKEN.to_owned();

        let eos_slot = id_to_token
            .get_mut(EOS_ID.get() as usize)
            .ok_or_else(|| MarkovError::Boundary("EOS_ID is out of bounds".into()))?;
        *eos_slot = EOS_TOKEN.to_owned();

        Ok(Self {
            order,
            token_to_id,
            id_to_token,
            models: vec![HashMap::new(); order.as_usize()],
            starts: HashMap::new(),
        })
    }

    #[must_use]
    pub const fn order(&self) -> NgramOrder {
        self.order
    }

    #[must_use]
    pub fn id_to_token(&self) -> &[String] {
        &self.id_to_token
    }

    #[must_use]
    pub const fn starts(&self) -> &HashMap<Prefix, Count> {
        &self.starts
    }

    #[must_use]
    pub const fn token_to_id(&self) -> &HashMap<String, TokenId> {
        &self.token_to_id
    }

    #[must_use]
    pub fn models(&self) -> &[HashMap<Prefix, HashMap<TokenId, Count>>] {
        &self.models
    }

    /// # Errors
    /// Returns `MarkovError::Boundary` if the parts are inconsistent.
    pub fn from_parts(
        order: NgramOrder,
        token_to_id: HashMap<String, TokenId>,
        id_to_token: Vec<String>,
        models: Vec<HashMap<Prefix, HashMap<TokenId, Count>>>,
        starts: HashMap<Prefix, Count>,
    ) -> Result<Self, MarkovError> {
        if models.len() != order.as_usize() {
            return Err(MarkovError::Boundary(format!(
                "models count ({}) must match ngram_order ({})",
                models.len(),
                order.as_usize()
            )));
        }

        Ok(Self {
            order,
            token_to_id,
            id_to_token,
            models,
            starts,
        })
    }

    /// # Errors
    /// Returns `MarkovError::Boundary` if training fails due to internal inconsistency.
    pub fn train_tokens(&mut self, tokens: &[String]) -> Result<(), MarkovError> {
        if tokens.is_empty() {
            return Ok(())
        }

        let mut ids = Vec::with_capacity(tokens.len() + self.order.as_usize() + 1);
        for _ in 0..self.order.as_usize() {
            ids.push(BOS_ID);
        }
        for token in tokens {
            ids.push(self.get_or_insert_token(token));
        }
        ids.push(EOS_ID);

        // Update starts
        let start_range = ids.get(0..self.order.as_usize()).ok_or_else(|| {
            MarkovError::Boundary("failed to get start prefix range".into())
        })?;
        let start_prefix = Prefix::new(start_range.to_vec());
        let start_count = self.starts.entry(start_prefix).or_insert(Count::ZERO);
        *start_count = start_count.saturating_add(1);

        // Update models
        for window in ids.windows(self.order.as_usize() + 1) {
            let next = *window.last().ok_or_else(|| {
                MarkovError::Boundary("training window is unexpectedly empty".into())
            })?;

            for order_val in 1..=self.order.as_usize() {
                let prefix_start = self.order.as_usize() - order_val;
                let prefix_range = window.get(prefix_start..self.order.as_usize()).ok_or_else(|| {
                    MarkovError::Boundary("training prefix range is invalid".into())
                })?;
                let prefix = Prefix::new(prefix_range.to_vec());
                let model = self.models.get_mut(order_val - 1).ok_or_else(|| {
                    MarkovError::Boundary("training model index is out of bounds".into())
                })?;
                increment_nested_count(model, prefix, next);
            }
        }

        Ok(())
    }

    pub fn generate_sentence_with_options<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
    ) -> Option<String> {
        if self.starts.is_empty() {
            return None;
        }

        let mut context = sampling::choose_weighted_prefix(&self.starts, rng, options.temperature())?;
        let mut generated = Vec::new();

        self.seed_generated_tokens_from_context(
            context.as_slice(),
            options.max_words().get(),
            &mut generated,
        )?;
        self.collect_generated_tokens(rng, options, &mut context, &mut generated)?;

        (!generated.is_empty()).then_some(generated.join(""))
    }

    fn collect_generated_tokens<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
        context: &mut Prefix,
        generated: &mut Vec<String>,
    ) -> Option<()> {
        while generated.len() < options.max_words().get() {
            let policy = if generated.len() >= options.min_words_before_eos().get() {
                EosPolicy::Allowed
            } else {
                EosPolicy::Forbidden
            };
            let next =
                self.choose_next_token(rng, context.as_slice(), options.temperature(), policy)?;

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

            if *token != BOS_ID && *token != EOS_ID {
                self.push_generated_token(generated, *token)?;
            }
        }
        Some(())
    }

    fn push_generated_token(&self, generated: &mut Vec<String>, id: TokenId) -> Option<()> {
        let token = self.id_to_token.get(id.get() as usize)?;
        generated.push(token.clone());
        Some(())
    }

    fn advance_context(context: &mut Prefix, next: TokenId) -> Option<()> {
        if context.is_empty() {
            return None;
        }
        context.rotate_left(1);
        if let Some(last) = context.last_mut() {
            *last = next;
        }
        Some(())
    }

    fn choose_next_token<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        context: &[TokenId],
        temperature: Temperature,
        policy: EosPolicy,
    ) -> Option<TokenId> {
        if let Some(next) = self.choose_with_backoff(context, rng, temperature, policy) {
            return Some(next);
        }

        if policy == EosPolicy::Allowed {
            return Some(EOS_ID);
        }

        self.choose_global_non_eos(rng, temperature)
    }

    fn choose_with_backoff<R: Rng + ?Sized>(
        &self,
        context: &[TokenId],
        rng: &mut R,
        temperature: Temperature,
        policy: EosPolicy,
    ) -> Option<TokenId> {
        for order_val in (1..=self.order.as_usize()).rev() {
            let prefix_start = context.len().checked_sub(order_val)?;
            let prefix_slice = context.get(prefix_start..)?;
            let prefix = Prefix::new(prefix_slice.to_vec());
            let model = self.models.get(order_val - 1)?;
            if let Some(edges) = model.get(&prefix)
                && let Some(next) = sampling::choose_weighted_token(edges, rng, temperature, policy)
            {
                return Some(next);
            }
        }

        None
    }

    fn choose_global_non_eos<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        temperature: Temperature,
    ) -> Option<TokenId> {
        let mut totals = HashMap::<TokenId, Count>::new();

        for edges in self.models.first()?.values() {
            for (token, count) in edges {
                if *token == EOS_ID || count.get() == 0 {
                    continue;
                }

                let total = totals.entry(*token).or_insert(Count::ZERO);
                *total = total.saturating_add(count.get());
            }
        }

        sampling::choose_weighted_token(&totals, rng, temperature, EosPolicy::Forbidden)
    }

    fn get_or_insert_token(&mut self, token: &str) -> TokenId {
        if let Some(id) = self.token_to_id.get(token) {
            return *id;
        }

        let id = TokenId::new(self.id_to_token.len() as u32);
        self.token_to_id.insert(token.to_owned(), id);
        self.id_to_token.push(token.to_owned());
        id
    }
}

fn increment_nested_count(
    model: &mut HashMap<Prefix, HashMap<TokenId, Count>>,
    prefix: Prefix,
    next: TokenId,
) {
    let edges = model.entry(prefix).or_default();
    let count = edges.entry(next).or_insert(Count::ZERO);
    *count = count.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use crate::{MaxWords, MinWordsBeforeEos, Temperature};
    use crate::test_support::{ensure, ensure_eq};

    #[test]
    fn new_chain_has_correct_ngram_order() -> Result<(), MarkovError> {
        let order = NgramOrder::new(3)?;
        let chain = MarkovChain::new(order)?;
        ensure_eq(&chain.order(), &order, "ngram order should be 3")?;
        Ok(())
    }

    #[test]
    fn training_increases_token_count() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(NgramOrder::new(3)?)?;
        chain.train_tokens(&["a".to_owned(), "b".to_owned(), "c".to_owned()])?;
        ensure(chain.id_to_token().len() > 2, "token count should increase")?;
        Ok(())
    }

    #[test]
    fn generation_reproduces_trained_sequence_with_low_temp() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(NgramOrder::new(2)?)?;
        let tokens = vec!["apple".to_owned(), "banana".to_owned(), "cherry".to_owned()];
        chain.train_tokens(&tokens)?;

        let mut rng = StdRng::seed_from_u64(42);
        let options = GenerationOptions::new(
            MaxWords::new(10)?,
            Temperature::new(0.01)?,
            MinWordsBeforeEos::new(0),
        )?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

        ensure(sentence.is_some(), "should generate a sentence")?;
        ensure_eq(
            &sentence.ok_or_else(|| MarkovError::Boundary("sentence should be some".into()))?,
            &"applebananacherry".to_owned(),
            "should reproduce exactly at very low temperature",
        )?;
        Ok(())
    }

    #[test]
    fn temperature_affects_sampling_probabilities() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(NgramOrder::new(1)?)?;
        chain.train_tokens(&["c".to_owned()])?;
        for _ in 0..10 {
            chain.train_tokens(&["c".to_owned()])?;
        }
        chain.train_tokens(&["b".to_owned()])?;

        let mut low_rng = StdRng::seed_from_u64(11);
        let mut high_rng = StdRng::seed_from_u64(11);
        let low_temp = Temperature::new(0.1)?;
        let high_temp = Temperature::new(2.2)?;
        let sample_count = 200;

        let low_b = sample_b_frequency(&chain, &mut low_rng, sample_count, low_temp)?;
        let high_b = sample_b_frequency(&chain, &mut high_rng, sample_count, high_temp)?;

        ensure(
            high_b > low_b,
            &format!("higher temperature should increase sampling frequency of the rarer token (low_b: {low_b}, high_b: {high_b})"),
        )?;
        Ok(())
    }

    fn sample_b_frequency(
        chain: &MarkovChain,
        rng: &mut StdRng,
        sample_count: usize,
        temperature: Temperature,
    ) -> Result<usize, MarkovError> {
        let mut hits = 0_usize;

        for _ in 0..sample_count {
            let options = GenerationOptions::new(
                MaxWords::new(1)?,
                temperature,
                MinWordsBeforeEos::new(0),
            )?;
            let sentence = chain.generate_sentence_with_options(rng, options);
            if sentence.as_deref() == Some("b") {
                hits += 1;
            }
        }

        Ok(hits)
    }
}
