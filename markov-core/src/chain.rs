use std::collections::HashMap;

use rand::Rng;

use crate::{
    BOS_ID, BOS_TOKEN, Count, EOS_ID, EOS_TOKEN, MarkovError, Prefix, TokenId,
    options::{EosPolicy, GenerationOptions},
    sampling, validate_ngram_order,
};

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

        let mut id_to_token = vec![String::new(); 2];
        BOS_TOKEN.clone_into(
            id_to_token
                .get_mut(usize::from(BOS_ID))
                .ok_or_else(|| MarkovError::Invalid("BOS_ID is out of bounds".into()))?,
        );
        EOS_TOKEN.clone_into(
            id_to_token
                .get_mut(usize::from(EOS_ID))
                .ok_or_else(|| MarkovError::Invalid("EOS_ID is out of bounds".into()))?,
        );

        Ok(Self {
            ngram_order,
            token_to_id,
            id_to_token,
            models: vec![HashMap::new(); ngram_order],
            starts: HashMap::new(),
        })
    }

    #[must_use]
    pub const fn ngram_order(&self) -> usize {
        self.ngram_order
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
    /// Returns `MarkovError::Invalid` if the parts are invalid.
    pub fn from_parts(
        ngram_order: usize,
        token_to_id: HashMap<String, TokenId>,
        id_to_token: Vec<String>,
        models: Vec<HashMap<Prefix, HashMap<TokenId, Count>>>,
        starts: HashMap<Prefix, Count>,
    ) -> Result<Self, MarkovError> {
        validate_ngram_order(ngram_order, "ngram_order")?;

        if models.len() != ngram_order {
            return Err(MarkovError::Invalid(format!(
                "models count ({}) must match ngram_order ({})",
                models.len(),
                ngram_order
            )));
        }

        Ok(Self {
            ngram_order,
            token_to_id,
            id_to_token,
            models,
            starts,
        })
    }

    /// # Errors
    /// Returns `MarkovError::Invalid` if training fails.
    pub fn train_tokens(&mut self, tokens: &[String]) -> Result<(), MarkovError> {
        if tokens.is_empty() {
            return Ok(());
        }

        let mut ids = Vec::with_capacity(tokens.len() + self.ngram_order + 1);
        ids.extend(std::iter::repeat_n(BOS_ID, self.ngram_order));
        for token in tokens {
            ids.push(self.get_or_insert_token(token));
        }
        ids.push(EOS_ID);

        // Update starts
        let start_prefix = Prefix(
            ids.get(0..self.ngram_order)
                .ok_or_else(|| MarkovError::Invalid("failed to get start prefix".into()))?
                .to_vec(),
        );
        let start_count = self.starts.entry(start_prefix).or_insert(Count::ZERO);
        *start_count = start_count.saturating_add(1);

        // Update models
        for window in ids.windows(self.ngram_order + 1) {
            let next = *window
                .last()
                .ok_or_else(|| MarkovError::Invalid("training window is unexpectedly empty".into()))?;

            for order in 1..=self.ngram_order {
                let prefix_start = self.ngram_order - order;
                let prefix = window
                    .get(prefix_start..self.ngram_order)
                    .ok_or_else(|| {
                        MarkovError::Invalid("training prefix range is invalid".into())
                    })?
                    .to_vec();
                let model = self.models.get_mut(order - 1).ok_or_else(|| {
                    MarkovError::Invalid("training model index is out of bounds".into())
                })?;
                increment_nested_count(model, Prefix(prefix), next);
            }
        }

        Ok(())
    }

    pub fn generate_sentence_with_options<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
    ) -> Option<String> {
        if !self.can_generate(options) {
            return None;
        }

        let mut context = sampling::choose_weighted_prefix(&self.starts, rng, options.temperature())?;
        let mut generated = Vec::new();

        self.seed_generated_tokens_from_context(
            context.as_slice(),
            options.max_words(),
            &mut generated,
        )?;
        self.collect_generated_tokens(rng, options, &mut context, &mut generated)?;

        (!generated.is_empty()).then_some(generated.join(""))
    }

    fn can_generate(&self, options: GenerationOptions) -> bool {
        !self.starts.is_empty()
            && self.models.len() == self.ngram_order
            && self.ngram_order > 0
            && options.max_words() > 0
    }

    fn collect_generated_tokens<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        options: GenerationOptions,
        context: &mut Prefix,
        generated: &mut Vec<String>,
    ) -> Option<()> {
        while generated.len() < options.max_words() {
            let policy = if generated.len() >= options.min_words_before_eos() {
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
        let token = self.id_to_token.get(usize::from(id))?;
        generated.push(token.clone());
        Some(())
    }

    fn advance_context(context: &mut Prefix, next: TokenId) -> Option<()> {
        if context.0.is_empty() {
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
        temperature: f64,
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
        temperature: f64,
        policy: EosPolicy,
    ) -> Option<TokenId> {
        for order in (1..=self.ngram_order).rev() {
            let prefix_start = context.len().checked_sub(order)?;
            let prefix = context.get(prefix_start..)?;
            let model = self.models.get(order - 1)?;
            if let Some(edges) = model.get(&Prefix(prefix.to_vec()))
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
        temperature: f64,
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

        let id = TokenId::new(u32::try_from(self.id_to_token.len()).unwrap_or(0));
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
    use crate::test_support::{ensure, ensure_eq};

    #[test]
    fn new_chain_has_correct_ngram_order() -> Result<(), MarkovError> {
        let chain = MarkovChain::new(3)?;
        ensure_eq(&chain.ngram_order(), &3, "ngram order should be 3")?;
        Ok(())
    }

    #[test]
    fn training_increases_token_count() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(3)?;
        chain.train_tokens(&["a".to_owned(), "b".to_owned(), "c".to_owned()])?;
        ensure(chain.id_to_token().len() > 2, "token count should increase")?;
        Ok(())
    }

    #[test]
    fn generation_reproduces_trained_sequence_with_low_temp() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(2)?;
        let tokens = vec!["apple".to_owned(), "banana".to_owned(), "cherry".to_owned()];
        chain.train_tokens(&tokens)?;

        let mut rng = StdRng::seed_from_u64(42);
        let options = GenerationOptions::new(10, 0.01, 0)?;
        let sentence = chain.generate_sentence_with_options(&mut rng, options);

        ensure(sentence.is_some(), "should generate a sentence")?;
        ensure_eq(
            &sentence.ok_or_else(|| MarkovError::Invalid("sentence should be some".into()))?,
            &"applebananacherry".to_owned(),
            "should reproduce exactly at very low temperature",
        )?;
        Ok(())
    }

    #[test]
    fn temperature_affects_sampling_probabilities() -> Result<(), MarkovError> {
        let mut chain = MarkovChain::new(1)?;
        // Create a fork at BOS: BOS -> "c" or BOS -> "b"
        // "c" is much more frequent
        chain.train_tokens(&["c".to_owned()])?;
        for _ in 0..10 {
            chain.train_tokens(&["c".to_owned()])?;
        }
        chain.train_tokens(&["b".to_owned()])?;

        let mut low_rng = StdRng::seed_from_u64(11);
        let mut high_rng = StdRng::seed_from_u64(11);
        let low_temp = 0.1;
        let high_temp = 2.2;
        let sample_count = 200;

        let low_b = sample_b_frequency(&chain, &mut low_rng, sample_count, low_temp)?;
        let high_b = sample_b_frequency(&chain, &mut high_rng, sample_count, high_temp)?;

        println!("low_b: {low_b}, high_b: {high_b}");

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
        temperature: f64,
    ) -> Result<usize, MarkovError> {
        let mut hits = 0_usize;

        for _ in 0..sample_count {
            let options = GenerationOptions::new(1, temperature, 0)?;
            let sentence = chain.generate_sentence_with_options(rng, options);
            if sentence.as_deref() == Some("b") {
                hits += 1;
            }
        }

        Ok(hits)
    }
}
