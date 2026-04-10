use std::{collections::HashMap, hash::Hash};

use rand::{Rng, RngExt};

use crate::config::DynError;

pub type TokenId = u32;
pub type Count = u64;

pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

pub const BOS_ID: TokenId = 0;
pub const EOS_ID: TokenId = 1;

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

    pub fn generate_sentence<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        max_words: usize,
    ) -> Option<String> {
        if max_words == 0 || self.starts.is_empty() {
            return None;
        }

        let mut context = choose_weighted_prefix(&self.starts, rng)?;
        let mut generated = Vec::new();

        for _ in 0..max_words {
            let next = self.choose_next_token(context, rng);

            if next == EOS_ID {
                break;
            }

            context = [context[1], context[2], next];

            if next == BOS_ID {
                continue;
            }

            let token_index = usize::try_from(next).ok()?;
            let token = self.id_to_token.get(token_index)?.clone();
            generated.push(token);
        }

        if generated.is_empty() {
            return None;
        }

        Some(generated.join(""))
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

    fn choose_next_token<R: Rng + ?Sized>(&self, context: [TokenId; 3], rng: &mut R) -> TokenId {
        if let Some(edges) = self.model3.get(&context)
            && let Some(next) = choose_weighted_token(edges, rng)
        {
            return next;
        }

        let suffix2 = [context[1], context[2]];
        if let Some(edges) = self.model2.get(&suffix2)
            && let Some(next) = choose_weighted_token(edges, rng)
        {
            return next;
        }

        if let Some(edges) = self.model1.get(&context[2])
            && let Some(next) = choose_weighted_token(edges, rng)
        {
            return next;
        }

        EOS_ID
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
    choose_weighted_key(entries.as_slice(), rng)
}

fn choose_weighted_token<R: Rng + ?Sized>(
    edges: &HashMap<TokenId, Count>,
    rng: &mut R,
) -> Option<TokenId> {
    let mut entries = edges
        .iter()
        .filter_map(|(token, count)| (*count > 0).then_some((*token, *count)))
        .collect::<Vec<_>>();

    entries.sort_unstable_by_key(|(token, _)| *token);
    choose_weighted_key(entries.as_slice(), rng)
}

fn choose_weighted_key<K: Copy, R: Rng + ?Sized>(entries: &[(K, Count)], rng: &mut R) -> Option<K> {
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

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::MarkovChain;

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

    fn tokens<const N: usize>(items: [&str; N]) -> Vec<String> {
        items.into_iter().map(ToOwned::to_owned).collect()
    }
}
