use std::collections::HashMap;

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

const STATE_DELIMITER: char = '\u{1f}';

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarkovChain {
    pub transitions: HashMap<String, HashMap<String, usize>>,
    pub starts: HashMap<String, usize>,
}

impl MarkovChain {
    pub fn train_tokens(&mut self, tokens: &[String]) {
        if tokens.len() < 3 {
            return;
        }

        let start_key = encode_state(&tokens[0], &tokens[1], &tokens[2]);
        increment_weight(&mut self.starts, start_key);

        if tokens.len() < 4 {
            return;
        }

        for window in tokens.windows(4) {
            let key = encode_state(&window[0], &window[1], &window[2]);
            let next = window[3].clone();
            let entry = self.transitions.entry(key).or_default();
            increment_weight(entry, next);
        }
    }

    pub fn generate_sentence<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        min_words: usize,
        max_words: usize,
    ) -> Option<String> {
        if self.starts.is_empty() || min_words == 0 || max_words == 0 || min_words > max_words {
            return None;
        }

        let target_len = rng.random_range(min_words..=max_words);
        let mut words = self.generate_tokens(rng, target_len)?;

        if words.len() < min_words {
            return None;
        }

        if words.len() > max_words {
            words.truncate(max_words);
        }

        Some(words.join(" "))
    }

    fn generate_tokens<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        target_len: usize,
    ) -> Option<Vec<String>> {
        let start_key = choose_weighted_key(&self.starts, rng)?;
        let mut words = decode_state(start_key)?.to_vec();

        while words.len() < target_len {
            let state = encode_state(
                &words[words.len().saturating_sub(3)],
                &words[words.len().saturating_sub(2)],
                &words[words.len().saturating_sub(1)],
            );

            let Some(candidates) = self.transitions.get(&state) else {
                break;
            };

            let Some(next) = choose_weighted_key(candidates, rng) else {
                break;
            };

            words.push(next.clone());
        }

        Some(words)
    }
}

fn encode_state(w1: &str, w2: &str, w3: &str) -> String {
    [w1, w2, w3].join(&STATE_DELIMITER.to_string())
}

fn decode_state(state: &str) -> Option<[String; 3]> {
    let mut segments = state.split(STATE_DELIMITER);
    let first = segments.next()?.to_owned();
    let second = segments.next()?.to_owned();
    let third = segments.next()?.to_owned();

    if segments.next().is_some() {
        return None;
    }

    Some([first, second, third])
}

fn increment_weight(map: &mut HashMap<String, usize>, key: String) {
    let count = map.entry(key).or_insert(0);
    *count += 1;
}

fn choose_weighted_key<'a, R: Rng + ?Sized>(
    map: &'a HashMap<String, usize>,
    rng: &mut R,
) -> Option<&'a String> {
    let total = map.values().copied().sum::<usize>();
    if total == 0 {
        return None;
    }

    let mut threshold = rng.random_range(0..total);

    for (key, weight) in map {
        if *weight == 0 {
            continue;
        }

        if threshold < *weight {
            return Some(key);
        }

        threshold -= *weight;
    }

    None
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::MarkovChain;

    #[test]
    fn trains_and_generates_in_requested_range() {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["今日", "は", "良い", "天気", "です", "ね"]));
        chain.train_tokens(&tokens(["今日", "は", "少し", "寒い", "です", "ね"]));

        let mut rng = StdRng::seed_from_u64(42);
        let sentence = chain
            .generate_sentence(&mut rng, 5, 20)
            .expect("sentence should be generated");

        let word_count = sentence.split_whitespace().count();
        assert!((5..=20).contains(&word_count));
    }

    #[test]
    fn ignores_too_short_tokens() {
        let mut chain = MarkovChain::default();
        chain.train_tokens(&tokens(["短い", "文"]));

        let mut rng = StdRng::seed_from_u64(1);
        assert!(chain.generate_sentence(&mut rng, 5, 20).is_none());
    }

    fn tokens<const N: usize>(items: [&str; N]) -> Vec<String> {
        items.into_iter().map(ToOwned::to_owned).collect()
    }
}
