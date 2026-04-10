use lindera::{
    dictionary::load_dictionary,
    mode::Mode,
    segmenter::Segmenter,
    tokenizer::Tokenizer as LinderaTokenizer,
};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::DynError;

#[derive(Clone)]
pub struct Tokenizer {
    lindera_tokenizer: Option<LinderaTokenizer>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            lindera_tokenizer: build_lindera_tokenizer().ok(),
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = normalize_text(text);
        if normalized.is_empty() {
            return Vec::new();
        }

        match self.tokenize_with_lindera(&normalized) {
            Ok(tokens) if !tokens.is_empty() => tokens,
            _ => fallback_tokenize(&normalized),
        }
    }

    fn tokenize_with_lindera(&self, text: &str) -> Result<Vec<String>, DynError> {
        let tokenizer = self
            .lindera_tokenizer
            .as_ref()
            .ok_or("lindera tokenizer is not initialized")?;

        let tokens = tokenizer
            .tokenize(text)?
            .into_iter()
            .map(|token| token.surface.as_ref().to_owned())
            .filter(|token| !token.trim().is_empty())
            .collect::<Vec<_>>();

        Ok(tokens)
    }
}

fn build_lindera_tokenizer() -> Result<LinderaTokenizer, DynError> {
    let dictionary = load_dictionary("embedded://ipadic")?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);

    Ok(LinderaTokenizer::new(segmenter))
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .filter(|token| {
            !token.starts_with("http://")
                && !token.starts_with("https://")
                && !token.starts_with("<@")
                && !token.starts_with("<#")
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned()
}

fn fallback_tokenize(text: &str) -> Vec<String> {
    UnicodeSegmentation::unicode_words(text)
        .map(ToOwned::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::fallback_tokenize;

    #[test]
    fn fallback_tokenize_extracts_words() {
        let tokens = fallback_tokenize("これは test です");
        assert!(tokens.iter().any(|token| token == "test"));
    }
}
