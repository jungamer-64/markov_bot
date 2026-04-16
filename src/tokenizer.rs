use lindera::{
    dictionary::load_dictionary, mode::Mode, segmenter::Segmenter,
    tokenizer::Tokenizer as LinderaTokenizer,
};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::DynError;

#[derive(Clone)]
pub(crate) enum Tokenizer {
    Lindera(Box<LinderaTokenizer>),
    Fallback,
}

impl Tokenizer {
    /// # Errors
    /// Returns `DynError` if Lindera tokenizer initialization fails.
    pub(crate) fn new() -> Result<Self, DynError> {
        let tokenizer = build_lindera_tokenizer()?;
        Ok(Self::Lindera(Box::new(tokenizer)))
    }

    #[must_use]
    pub(crate) fn with_fallback() -> Self {
        Self::Fallback
    }

    pub(crate) fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = normalize_text(text);
        if normalized.is_empty() {
            return Vec::new();
        }

        match self {
            Self::Lindera(tokenizer) => match tokenize_with_lindera(tokenizer, &normalized) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                _ => fallback_tokenize(&normalized),
            },
            Self::Fallback => fallback_tokenize(&normalized),
        }
    }
}

fn tokenize_with_lindera(tokenizer: &LinderaTokenizer, text: &str) -> Result<Vec<String>, DynError> {
    let tokens = tokenizer
        .tokenize(text)?
        .into_iter()
        .map(|token| token.surface.as_ref().to_owned())
        .filter(|token| !token.trim().is_empty())
        .collect::<Vec<_>>();

    Ok(tokens)
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
