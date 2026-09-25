use std::borrow::Cow;

use lindera::{dictionary::load_dictionary, mode::Mode, segmenter::Segmenter};
use thiserror::Error;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Error)]
pub(crate) enum TokenizerError {
    #[error("Lindera initialization failed: {0}")]
    Lindera(String),

    #[error("Lindera tokenization failed: {0}")]
    LinderaTokenization(#[from] lindera::error::LinderaError),
}

#[derive(Clone)]
pub(crate) enum Tokenizer {
    Lindera(Box<Segmenter>),
    Fallback,
}

impl Tokenizer {
    /// # Errors
    /// Returns `TokenizerError` if Lindera tokenizer initialization fails.
    pub(crate) fn new() -> Result<Self, TokenizerError> {
        let segmenter = build_lindera_segmenter()?;
        Ok(Self::Lindera(Box::new(segmenter)))
    }

    #[must_use]
    pub(crate) const fn with_fallback() -> Self {
        Self::Fallback
    }

    pub(crate) fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = normalize_text(text);
        if normalized.is_empty() {
            return Vec::new();
        }

        match self {
            Self::Lindera(segmenter) => match segment_with_lindera(segmenter, &normalized) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                _ => fallback_tokenize(&normalized),
            },
            Self::Fallback => fallback_tokenize(&normalized),
        }
    }
}

fn segment_with_lindera(segmenter: &Segmenter, text: &str) -> Result<Vec<String>, TokenizerError> {
    let tokens = segmenter
        .segment(Cow::Borrowed(text))?
        .into_iter()
        .map(|token| token.surface.as_ref().to_owned())
        .filter(|token| !token.trim().is_empty())
        .collect::<Vec<_>>();

    Ok(tokens)
}

fn build_lindera_segmenter() -> Result<Segmenter, TokenizerError> {
    let dictionary =
        load_dictionary("embedded://ipadic").map_err(|e| TokenizerError::Lindera(e.to_string()))?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);

    Ok(segmenter)
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
    use super::{Tokenizer, fallback_tokenize};

    #[test]
    fn lindera_segments_japanese_text() {
        let result =
            Tokenizer::new().map(|tokenizer| tokenizer.tokenize("関西国際空港限定トートバッグ"));
        assert!(
            matches!(result, Ok(ref tokens) if tokens == &[
                "関西国際空港", "限定", "トートバッグ"
            ]),
            "unexpected segmentation: {result:?}"
        );
    }

    #[test]
    fn fallback_tokenize_extracts_words() {
        let tokens = fallback_tokenize("これは test です");
        assert!(tokens.iter().any(|token| token == "test"));
    }
}
