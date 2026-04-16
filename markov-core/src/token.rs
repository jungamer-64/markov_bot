use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::MarkovError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenId(u32);

pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

pub const BOS_ID: TokenId = TokenId(0);
pub const EOS_ID: TokenId = TokenId(1);

impl TokenId {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Default)]
pub struct TokenRegistry {
    token_to_id: HashMap<String, TokenId>,
    id_to_token: Vec<String>,
}

impl TokenRegistry {
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        };

        // Initialize with special tokens
        registry.id_to_token.push(BOS_TOKEN.to_owned());
        registry.token_to_id.insert(BOS_TOKEN.to_owned(), BOS_ID);

        registry.id_to_token.push(EOS_TOKEN.to_owned());
        registry.token_to_id.insert(EOS_TOKEN.to_owned(), EOS_ID);

        registry
    }

    /// # Errors
    /// Returns `MarkovError::TokenLimitExceeded` if the number of tokens exceeds `u32::MAX`.
    pub fn get_or_insert(&mut self, token: &str) -> Result<TokenId, MarkovError> {
        if let Some(id) = self.token_to_id.get(token) {
            return Ok(*id);
        }

        let id_val = u32::try_from(self.id_to_token.len())
            .map_err(|_| MarkovError::TokenLimitExceeded)?;
        let id = TokenId::new(id_val);

        self.id_to_token.push(token.to_owned());
        self.token_to_id.insert(token.to_owned(), id);

        Ok(id)
    }

    #[must_use]
    pub fn get_token(&self, id: TokenId) -> Option<&str> {
        let idx = usize::try_from(id.get()).ok()?;
        self.id_to_token.get(idx).map(String::as_str)
    }

    #[must_use]
    pub fn get_id(&self, token: &str) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.id_to_token
    }

    #[must_use]
    pub const fn token_to_id(&self) -> &HashMap<String, TokenId> {
        &self.token_to_id
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// # Errors
    /// Returns `MarkovError::Boundary` if the parts are inconsistent (e.g., special tokens missing or misaligned).
    pub fn from_parts(
        token_to_id: HashMap<String, TokenId>,
        id_to_token: Vec<String>,
    ) -> Result<Self, MarkovError> {
        if id_to_token.len() != token_to_id.len() {
            return Err(MarkovError::Boundary("Token registry parts size mismatch".into()));
        }

        if id_to_token.get(0).map(String::as_str) != Some(BOS_TOKEN)
            || token_to_id.get(BOS_TOKEN) != Some(&BOS_ID)
        {
            return Err(MarkovError::Boundary("BOS token missing or misaligned".into()));
        }

        if id_to_token.get(1).map(String::as_str) != Some(EOS_TOKEN)
            || token_to_id.get(EOS_TOKEN) != Some(&EOS_ID)
        {
            return Err(MarkovError::Boundary("EOS token missing or misaligned".into()));
        }

        Ok(Self {
            token_to_id,
            id_to_token,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Count(u64);

impl Count {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

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

impl Prefix {
    #[must_use]
    pub const fn new(tokens: Vec<TokenId>) -> Self {
        Self(tokens)
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
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
    pub fn get(&self, range: std::ops::Range<usize>) -> Option<&[TokenId]> {
        self.0.get(range)
    }
}
