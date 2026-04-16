use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenId(u32);

pub const BOS_TOKEN: &str = "<BOS>";
pub const EOS_TOKEN: &str = "<EOS>";

pub const BOS_ID: TokenId = TokenId(0);
pub const EOS_ID: TokenId = TokenId(1);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Count(u64);

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
