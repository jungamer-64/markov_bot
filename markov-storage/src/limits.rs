use crate::StorageError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    FileBytes,
    VocabBytes,
}

/// Byte limits bound file input/output and decompressed vocabulary independently.
/// They are not a bound on the total resident size of the reconstructed model.
#[derive(Debug, Clone, Copy)]
pub struct StorageLimits {
    file_bytes: u64,
    vocab_bytes: u64,
}

impl Default for StorageLimits {
    fn default() -> Self {
        Self {
            file_bytes: 512 * 1024 * 1024,
            vocab_bytes: 256 * 1024 * 1024,
        }
    }
}

impl StorageLimits {
    /// # Errors
    /// Rejects zero limits and limits that cannot be represented by this process.
    pub fn new(file_bytes: u64, vocab_bytes: u64) -> Result<Self, StorageError> {
        for limit in [file_bytes, vocab_bytes] {
            if limit == 0 || usize::try_from(limit).is_err() || limit == u64::MAX {
                return Err(StorageError::Format(
                    "storage limits must be positive and fit usize with room for an overflow probe"
                        .into(),
                ));
            }
        }
        Ok(Self {
            file_bytes,
            vocab_bytes,
        })
    }

    #[must_use]
    pub const fn file_bytes(self) -> u64 {
        self.file_bytes
    }
    #[must_use]
    pub const fn vocab_bytes(self) -> u64 {
        self.vocab_bytes
    }

    pub(super) const fn check(self, kind: LimitKind, actual: u64) -> Result<(), StorageError> {
        let limit = match kind {
            LimitKind::FileBytes => self.file_bytes,
            LimitKind::VocabBytes => self.vocab_bytes,
        };
        if actual > limit {
            return Err(StorageError::Limit {
                kind,
                actual,
                limit,
            });
        }
        Ok(())
    }

    pub(super) fn check_vocab_tokens(self, tokens: &[String]) -> Result<(), StorageError> {
        let size = tokens.iter().try_fold(0_u64, |size, token| {
            size.checked_add(crate::u64_from_usize(token.len(), "token size")?)
                .ok_or_else(|| StorageError::Format("vocabulary size overflow".into()))
        })?;
        self.check(LimitKind::VocabBytes, size)
    }
}
