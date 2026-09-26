use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use markov_core::{Count, MarkovChain, NgramOrder};
use thiserror::Error;

use crate::{Codec, LimitKind, StorageCompressionMode, StorageError, StorageLimits};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Publication {
    /// The destination has not been changed by this attempt.
    NotPublished,
    /// Replacement succeeded, but syncing the namespace did not.
    PublishedNotDurable,
    /// The OS replacement operation failed without a definitive publication outcome.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveStage {
    CreateTemporary,
    Write,
    SyncFile,
    Replace,
    SyncDirectory,
}

#[derive(Debug, Error)]
pub enum SaveError {
    #[error("cannot prepare storage: {0}")]
    Preparation(#[from] StorageError),
    #[error("save failed at {stage:?} ({publication:?}): {source}")]
    Io {
        stage: SaveStage,
        publication: Publication,
        #[source]
        source: io::Error,
    },
}

impl SaveError {
    /// Retrying replaces the complete snapshot under the same writer lease. It never
    /// repeats learning, and is safe even if the prior publication outcome is unknown.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Io { .. })
    }
}

/// Exclusive authority to replace one canonical destination.
///
/// The sidecar lock is never deleted: unlinking it would let another writer lock a different inode.
/// Readers open the destination independently and see one complete generation.
#[derive(Debug)]
pub struct FileStore {
    path: PathBuf,
    _lock: File,
    limits: StorageLimits,
}

impl FileStore {
    /// Creates missing parents durably and acquires a nonblocking writer lease.
    /// The containing directory must not be moved or modified by uncooperative writers.
    ///
    /// # Errors
    /// Returns an error for invalid paths, lock contention, or unsupported/failed I/O.
    pub fn open(path: &Path, limits: StorageLimits) -> Result<Self, StorageError> {
        let parent = parent_of(path);
        create_parents(parent)?;
        let path = canonical_destination(path)?;
        if path.extension().is_some_and(|extension| {
            extension
                .to_str()
                .is_some_and(|name| name.eq_ignore_ascii_case("lock"))
        }) {
            return Err(StorageError::Format(
                ".lock destinations are reserved for writer leases".into(),
            ));
        }
        let mut lock_name = path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let lock_path = PathBuf::from(lock_name);
        if let Ok(meta) = fs::symlink_metadata(&lock_path)
            && !meta.is_file()
        {
            return Err(StorageError::Format(
                "writer lock must be a regular file".into(),
            ));
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(StorageError::Busy(path)),
            Err(TryLockError::Error(error)) => return Err(error.into()),
        }
        Ok(Self {
            path,
            _lock: lock,
            limits,
        })
    }

    /// # Errors
    /// Only a missing destination creates an empty model; all other failures propagate.
    pub fn load(&self, order: NgramOrder) -> Result<MarkovChain, StorageError> {
        match read_file(&self.path, self.limits) {
            Ok(bytes) => Codec::new(self.limits).decode_chain(&bytes, order),
            Err(StorageError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
                Ok(MarkovChain::new(order)?)
            }
            Err(error) => Err(error),
        }
    }

    /// Success acknowledges durability of the filtered snapshot, not of discarded edges.
    ///
    /// # Errors
    /// Preparation errors leave the destination unchanged; I/O errors carry publication state.
    pub fn save(
        &mut self,
        chain: &MarkovChain,
        min_count: Count,
        compression: StorageCompressionMode,
    ) -> Result<(), SaveError> {
        let bytes = Codec::new(self.limits).encode_chain(chain, min_count, compression)?;
        self.publish(&bytes)
    }

    /// Publishes a fully prepared document, including JSON exports, under this writer lease.
    /// Interrupted attempts may leave temporary files; only the destination is authoritative.
    /// Cleanup failure after durable publication does not undo or fail that publication.
    ///
    /// # Errors
    /// Reports the failing phase and whether replacement may have happened.
    pub fn publish(&mut self, bytes: &[u8]) -> Result<(), SaveError> {
        self.limits.check(
            LimitKind::FileBytes,
            crate::u64_from_usize(bytes.len(), "file size")?,
        )?;
        let parent = parent_of(&self.path);
        let mut temporary = tempfile::Builder::new()
            .prefix(".markov-")
            .tempfile_in(parent)
            .map_err(|source| failure(SaveStage::CreateTemporary, source))?;
        temporary
            .write_all(bytes)
            .map_err(|source| failure(SaveStage::Write, source))?;
        sync_file(temporary.as_file()).map_err(|source| failure(SaveStage::SyncFile, source))?;
        // Close the temporary handle before Windows replacement; retain automatic cleanup.
        let temporary = temporary.into_temp_path();
        replace(&temporary, &self.path).map_err(|source| failure(SaveStage::Replace, source))?;
        sync_namespace(parent).map_err(|source| failure(SaveStage::SyncDirectory, source))?;
        Ok(())
    }
}

const fn failure(stage: SaveStage, source: io::Error) -> SaveError {
    let publication = match stage {
        SaveStage::CreateTemporary | SaveStage::Write | SaveStage::SyncFile => {
            Publication::NotPublished
        }
        SaveStage::Replace => Publication::Unknown,
        SaveStage::SyncDirectory => Publication::PublishedNotDurable,
    };
    SaveError::Io {
        stage,
        publication,
        source,
    }
}

/// Reads at most the configured limit plus a one-byte overflow probe, including
/// when a file grows after metadata inspection.
///
/// # Errors
/// Reports I/O, allocation, non-regular input, or size-limit failures.
pub fn read_file(path: &Path, limits: StorageLimits) -> Result<Vec<u8>, StorageError> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(StorageError::Format("input must be a regular file".into()));
    }
    limits.check(LimitKind::FileBytes, metadata.len())?;
    let mut bytes = Vec::new();
    let mut reader = file.take(limits.file_bytes() + 1);
    let mut buffer = [0_u8; 8192];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let chunk = buffer
            .get(..count)
            .ok_or_else(|| StorageError::Format("invalid read length".into()))?;
        bytes.try_reserve(count)?;
        bytes.extend_from_slice(chunk);
        limits.check(
            LimitKind::FileBytes,
            crate::u64_from_usize(bytes.len(), "file size")?,
        )?;
    }
    Ok(bytes)
}

/// # Errors
/// Rejects aliases of the same file (including hard links) before a CLI output is opened.
/// Parent directories of a new output need not exist yet.
pub fn ensure_distinct_paths(input: &Path, output: &Path) -> Result<(), StorageError> {
    let input_canonical = fs::canonicalize(input)?;
    let output_canonical = canonical_destination(output)?;
    if input_canonical == output_canonical {
        return Err(StorageError::SameFile);
    }
    match fs::metadata(output) {
        Ok(_) if same_file::is_same_file(input, output)? => Err(StorageError::SameFile),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn parent_of(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn canonical_destination(path: &Path) -> io::Result<PathBuf> {
    match fs::canonicalize(path) {
        Ok(path) => Ok(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // A dangling symlink is not a missing model and must not silently become a new file.
            if fs::symlink_metadata(path).is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "dangling storage link",
                ));
            }
            let name = path.file_name().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "storage path needs a filename")
            })?;
            Ok(canonical_destination(parent_of(path))?.join(name))
        }
        Err(error) => Err(error),
    }
}

// Each missing directory is published within its own existing parent. Windows
// uses a write-through move for directory creation as well as file replacement.
fn create_parents(path: &Path) -> io::Result<()> {
    match fs::metadata(path) {
        Ok(meta) if meta.is_dir() => return sync_existing_directory(path),
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "storage parent is not a directory",
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let parent = parent_of(path);
    create_parents(parent)?;
    let temporary = tempfile::Builder::new()
        .prefix(".markov-dir-")
        .tempdir_in(parent)?;
    sync_existing_directory(temporary.path())?;
    match move_directory(temporary.path(), path) {
        Ok(()) => sync_namespace(parent),
        Err(error) if path.is_dir() => {
            // Another creator won. Sync that directory and its parent before use.
            sync_existing_directory(path)?;
            sync_namespace(parent).map_err(|sync_error| {
                io::Error::new(
                    sync_error.kind(),
                    format!(
                        "directory creation raced ({error}); synchronization failed: {sync_error}"
                    ),
                )
            })
        }
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "macos")]
fn sync_file(file: &File) -> io::Result<()> {
    rustix::fs::fcntl_fullfsync(file).map_err(Into::into)
}
#[cfg(not(target_os = "macos"))]
fn sync_file(file: &File) -> io::Result<()> {
    file.sync_all()
}

#[cfg(unix)]
fn sync_namespace(path: &Path) -> io::Result<()> {
    sync_file(&File::open(path)?)
}
#[cfg(windows)]
fn sync_namespace(_path: &Path) -> io::Result<()> {
    // Publication already uses MoveFileExW(MOVEFILE_WRITE_THROUGH).
    Ok(())
}
#[cfg(not(any(unix, windows)))]
fn sync_namespace(_path: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "durable namespace publication is unsupported",
    ))
}

#[cfg(not(windows))]
fn sync_existing_directory(path: &Path) -> io::Result<()> {
    sync_namespace(path)
}
#[cfg(windows)]
fn sync_existing_directory(_path: &Path) -> io::Result<()> {
    // Existing parents are the durable root; newly created parents are published
    // by move_directory with WRITE_THROUGH before this function can return to callers.
    Ok(())
}

#[cfg(windows)]
fn replace(source: &Path, target: &Path) -> io::Result<()> {
    atomicwrites::replace_atomic(source, target)
}
#[cfg(not(windows))]
fn replace(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}
#[cfg(windows)]
fn move_directory(source: &Path, target: &Path) -> io::Result<()> {
    atomicwrites::move_atomic(source, target)
}
#[cfg(not(windows))]
fn move_directory(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure(condition: bool, message: &str) -> std::io::Result<()> {
        if condition {
            Ok(())
        } else {
            Err(std::io::Error::other(message.to_owned()))
        }
    }

    fn ensure_eq<L: PartialEq<R> + std::fmt::Debug + ?Sized, R: std::fmt::Debug + ?Sized>(
        left: &L,
        right: &R,
        message: &str,
    ) -> std::io::Result<()> {
        if left == right {
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "{message}: left={left:?}, right={right:?}"
            )))
        }
    }

    type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    #[test]
    fn writer_lease_survives_replacement_and_releases_on_drop() -> Result {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("nested/model.mkv3");
        let limits = StorageLimits::default();
        let mut store = FileStore::open(&path, limits)?;
        store.publish(b"old")?;
        ensure(
            matches!(FileStore::open(&path, limits), Err(StorageError::Busy(_))),
            "assert contract failed",
        )?;
        store.publish(b"new")?;
        ensure_eq(
            &(read_file(&path, limits)?),
            &(b"new"),
            "assert_eq contract failed",
        )?;
        ensure(
            matches!(FileStore::open(&path, limits), Err(StorageError::Busy(_))),
            "assert contract failed",
        )?;
        drop(store);
        FileStore::open(&path, limits)?;
        Ok(())
    }

    #[test]
    fn preparation_failure_preserves_previous_file() -> Result {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.mkv3");
        let limits = StorageLimits::new(3, 10)?;
        let mut store = FileStore::open(&path, limits)?;
        store.publish(b"old")?;
        ensure(
            matches!(
                store.publish(b"too long"),
                Err(SaveError::Preparation(StorageError::Limit { .. }))
            ),
            "assert contract failed",
        )?;
        ensure_eq(
            &(read_file(&path, limits)?),
            &(b"old"),
            "assert_eq contract failed",
        )?;
        Ok(())
    }

    #[test]
    fn only_missing_model_starts_empty() -> Result {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.mkv3");
        let store = FileStore::open(&path, StorageLimits::default())?;
        let order = NgramOrder::new(1)?;
        ensure(
            store.load(order)?.starts().is_empty(),
            "assert contract failed",
        )?;
        fs::write(&path, b"corrupt")?;
        ensure(store.load(order).is_err(), "assert contract failed")?;
        Ok(())
    }

    #[test]
    fn rejects_input_output_aliases_and_bounds_file_read() -> Result {
        let directory = tempfile::tempdir()?;
        let input = directory.path().join("input");
        let alias = directory.path().join("alias");
        fs::write(&input, b"1234")?;
        fs::hard_link(&input, &alias)?;
        ensure(
            matches!(
                ensure_distinct_paths(&input, &alias),
                Err(StorageError::SameFile)
            ),
            "assert contract failed",
        )?;
        ensure(
            matches!(
                ensure_distinct_paths(&input, &directory.path().join("./input")),
                Err(StorageError::SameFile)
            ),
            "assert contract failed",
        )?;
        ensure_distinct_paths(&input, &directory.path().join("new/subdir/output"))?;
        ensure(
            matches!(
                read_file(&input, StorageLimits::new(3, 10)?),
                Err(StorageError::Limit { .. })
            ),
            "assert contract failed",
        )?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_alias_shares_lease_and_dangling_link_is_rejected() -> Result {
        let directory = tempfile::tempdir()?;
        let input = directory.path().join("input");
        let alias = directory.path().join("alias");
        fs::write(&input, b"model")?;
        std::os::unix::fs::symlink(&input, &alias)?;
        let _store = FileStore::open(&input, StorageLimits::default())?;
        ensure(
            matches!(
                FileStore::open(&alias, StorageLimits::default()),
                Err(StorageError::Busy(_))
            ),
            "assert contract failed",
        )?;
        ensure(
            matches!(
                ensure_distinct_paths(&input, &alias),
                Err(StorageError::SameFile)
            ),
            "assert contract failed",
        )?;
        fs::remove_file(&input)?;
        ensure(
            FileStore::open(&alias, StorageLimits::default()).is_err(),
            "assert contract failed",
        )?;
        Ok(())
    }
}
