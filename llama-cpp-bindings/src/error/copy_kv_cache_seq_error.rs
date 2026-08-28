use std::num::TryFromIntError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum CopyKvCacheSeqError {
    #[error("provided start position is too large for an i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("the context has no memory module attached")]
    MemoryHandleUnavailable,
}
