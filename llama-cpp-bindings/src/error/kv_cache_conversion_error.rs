use std::ffi::c_int;
use std::num::TryFromIntError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvCacheConversionError {
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("the context has no memory module attached")]
    MemoryHandleUnavailable,
    #[error("sequence {seq_id} could not be partially removed over positions [{p0}, {p1})")]
    PartialSequenceNotRemoved { seq_id: c_int, p0: c_int, p1: c_int },
}
