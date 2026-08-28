use std::ffi::c_int;
use std::num::TryFromIntError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ClearKvCacheSeqError {
    #[error("provided sequence id is too large for an i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    #[error("provided start position is too large for an i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("sequence {seq_id} could not be partially removed over positions [{p0}, {p1})")]
    PartialSequenceNotRemoved { seq_id: c_int, p0: c_int, p1: c_int },
}
