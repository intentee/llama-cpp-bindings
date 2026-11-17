use std::num::TryFromIntError;

use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_model::model_error::ModelError;

use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MemorySeqDivError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("provided start position is too large for an i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("model rope type is incompatible with sequence position arithmetic")]
    IncompatibleRopeType,
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error(transparent)]
    SequenceIdOutOfRange(#[from] SequenceIdOutOfRange),
    #[error(transparent)]
    UnknownRopeType(#[from] ModelError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
