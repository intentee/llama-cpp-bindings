use std::ffi::c_int;
use std::num::TryFromIntError;

use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_model::model_error::ModelError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextMemoryError {
    #[error("provided end position is too large for an i32")]
    EndPositionTooLarge(#[source] TryFromIntError),
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("model rope type is incompatible with sequence position arithmetic")]
    IncompatibleRopeType,
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error("sequence {seq_id} could not be partially removed over positions [{p0}, {p1})")]
    PartialSequenceNotRemoved { seq_id: c_int, p0: c_int, p1: c_int },
    #[error("the llama.cpp library ran out of memory")]
    SeqAddLlamaCppOutOfMemory,
    #[error("not enough memory")]
    SeqAddNotEnoughMemory,
    #[error("{message}")]
    SeqAddReported { message: String },
    #[error("the context has no memory module attached")]
    SeqCpMemoryHandleUnavailable,
    #[error("the llama.cpp library ran out of memory")]
    SeqDivLlamaCppOutOfMemory,
    #[error("not enough memory")]
    SeqDivNotEnoughMemory,
    #[error("{message}")]
    SeqDivReported { message: String },
    #[error("the llama.cpp library ran out of memory")]
    SeqPosMaxLlamaCppOutOfMemory,
    #[error("not enough memory")]
    SeqPosMaxNotEnoughMemory,
    #[error("{message}")]
    SeqPosMaxReported { message: String },
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("provided sequence id is too large for an i32")]
    SequenceIdTooLarge(#[source] TryFromIntError),
    #[error("provided start position is too large for an i32")]
    StartPositionTooLarge(#[source] TryFromIntError),
    #[error(transparent)]
    UnknownRopeType(#[from] ModelError),
}
