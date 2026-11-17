use std::ffi::c_int;
use std::num::TryFromIntError;

use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_model::model_error::ModelError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextMemoryError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("provided end position is too large for an i32")]
    SeqAddEndPositionTooLarge(#[source] TryFromIntError),
    #[error("model rope type is incompatible with sequence position arithmetic")]
    SeqAddIncompatibleRopeType,
    #[error("the llama.cpp library ran out of memory")]
    SeqAddLlamaCppOutOfMemory,
    #[error("context has no memory module available")]
    SeqAddMemoryHandleUnavailable,
    #[error("not enough memory")]
    SeqAddNotEnoughMemory,
    #[error("{message}")]
    SeqAddReported { message: String },
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqAddSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("provided start position is too large for an i32")]
    SeqAddStartPositionTooLarge(#[source] TryFromIntError),
    #[error(transparent)]
    SeqAddUnknownRopeType(ModelError),
    #[error("provided end position is too large for an i32")]
    SeqCpEndPositionTooLarge(#[source] TryFromIntError),
    #[error("the context has no memory module attached")]
    SeqCpMemoryHandleUnavailable,
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqCpSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("provided start position is too large for an i32")]
    SeqCpStartPositionTooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    SeqDivEndPositionTooLarge(#[source] TryFromIntError),
    #[error("model rope type is incompatible with sequence position arithmetic")]
    SeqDivIncompatibleRopeType,
    #[error("the llama.cpp library ran out of memory")]
    SeqDivLlamaCppOutOfMemory,
    #[error("context has no memory module available")]
    SeqDivMemoryHandleUnavailable,
    #[error("not enough memory")]
    SeqDivNotEnoughMemory,
    #[error("{message}")]
    SeqDivReported { message: String },
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqDivSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("provided start position is too large for an i32")]
    SeqDivStartPositionTooLarge(#[source] TryFromIntError),
    #[error(transparent)]
    SeqDivUnknownRopeType(ModelError),
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqKeepSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("the llama.cpp library ran out of memory")]
    SeqPosMaxLlamaCppOutOfMemory,
    #[error("context has no memory module available")]
    SeqPosMaxMemoryHandleUnavailable,
    #[error("not enough memory")]
    SeqPosMaxNotEnoughMemory,
    #[error("{message}")]
    SeqPosMaxReported { message: String },
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqPosMaxSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("provided end position is too large for an i32")]
    SeqRmEndPositionTooLarge(#[source] TryFromIntError),
    #[error("sequence {seq_id} could not be partially removed over positions [{p0}, {p1})")]
    SeqRmPartialSequenceNotRemoved { seq_id: c_int, p0: c_int, p1: c_int },
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqRmSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("provided sequence id is too large for an i32")]
    SeqRmSequenceIdTooLarge(#[source] TryFromIntError),
    #[error("provided start position is too large for an i32")]
    SeqRmStartPositionTooLarge(#[source] TryFromIntError),
}
