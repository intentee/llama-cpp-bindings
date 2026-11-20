use std::ffi::c_int;
use std::num::TryFromIntError;
use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;
use llama_cpp_model::model_error::ModelError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextMemoryError {
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("provided end position is too large for an i32")]
    SeqAddEndPositionTooLarge(#[source] TryFromIntError),
    #[error("the highest position of the sequence to shift could not be read")]
    SeqAddHighestPositionUnreadable(#[source] Box<Self>),
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
    #[error(
        "shifting the range of positions up to {range_upper_bound} by {delta} can move a position past {max_position}, the highest position llama.cpp can hold in this context"
    )]
    SeqAddShiftsPositionPastMax {
        range_upper_bound: i32,
        delta: i32,
        max_position: i32,
    },
    #[error("provided start position is too large for an i32")]
    SeqAddStartPositionTooLarge(#[source] TryFromIntError),
    #[error(transparent)]
    SeqAddUnknownRopeType(ModelError),
    #[error(
        "copying up to position {p1} reaches past {max_position}, the highest position llama.cpp can hold in this context"
    )]
    SeqCpEndPositionPastMax { p1: i32, max_position: i32 },
    #[error("provided end position is too large for an i32")]
    SeqCpEndPositionTooLarge(#[source] TryFromIntError),
    #[error("the context has no memory module attached")]
    SeqCpMemoryHandleUnavailable,
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SeqCpSequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error(
        "copying from position {p0} starts past {max_position}, the highest position llama.cpp can hold in this context"
    )]
    SeqCpStartPositionPastMax { p0: i32, max_position: i32 },
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

impl ExceptionMessageError for ContextMemoryError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}
