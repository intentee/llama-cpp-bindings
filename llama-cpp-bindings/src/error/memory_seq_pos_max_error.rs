use llama_cpp_ffi_status::ffi_error::FfiError;

use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MemorySeqPosMaxError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error(transparent)]
    SequenceIdOutOfRange(#[from] SequenceIdOutOfRange),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
