use std::num::TryFromIntError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MemorySeqAddError {
    #[error(transparent)]
    FfiStatus(#[from] llama_cpp_ffi_status::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] llama_cpp_ffi_status::FfiContractError),
    #[error("provided start position is too large for an i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("model rope type is incompatible with sequence position arithmetic")]
    IncompatibleRopeType,
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
