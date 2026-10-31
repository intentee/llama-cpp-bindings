#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MemorySeqPosMaxError {
    #[error(transparent)]
    FfiContract(#[from] llama_cpp_ffi_status::FfiContractError),
    #[error(transparent)]
    FfiStatus(#[from] llama_cpp_ffi_status::FfiStatusError),
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error("sequence id {seq_id} is outside the context sequence range")]
    SequenceIdOutOfRange { seq_id: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
