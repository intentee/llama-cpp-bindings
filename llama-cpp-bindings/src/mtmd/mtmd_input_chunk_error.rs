use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdInputChunkError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
