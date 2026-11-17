use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SamplerAcceptError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("grammar state corrupted during accept: {message}")]
    GrammarStateCorrupted { message: String },
    #[error("the grammar sampler callback failed during accept: {message}")]
    GrammarCallbackFailed { message: String },
}
