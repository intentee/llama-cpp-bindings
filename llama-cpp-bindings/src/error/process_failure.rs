use llama_cpp_ffi_status::ffi_error::FfiError;

/// How `llama_rs_process` failed before llama.cpp returned a code of its own.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ProcessFailure {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
