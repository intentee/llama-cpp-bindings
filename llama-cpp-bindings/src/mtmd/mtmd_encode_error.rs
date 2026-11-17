use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdEncodeError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("multimodal chunk encoding failed with code: {code}")]
    EncodingFailed { code: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
