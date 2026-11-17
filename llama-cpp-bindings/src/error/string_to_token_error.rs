use std::ffi::NulError;

use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum StringToTokenError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    CIntConversionError(#[from] std::num::TryFromIntError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
