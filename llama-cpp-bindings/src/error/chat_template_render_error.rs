use std::ffi::NulError;

use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplateRenderError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("template globals contain an interior NUL byte: {0}")]
    GlobalsContainNulByte(#[from] NulError),
    #[error("template globals are not valid JSON: {message}")]
    InvalidJson { message: String },
    #[error("template globals are not a JSON object")]
    GlobalsNotAJsonObject,
    #[error("template is invalid: {message}")]
    InvalidTemplate { message: String },
    #[error("template could not be executed: {message}")]
    ExecutionFailed { message: String },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
