use std::ffi::NulError;

use llama_cpp_ffi_status::cpp_string_error::CppStringError;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplateRenderError {
    #[error(transparent)]
    CppString(#[from] CppStringError),
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
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
