use std::ffi::CString;

use llama_cpp_ffi_status::ffi_error::FfiError;

use super::mtmd_context_params_error::MtmdContextParamsError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdMemoryUsageError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error(transparent)]
    ContextParams(#[from] MtmdContextParamsError),
    #[error("the memory usage of mmproj {path:?} could not be measured, because it does not load")]
    Unloadable { path: CString },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
