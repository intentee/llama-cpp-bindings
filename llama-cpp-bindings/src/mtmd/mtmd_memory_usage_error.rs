use std::ffi::CString;

use super::mtmd_context_params_error::MtmdContextParamsError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdMemoryUsageError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
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
