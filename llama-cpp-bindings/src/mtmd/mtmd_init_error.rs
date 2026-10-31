use std::path::PathBuf;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

use super::mtmd_context_params_error::MtmdContextParamsError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdInitError {
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error(transparent)]
    ContextParams(#[from] MtmdContextParamsError),
    #[error("Failed to create CString from mmproj path: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Mmproj path is not valid UTF-8: {0:?}")]
    PathToStrError(PathBuf),
    #[error("mmproj could not be loaded: {path:?}")]
    Unloadable { path: PathBuf },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
