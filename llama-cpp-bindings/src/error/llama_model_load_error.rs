use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum LlamaModelLoadError {
    #[error(transparent)]
    FfiStatus(#[from] llama_cpp_ffi_status::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] llama_cpp_ffi_status::FfiContractError),
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
    #[error("model file not found: {0}")]
    FileNotFound(PathBuf),
    #[error("model could not be loaded")]
    Unloadable,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
