use std::ffi::NulError;
use std::path::PathBuf;

use llama_cpp_ffi_status::ffi_error::FfiError;

use crate::error::path_c_string_error::PathCStringError;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum LlamaModelLoadError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
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

impl From<PathCStringError> for LlamaModelLoadError {
    fn from(error: PathCStringError) -> Self {
        match error {
            PathCStringError::NotUtf8(path) => Self::PathToStrError(path),
            PathCStringError::ContainsNulByte(nul_error) => Self::NullError(nul_error),
        }
    }
}
