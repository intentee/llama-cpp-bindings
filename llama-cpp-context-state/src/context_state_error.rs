use std::ffi::NulError;
use std::path::PathBuf;

use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextStateError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("null byte in string")]
    LoadSeqStatePathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    LoadSeqStatePathNotUtf8(PathBuf),
    #[error("null byte in string")]
    LoadSessionPathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    LoadSessionPathNotUtf8(PathBuf),
    #[error("null byte in string")]
    SaveSeqStatePathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    SaveSeqStatePathNotUtf8(PathBuf),
    #[error("null byte in string")]
    SaveSessionPathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    SaveSessionPathNotUtf8(PathBuf),
    #[error("Failed to load sequence state file")]
    SeqStateFileNotLoaded,
    #[error("Failed to save sequence state file")]
    SeqStateFileNotSaved,
    #[error("Failed to load session file")]
    SessionFileNotLoaded,
    #[error("Failed to save session file")]
    SessionFileNotSaved,
    #[error("the llama.cpp library ran out of memory")]
    StateDataLlamaCppOutOfMemory,
    #[error(
        "the llama.cpp deserializer restored nothing from a {provided_bytes}-byte snapshot; \
         llama.cpp logs the cause and reports zero bytes rather than throwing"
    )]
    StateDataNothingRestored { provided_bytes: usize },
    #[error("not enough memory")]
    StateDataNotEnoughMemory,
    #[error("{message}")]
    StateDataReported { message: String },
}
