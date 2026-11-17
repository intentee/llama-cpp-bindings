use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum PathCStringError {
    #[error("failed to convert path {0} to str")]
    NotUtf8(PathBuf),
    #[error("null byte in string {0}")]
    ContainsNulByte(#[from] NulError),
}
