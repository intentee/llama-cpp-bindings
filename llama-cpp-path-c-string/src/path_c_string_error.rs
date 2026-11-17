use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PathCStringError {
    #[error("the path contains a NUL byte")]
    ContainsNulByte(#[source] NulError),
    #[error("the path {0} is not valid UTF-8")]
    NotUtf8(PathBuf),
}
