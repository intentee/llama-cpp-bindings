use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum ChatToolsError {
    #[error("chat tools are not valid JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("chat tools must be a JSON array")]
    NotArray,
    #[error("chat tools contain an interior NUL byte at position {}", .0.nul_position())]
    ContainsNulByte(#[source] NulError),
}
