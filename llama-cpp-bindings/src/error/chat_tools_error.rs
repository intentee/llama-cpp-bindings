use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum ChatToolsError {
    #[error("tools_json is not valid JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("tools_json must be a JSON array")]
    NotArray,
    #[error("tools_json contains an interior NUL byte")]
    ContainsNulByte(#[source] NulError),
}
