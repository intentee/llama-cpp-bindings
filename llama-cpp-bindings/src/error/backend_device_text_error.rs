use std::str::Utf8Error;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum BackendDeviceTextError {
    #[error("ggml reported no text")]
    Null,
    #[error("ggml reported text that is not valid UTF-8")]
    NotUtf8(#[from] Utf8Error),
}
