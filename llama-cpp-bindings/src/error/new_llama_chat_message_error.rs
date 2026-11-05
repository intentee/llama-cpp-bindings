use std::ffi::NulError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum NewLlamaChatMessageError {
    #[error("{0}")]
    NulError(#[from] NulError),
}
