#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelProvidedToolCallIdError {
    #[error("a tool call id provided by the model must not be empty")]
    Empty,
}
