use std::ffi::NulError;

use crate::SamplingError;
use crate::error::token_to_string_error::TokenToStringError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GrammarError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error("sampler initialization failed: {0}")]
    SamplerInitialization(#[from] SamplingError),
    #[error("the approximate token environment could not be built: {0}")]
    TokEnvUnavailable(#[from] TokenToStringError),
    #[error("grammar root not found in grammar string")]
    RootNotFound,
    #[error("grammar string or root contains null bytes: {0}")]
    GrammarNullBytes(NulError),
    #[error("string contains null bytes: {0}")]
    NulError(#[from] NulError),
    #[error("integer overflow: {0}")]
    IntegerOverflow(String),
    #[error("llguidance error: {0}")]
    LlguidanceError(String),
    #[error("grammar is malformed")]
    GrammarMalformed,
    #[error("lazy grammar is malformed")]
    LazyGrammarMalformed,
    #[error("trigger pattern is not a valid regex: {message}")]
    InvalidTriggerPattern { message: String },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
