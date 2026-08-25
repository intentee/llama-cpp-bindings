use std::string::FromUtf8Error;

use crate::error::marker_detection_error::MarkerDetectionError;

#[derive(Debug, thiserror::Error)]
pub enum ParseChatMessageError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("model has no chat template")]
    NoChatTemplate,
    #[error("model has no vocab")]
    NoVocab,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("the chat parser could not be constructed: {message}")]
    ParserCreationFailed { message: String },
    #[error("the chat parser did not recognize the message: {message}")]
    MessageUnrecognized { message: String },
    #[error("the chat parser destructor threw: {message}")]
    DestructorFailed { message: String },
    #[error("tool-call id index {index} out of bounds")]
    ToolCallIdIndexOutOfBounds { index: usize },
    #[error("tool-call name index {index} out of bounds")]
    ToolCallNameIndexOutOfBounds { index: usize },
    #[error("tool-call arguments index {index} out of bounds")]
    ToolCallArgumentsIndexOutOfBounds { index: usize },
    #[error("ffi returned non-utf8 string: {0}")]
    StringUtf8Error(#[from] FromUtf8Error),
    #[error("tools_json is not valid JSON: {0}")]
    ToolsJsonInvalid(#[source] serde_json::Error),
    #[error("tools_json must be a JSON array")]
    ToolsJsonNotArray,
    #[error("tools_json contains an interior NUL byte")]
    ToolsJsonContainsNulByte(#[source] std::ffi::NulError),
    #[error("the message to parse contains an interior NUL byte")]
    InputContainsNulByte(#[source] std::ffi::NulError),
    #[error("reasoning-marker detection failed: {0}")]
    MarkerDetection(#[from] MarkerDetectionError),
    #[error("{message}")]
    Reported { message: String },
}
