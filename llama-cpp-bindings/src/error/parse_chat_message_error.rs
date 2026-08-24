use std::string::FromUtf8Error;

use crate::error::marker_detection_error::MarkerDetectionError;
use crate::error::tool_call_format_failure::ToolCallFormatFailure;

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
    #[error("chat-template parse failed: {message}")]
    ParseFailed { message: String },
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
    #[error("could not serialize tools to JSON: {0}")]
    ToolsSerialization(String),
    #[error("tool-call format parser failed: {0}")]
    ToolCallFormat(#[from] ToolCallFormatFailure),
    #[error("reasoning-marker detection failed: {0}")]
    MarkerDetection(#[from] MarkerDetectionError),
    #[error("{message}")]
    Reported { message: String },
}
