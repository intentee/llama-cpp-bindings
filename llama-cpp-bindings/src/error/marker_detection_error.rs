use std::str::Utf8Error;
use std::string::FromUtf8Error;

use crate::error::chat_template_error::ChatTemplateError;
use crate::error::string_to_token_error::StringToTokenError;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum MarkerDetectionError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("ffi returned non-utf8 marker bytes: {0}")]
    MarkerUtf8Error(#[from] FromUtf8Error),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{operation} could not run because the model has no chat template")]
    ModelHasNoChatTemplate { operation: &'static str },
    #[error("{operation} could not run because the model has no vocab")]
    ModelHasNoVocab { operation: &'static str },
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("reasoning-marker detection failed: {message}")]
    ReasoningMarkerDetectionFailed { message: String },
    #[error("tool-call haystack computation failed: {message}")]
    ToolCallHaystackComputationFailed { message: String },
    #[error("tool-call synthetic-render diagnosis failed: {message}")]
    ToolCallSyntheticRenderDiagnosisFailed { message: String },
    #[error("the reasoning-markers destructor threw: {message}")]
    ReasoningMarkersFreeFailed { message: String },
    #[error("a detected marker string could not be tokenised: {0}")]
    MarkerTokenizationFailed(#[from] StringToTokenError),
    #[error("the chat template is not valid UTF-8: {0}")]
    ToolCallTemplateNotUtf8(#[from] Utf8Error),
    #[error("the chat template could not be retrieved for tool-call marker detection: {0}")]
    ChatTemplateUnavailable(#[source] ChatTemplateError),
    #[error("{operation} rejected the Rust-owned argument {argument}")]
    WrapperRejectedArgument {
        operation: &'static str,
        argument: &'static str,
    },
}
