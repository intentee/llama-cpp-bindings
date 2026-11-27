use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;

#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    #[error("llama.cpp formatted answers that are not decision answers")]
    AnswersAreNotDecisionAnswers(#[source] serde_json::Error),
    #[error("the llama.cpp library ran out of memory while reading answers")]
    AnswersLlamaCppOutOfMemory,
    #[error("not enough memory to report how llama.cpp failed to read answers")]
    AnswersNotEnoughMemory,
    #[error("llama.cpp threw while reading answers: {message}")]
    AnswersReported { message: String },
    #[error("llama.cpp rejected the scores of a question: {message}")]
    AnswersScoresRejected { message: String },
    #[error("the answers llama.cpp formatted are not valid UTF-8")]
    AnswersTextNotUtf8(#[source] FromUtf8Error),
    #[error("the model declares a decision type that llama.cpp does not support")]
    DecisionTypeUnsupportedByLlamaCpp,
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("the request carries media, but the decision model reads no media")]
    MediaUnsupportedByDecisionModel,
    #[error("the request carries media, but no multimodal projector was given")]
    MediaRequiresAProjector,
    #[error("the llama.cpp library ran out of memory while loading the decision model")]
    ModelLlamaCppOutOfMemory,
    #[error("the model is not a decision model")]
    ModelIsNotADecisionModel,
    #[error("not enough memory to report how llama.cpp rejected the decision model")]
    ModelNotEnoughMemory,
    #[error("llama.cpp rejected the decision model: {message}")]
    ModelRejected { message: String },
    #[error("the outputs of the decoded prompt cannot be read: {message}")]
    OutputsUnreadable { message: String },
    #[error("the llama.cpp library ran out of memory while laying out a prompt")]
    PromptLlamaCppOutOfMemory,
    #[error("not enough memory to report how llama.cpp failed to lay out a prompt")]
    PromptNotEnoughMemory,
    #[error("llama.cpp rejected the decision request while laying out a prompt: {message}")]
    PromptRejected { message: String },
    #[error("llama.cpp threw while laying out a prompt: {message}")]
    PromptReported { message: String },
    #[error("llama.cpp ran out of memory while finding the tokens the prompts of a decision share")]
    PromptSharingLlamaCppOutOfMemory,
    #[error("llama.cpp could not read the decision request as JSON: {message}")]
    RequestBodyIsNotJson { message: String },
    #[error("the llama.cpp library ran out of memory while reading the decision request")]
    RequestLlamaCppOutOfMemory,
    #[error("not enough memory to report how llama.cpp rejected the decision request")]
    RequestNotEnoughMemory,
    #[error("llama.cpp rejected the decision request: {message}")]
    RequestRejected { message: String },
    #[error("llama.cpp threw while reading the decision request: {message}")]
    RequestReported { message: String },
}

impl ExceptionMessageError for DecisionError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}
