use std::str::Utf8Error;

use llama_cpp_model::model_error::ModelError;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_wrapper_string::wrapper_string_error::WrapperStringError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatError {
    #[error("the name of the chat format llama.cpp expected is not valid UTF-8")]
    FormatNameNotUtf8(#[source] Utf8Error),
    #[error("the request's grammar sampler could not be built")]
    Grammar(#[source] SamplerError),
    #[error("the grammar sampler refused a generation-prompt token")]
    GrammarPrefill(#[source] SamplerError),
    #[error("the llama.cpp library ran out of memory")]
    ParseLlamaCppOutOfMemory,
    #[error("not enough memory")]
    ParseNotEnoughMemory,
    #[error("{message}")]
    ParseReported { message: String },
    #[error("text llama.cpp parsed out of the answer is not valid UTF-8")]
    ParsedTextNotUtf8(#[source] Utf8Error),
    #[error("the rendered prompt could not be tokenized")]
    PromptTokenization(#[source] ModelError),
    #[error("the llama.cpp library ran out of memory")]
    SessionLlamaCppOutOfMemory,
    #[error("not enough memory")]
    SessionNotEnoughMemory,
    #[error("{message}")]
    SessionReported { message: String },
    #[error("text the chat session holds is not valid UTF-8")]
    SessionTextNotUtf8(#[source] Utf8Error),
    #[error("the chat template cannot serve the request: {message}")]
    TemplateCannotServeRequest { message: String },
    #[error("{operation} returned unknown FFI status {code}")]
    UnknownWrapperStatus { operation: &'static str, code: i64 },
    #[error("{operation} returned an invalid FFI response: was given a null {argument} argument")]
    WrapperRefusedNullArgument {
        operation: &'static str,
        argument: &'static str,
    },
    #[error(
        "{operation} returned an invalid FFI response: reported a failure without an error message"
    )]
    WrapperReportedFailureWithoutMessage { operation: &'static str },
    #[error(transparent)]
    WrapperString(#[from] WrapperStringError),
    #[error(
        "{operation} returned an invalid FFI response: success status contained a null {output}"
    )]
    WrapperSucceededWithoutOutput {
        operation: &'static str,
        output: &'static str,
    },
}
