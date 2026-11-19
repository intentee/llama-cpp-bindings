use std::str::Utf8Error;
use std::string::FromUtf8Error;

use llama_cpp_model::model_error::ModelError;
use llama_cpp_sampler::sampler_error::SamplerError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatError {
    #[error(
        "llama.cpp attributed a span of the answer to a kind {value} this binding does not recognize"
    )]
    ChatSpanKindUnrecognized {
        value: llama_cpp_bindings_sys::bindings::llama_rs_chat_span_kind,
    },
    #[error("the exception message {operation} reported is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("the name of the chat format llama.cpp expected is not valid UTF-8")]
    FormatNameNotUtf8(#[source] Utf8Error),
    #[error("the request's grammar sampler could not be built")]
    Grammar(#[source] SamplerError),
    #[error("the grammar sampler refused a generation-prompt token")]
    GrammarPrefill(#[source] SamplerError),
    #[error(
        "llama.cpp produced a grammar trigger of a type {value} this binding does not recognize"
    )]
    GrammarTriggerTypeUnrecognized {
        value: llama_cpp_bindings_sys::bindings::llama_rs_chat_grammar_trigger_type,
    },
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
}
