use std::str::Utf8Error;

use llama_cpp_ffi_status::ffi_error::FfiError;

use crate::error::grammar_error::GrammarError;
use crate::error::sampler_accept_error::SamplerAcceptError;
use crate::error::string_to_token_error::StringToTokenError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatSessionError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("the chat template cannot serve the request: {message}")]
    TemplateCannotServeRequest { message: String },
    #[error("text the chat session holds is not valid UTF-8")]
    SessionTextNotUtf8(#[source] Utf8Error),
    #[error("the rendered prompt could not be tokenized: {0}")]
    PromptTokenization(#[from] StringToTokenError),
    #[error("the request's grammar sampler could not be built: {0}")]
    Grammar(#[from] GrammarError),
    #[error("the grammar sampler refused a generation-prompt token: {0}")]
    GrammarPrefill(#[from] SamplerAcceptError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
