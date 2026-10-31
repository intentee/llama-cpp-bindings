use std::ffi::NulError;
use std::str::Utf8Error;

use crate::error::chat_templates_error::ChatTemplatesError;
use crate::error::grammar_error::GrammarError;
use crate::error::sampler_accept_error::SamplerAcceptError;
use crate::error::string_to_token_error::StringToTokenError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatSessionError {
    #[error(transparent)]
    FfiStatus(#[from] llama_cpp_ffi_status::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] llama_cpp_ffi_status::FfiContractError),
    #[error("the model's chat templates are unavailable: {0}")]
    Templates(#[from] ChatTemplatesError),
    #[error("a chat message is not valid UTF-8")]
    MessageNotUtf8(#[source] Utf8Error),
    #[error("the chat request contains an interior NUL byte")]
    RequestContainsNul(#[source] NulError),
    #[error("the messages are not valid JSON: {message}")]
    MessagesNotJson { message: String },
    #[error("the messages are not valid chat messages: {message}")]
    MessagesInvalid { message: String },
    #[error("the tools are not valid JSON: {message}")]
    ToolsNotJson { message: String },
    #[error("the tools are not valid tool definitions: {message}")]
    ToolsInvalid { message: String },
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
