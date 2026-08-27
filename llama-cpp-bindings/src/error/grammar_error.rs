use std::ffi::NulError;

use crate::SamplingError;
use crate::error::token_to_string_error::TokenToStringError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GrammarError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("sampler initialization failed: {0}")]
    SamplerInitialization(#[from] SamplingError),
    #[error("the approximate token environment could not be built: {0}")]
    TokEnvUnavailable(#[from] TokenToStringError),
    #[error("grammar root not found in grammar string")]
    RootNotFound,
    #[error("the grammar was rejected by the GBNF parser: {0}")]
    GrammarRejected(#[source] llama_cpp_gbnf::gbnf_validation_error::GbnfValidationError),
    #[error("the grammar string contains an interior NUL byte")]
    GrammarContainsNul(#[source] NulError),
    #[error("a lazy-grammar trigger pattern contains an interior NUL byte")]
    TriggerPatternContainsNul(#[source] NulError),
    #[error("a DRY sequence breaker contains an interior NUL byte")]
    SequenceBreakerContainsNul(#[source] NulError),
    #[error("the llguidance parser factory could not be created: {message}")]
    LlguidanceFactoryUnavailable { message: String },
    #[error("the llguidance grammar could not be parsed: {message}")]
    LlguidanceGrammarInvalid { message: String },
    #[error("the llguidance parser could not be created for the grammar: {message}")]
    LlguidanceParserUnavailable { message: String },
    #[error("grammar is malformed")]
    GrammarMalformed,
    #[error("lazy grammar is malformed")]
    LazyGrammarMalformed,
    #[error("trigger pattern is not a valid regex: {message}")]
    InvalidTriggerPattern { message: String },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
