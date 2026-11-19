use std::ffi::NulError;
use std::num::TryFromIntError;
use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_gbnf::gbnf_error::GbnfError;
use llama_cpp_model::model_error::ModelError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplerError {
    #[error("the grammar sampler callback failed during accept: {message}")]
    AcceptGrammarCallbackFailed { message: String },
    #[error("the llama.cpp library ran out of memory")]
    AcceptLlamaCppOutOfMemory,
    #[error("not enough memory")]
    AcceptNotEnoughMemory,
    #[error("the llama.cpp sampler ran out of memory")]
    ApplyLlamaCppOutOfMemory,
    #[error("the sampler ran out of memory while applying to the token data array")]
    ApplyNotEnoughMemory,
    #[error(
        "the llama.cpp sampler threw a C++ exception while applying to the token data array: {message}"
    )]
    ApplyReported { message: String },
    #[error("the candidates could not be exchanged with the llama.cpp sampler")]
    CandidateExchange(#[source] ModelError),
    #[error("DRY sequence breaker {index} contains an interior NUL byte")]
    DrySequenceBreakerContainsNul {
        index: usize,
        #[source]
        source: NulError,
    },
    #[error("DRY sequence breaker {index} is empty")]
    DrySequenceBreakerEmpty { index: usize },
    #[error(
        "DRY sequence breaker {index} is {length} bytes long, more than the {max_length} bytes llama.cpp uses"
    )]
    DrySequenceBreakerTooLong {
        index: usize,
        length: usize,
        max_length: usize,
    },
    #[error("the regex-escaped trigger word is not valid UTF-8")]
    EscapedPatternNotUtf8 {
        #[source]
        source: FromUtf8Error,
    },
    #[error("the sampler excluded every candidate, leaving no token to select")]
    EveryCandidateExcluded,
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("the grammar string contains an interior NUL byte")]
    GrammarContainsNul(#[source] NulError),
    #[error("the grammar parser failed a step without stopping: {message}")]
    GrammarFailedWithoutStopping { message: String },
    #[error("the grammar parser reached an internal error state: {message}")]
    GrammarInternalParserError { message: String },
    #[error("the grammar lexer became too complex: {message}")]
    GrammarLexerTooComplex { message: String },
    #[error("the llama.cpp library ran out of memory")]
    GrammarLlamaCppOutOfMemory,
    #[error("grammar is malformed")]
    GrammarMalformed,
    #[error("the grammar parser exhausted its maximum token budget: {message}")]
    GrammarMaxTokensReached { message: String },
    #[error("not enough memory")]
    GrammarNotEnoughMemory,
    #[error("the grammar parser panicked during {operation}: {message}")]
    GrammarParserPanicked {
        operation: &'static str,
        message: String,
    },
    #[error("the grammar parser became too complex: {message}")]
    GrammarParserTooComplex { message: String },
    #[error("the grammar was rejected by the GBNF parser")]
    GrammarRejected(#[source] GbnfError),
    #[error("{message}")]
    GrammarReported { message: String },
    #[error("sampler initialization failed")]
    GrammarSamplerInitialization(#[source] Box<Self>),
    #[error("grammar state corrupted during accept: {message}")]
    GrammarStateCorrupted { message: String },
    #[error("the llama.cpp library ran out of memory initializing the {sampler} sampler")]
    InitLlamaCppOutOfMemory { sampler: &'static str },
    #[error("not enough memory to report why the sampler could not be initialized")]
    InitNotEnoughMemory,
    #[error("initializing the {sampler} sampler failed: {message}")]
    InitReported {
        sampler: &'static str,
        message: String,
    },
    #[error("a value does not fit into i32")]
    IntegerOverflow(#[source] TryFromIntError),
    #[error("trigger pattern is not a valid regex: {message}")]
    InvalidTriggerPattern { message: String },
    #[error("lazy grammar is malformed")]
    LazyGrammarMalformed,
    #[error("the llguidance parser factory could not be created: {message}")]
    LlguidanceFactoryUnavailable { message: String },
    #[error("the llguidance grammar could not be parsed: {message}")]
    LlguidanceGrammarInvalid { message: String },
    #[error("the llguidance parser could not be created for the grammar: {message}")]
    LlguidanceParserUnavailable { message: String },
    #[error("the logits to sample from are unavailable")]
    Logits(#[source] ContextError),
    #[error("the grammar root name contains an interior NUL byte")]
    RootContainsNul(#[source] NulError),
    #[error("grammar root not found in grammar string")]
    RootNotFound,
    #[error("the grammar sampler callback failed during sampling: {message}")]
    SampleGrammarCallbackFailed { message: String },
    #[error("the llama.cpp library ran out of memory")]
    SampleLlamaCppOutOfMemory,
    #[error("not enough memory")]
    SampleNotEnoughMemory,
    #[error("{message}")]
    SampleReported { message: String },
    #[error("{sampler} sampler could not be initialized")]
    SamplerUnavailable { sampler: &'static str },
    #[error("the approximate token environment could not be built")]
    TokEnvUnavailable(#[source] ModelError),
    #[error(
        "{distinct} distinct DRY sequence breakers were given, more than the {max} llama.cpp uses"
    )]
    TooManyDistinctDrySequenceBreakers { distinct: usize, max: usize },
    #[error("a lazy-grammar trigger pattern contains an interior NUL byte")]
    TriggerPatternContainsNul(#[source] NulError),
}

impl ExceptionMessageError for SamplerError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}
