use std::ffi::NulError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfValidationError {
    #[error("llama_rs_validate_gbnf returned unknown FFI status {code}")]
    FfiStatus { code: i64 },
    #[error("llama_rs_validate_gbnf violated its FFI contract: {detail}")]
    FfiContract { detail: &'static str },
    #[error("grammar string contains an interior NUL byte")]
    GrammarContainsNul(#[source] NulError),
    #[error("grammar root name contains an interior NUL byte")]
    RootContainsNul(#[source] NulError),
    #[error("grammar has a syntax error and could not be parsed")]
    SyntaxError,
    #[error("grammar defines no rules")]
    EmptyRuleSet,
    #[error("grammar does not define the root symbol {root:?}")]
    RootSymbolMissing { root: String },
    #[error("grammar is left-recursive and cannot be compiled by llama.cpp")]
    LeftRecursion,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp grammar engine failed: {message}")]
    Reported { message: String },
}
