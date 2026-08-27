use std::ffi::NulError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfValidationError {
    #[error(transparent)]
    FfiStatus(#[from] llama_cpp_ffi_status::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] llama_cpp_ffi_status::FfiContractError),
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
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("the llama.cpp grammar engine failed: {message}")]
    Reported { message: String },
}
