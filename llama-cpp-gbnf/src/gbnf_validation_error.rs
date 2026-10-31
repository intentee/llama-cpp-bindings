use std::ffi::NulError;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfValidationError {
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
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
    #[error(
        "grammar is left-recursive or chains rules deeper than llama.cpp's recursion limit, so llama.cpp cannot compile it"
    )]
    UnsupportedGrammar,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("the llama.cpp grammar engine failed: {message}")]
    Reported { message: String },
}
