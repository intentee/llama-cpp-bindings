use std::ffi::NulError;
use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfError {
    #[error("grammar defines no rules")]
    EmptyRuleSet,
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("grammar string contains an interior NUL byte")]
    GrammarContainsNul(#[source] NulError),
    #[error("grammar converted from the JSON schema is not valid UTF-8")]
    GrammarNotUtf8 {
        #[source]
        source: FromUtf8Error,
    },
    #[error("JSON schema is not valid JSON: {message}")]
    InvalidJson { message: String },
    #[error("JSON schema is invalid: {message}")]
    InvalidSchema { message: String },
    #[error("grammar root name contains an interior NUL byte")]
    RootContainsNul(#[source] NulError),
    #[error("grammar does not define the root symbol {root:?}")]
    RootSymbolMissing { root: String },
    #[error("schema string contains an interior NUL byte")]
    SchemaContainsNulByte(#[source] NulError),
    #[error("the llama.cpp library ran out of memory")]
    SchemaConversionLlamaCppOutOfMemory,
    #[error("not enough memory")]
    SchemaConversionNotEnoughMemory,
    #[error("{message}")]
    SchemaConversionReported { message: String },
    #[error("grammar has a syntax error and could not be parsed")]
    SyntaxError,
    #[error(
        "grammar is left-recursive or chains rules deeper than llama.cpp's recursion limit, so llama.cpp cannot compile it"
    )]
    UnsupportedGrammar,
    #[error("the llama.cpp library ran out of memory")]
    ValidationLlamaCppOutOfMemory,
    #[error("not enough memory")]
    ValidationNotEnoughMemory,
    #[error("the llama.cpp grammar engine failed: {message}")]
    ValidationReported { message: String },
}

impl ExceptionMessageError for GbnfError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}
