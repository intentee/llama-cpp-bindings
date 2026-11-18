use std::ffi::NulError;
use std::string::FromUtf8Error;

use llama_cpp_wrapper_string::wrapper_string_error::WrapperStringError;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfError {
    #[error(transparent)]
    WrapperString(#[from] WrapperStringError),
    #[error("{operation} returned unknown FFI status {code}")]
    UnknownWrapperStatus { operation: &'static str, code: i64 },
    #[error("{operation} returned an invalid FFI response: was given a null {argument} argument")]
    WrapperRefusedNullArgument {
        operation: &'static str,
        argument: &'static str,
    },
    #[error("{operation} returned an invalid FFI response: {pointer} pointer was null")]
    WrapperRefusedNullPointer {
        operation: &'static str,
        pointer: &'static str,
    },
    #[error(
        "{operation} returned an invalid FFI response: success status contained a null {output}"
    )]
    WrapperSucceededWithoutOutput {
        operation: &'static str,
        output: &'static str,
    },
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
    ValidationNotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    ValidationLlamaCppOutOfMemory,
    #[error("the llama.cpp grammar engine failed: {message}")]
    ValidationReported { message: String },
    #[error("schema string contains an interior NUL byte")]
    SchemaContainsNulByte(#[source] NulError),
    #[error("JSON schema is invalid: {message}")]
    InvalidSchema { message: String },
    #[error("JSON schema is not valid JSON: {message}")]
    InvalidJson { message: String },
    #[error("not enough memory")]
    SchemaConversionNotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    SchemaConversionLlamaCppOutOfMemory,
    #[error("{message}")]
    SchemaConversionReported { message: String },
    #[error("grammar returned by json_schema_to_grammar is not valid UTF-8")]
    GrammarNotUtf8(#[source] FromUtf8Error),
}
