use std::ffi::NulError;
use std::string::FromUtf8Error;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum JsonSchemaToGrammarError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("schema string contains an interior NUL byte: {0}")]
    SchemaContainsNulByte(#[from] NulError),
    #[error("JSON schema is invalid: {message}")]
    InvalidSchema { message: String },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
    #[error("grammar returned by json_schema_to_grammar is not valid UTF-8")]
    GrammarNotUtf8(#[from] FromUtf8Error),
}
