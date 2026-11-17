use std::str::Utf8Error;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum FfiError {
    #[error("{operation} returned unknown FFI status {code}")]
    UnknownStatus { operation: &'static str, code: i64 },
    #[error("{operation} returned an invalid FFI response: {detail}")]
    ContractViolated {
        operation: &'static str,
        detail: &'static str,
    },
    #[error("{operation} reported text that is not valid UTF-8")]
    TextNotUtf8 {
        operation: &'static str,
        bytes: Vec<u8>,
        #[source]
        source: Utf8Error,
    },
}
