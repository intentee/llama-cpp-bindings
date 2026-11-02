use std::str::Utf8Error;

use crate::ffi_contract_error::FfiContractError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CppStringError {
    #[error(transparent)]
    Missing(#[from] FfiContractError),
    #[error("{operation} reported text that is not valid UTF-8: {source}")]
    NotUtf8 {
        operation: &'static str,
        bytes: Vec<u8>,
        #[source]
        source: Utf8Error,
    },
}
