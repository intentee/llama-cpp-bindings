use std::os::raw::c_int;
use std::string::FromUtf8Error;

use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenToStringError {
    #[error(transparent)]
    TokenOutsideVocabulary(#[from] TokenOutsideVocabulary),
    #[error("Unknown Token Type")]
    UnknownTokenType,
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(c_int),
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    #[error("Integer conversion error: {0}")]
    IntConversionError(#[from] std::num::TryFromIntError),
}
