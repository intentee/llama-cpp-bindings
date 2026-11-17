use std::str::Utf8Error;

use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseChatMessageError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("text llama.cpp parsed out of the answer is not valid UTF-8: {0}")]
    ParsedTextNotUtf8(#[source] Utf8Error),
    #[error("the name of the chat format llama.cpp expected is not valid UTF-8: {0}")]
    FormatNameNotUtf8(#[source] Utf8Error),
    #[error("{message}")]
    Reported { message: String },
}
