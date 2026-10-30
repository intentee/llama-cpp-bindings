use std::str::Utf8Error;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseChatMessageError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("the parsed chat destructor threw: {message}")]
    DestructorFailed { message: String },
    #[error("text llama.cpp parsed out of the answer is not valid UTF-8: {0}")]
    ParsedTextNotUtf8(#[source] Utf8Error),
    #[error("{message}")]
    Reported { message: String },
}
