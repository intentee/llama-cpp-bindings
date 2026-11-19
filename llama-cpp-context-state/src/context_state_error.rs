use std::string::FromUtf8Error;

use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextStateError {
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("the sequence state file path to load cannot be handed to llama.cpp")]
    LoadSeqStatePathInvalid(#[source] PathCStringError),
    #[error("the session file path to load cannot be handed to llama.cpp")]
    LoadSessionPathInvalid(#[source] PathCStringError),
    #[error("the sequence state file path to save cannot be handed to llama.cpp")]
    SaveSeqStatePathInvalid(#[source] PathCStringError),
    #[error("the session file path to save cannot be handed to llama.cpp")]
    SaveSessionPathInvalid(#[source] PathCStringError),
    #[error("Failed to load sequence state file")]
    SeqStateFileNotLoaded,
    #[error("Failed to save sequence state file")]
    SeqStateFileNotSaved,
    #[error("Failed to load session file")]
    SessionFileNotLoaded,
    #[error("Failed to save session file")]
    SessionFileNotSaved,
    #[error("the llama.cpp library ran out of memory")]
    StateDataLlamaCppOutOfMemory,
    #[error(
        "the llama.cpp deserializer restored nothing from a {provided_bytes}-byte snapshot; \
         llama.cpp logs the cause and reports zero bytes rather than throwing"
    )]
    StateDataNothingRestored { provided_bytes: usize },
    #[error("not enough memory")]
    StateDataNotEnoughMemory,
    #[error("{message}")]
    StateDataReported { message: String },
}
