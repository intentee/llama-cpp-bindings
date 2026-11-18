use llama_cpp_path_c_string::path_c_string_error::PathCStringError;
use llama_cpp_wrapper_string::wrapper_string_error::WrapperStringError;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextStateError {
    #[error(transparent)]
    WrapperString(#[from] WrapperStringError),
    #[error("{operation} returned unknown FFI status {code}")]
    UnknownWrapperStatus { operation: &'static str, code: i64 },
    #[error("{operation} returned an invalid FFI response: was given a null {argument} argument")]
    WrapperRefusedNullArgument {
        operation: &'static str,
        argument: &'static str,
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
