use std::ffi::NulError;
use std::path::PathBuf;
use std::str::Utf8Error;

use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

use crate::gguf_type::GgufType;

#[derive(Debug, thiserror::Error)]
pub enum GgufContextError {
    #[error("GGUF key id {key_id} holds an array of type {actual_raw_type}, not {expected:?}")]
    ArrayElementTypeMismatch {
        key_id: i64,
        expected: GgufType,
        actual_raw_type: u32,
    },

    #[error("Failed to initialize GGUF context from file: {0}")]
    InitFailed(PathBuf),

    #[error("Failed to write the rewritten GGUF file to {path}")]
    DestinationFileWriteFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("the GGUF key to look up contains a NUL byte")]
    KeyContainsNulByte(#[source] NulError),

    #[error("GGUF key id {key_id} has a name that is not valid UTF-8")]
    KeyNameNotUtf8 {
        key_id: i64,
        #[source]
        source: Utf8Error,
    },

    #[error("Key not found in GGUF context: {key}")]
    KeyNotFound { key: String },

    #[error("GGUF key id {key_id} is outside of the {n_kv} keys the file holds")]
    KeyIdOutOfRange { key_id: i64, n_kv: i64 },

    #[error("the GGUF file path cannot be handed to ggml")]
    LoadPathInvalid(#[source] PathCStringError),

    #[error("Failed to read the GGUF file {path} this context was loaded from")]
    SourceFileReadFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "GGUF file {path} holds {file_size} bytes, but its tensor data begins at byte {data_offset}"
    )]
    SourceFileTruncated {
        path: PathBuf,
        data_offset: usize,
        file_size: usize,
    },

    #[error("GGUF key id {key_id} holds a string value that is not valid UTF-8")]
    StringValueNotUtf8 {
        key_id: i64,
        #[source]
        source: Utf8Error,
    },

    #[error("GGUF tensor {name} holds ggml type {ggml_type}, not F32")]
    TensorIsNotF32 { name: String, ggml_type: u32 },

    #[error("the GGUF tensor name to look up contains a NUL byte")]
    TensorNameContainsNulByte(#[source] NulError),

    #[error("Tensor not found in GGUF context: {name}")]
    TensorNotFound { name: String },

    #[error("Failed to read GGUF tensor {name} from {path}")]
    TensorReadFailed {
        name: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("GGUF key id {key_id} holds a value of unknown type {raw_type}")]
    UnknownValueType { key_id: i64, raw_type: u32 },

    #[error("GGUF key id {key_id} holds a {actual:?} value, not {expected:?}")]
    ValueTypeMismatch {
        key_id: i64,
        expected: GgufType,
        actual: GgufType,
    },
}
