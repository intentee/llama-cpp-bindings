use std::ffi::NulError;
use std::path::PathBuf;

use crate::gguf_type::GgufType;

#[derive(Debug, thiserror::Error)]
pub enum GgufContextError {
    #[error("Failed to initialize GGUF context from file: {0}")]
    InitFailed(PathBuf),

    #[error("Key not found in GGUF context: {key}")]
    KeyNotFound { key: String },

    #[error("GGUF key id {key_id} is outside of the {n_kv} keys the file holds")]
    KeyIdOutOfRange { key_id: i64, n_kv: i64 },

    #[error("null byte in string: {0}")]
    NulError(#[from] NulError),

    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    #[error("GGUF tensor {name} holds ggml type {ggml_type}, not F32")]
    TensorIsNotF32 { name: String, ggml_type: u32 },

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

    #[error("GGUF value is not valid UTF-8: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("GGUF key id {key_id} holds a {actual:?} value, not {expected:?}")]
    ValueTypeMismatch {
        key_id: i64,
        expected: GgufType,
        actual: GgufType,
    },
}
