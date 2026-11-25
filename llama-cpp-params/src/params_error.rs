use std::ffi::CString;
use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;
use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParamsError {
    #[error("Backend device {0} not found")]
    BackendDeviceNotFound(usize),
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("parameter fitting aborted")]
    FitAborted,
    #[error("the extra model path cannot be passed to llama.cpp")]
    FitExtraModelPathInvalid(#[source] PathCStringError),
    #[error(
        "the extra model context has {n_seq_max} sequences that need an output each, but a batch holds at most {output_capacity} outputs"
    )]
    FitExtraModelSequencesExceedOutputCapacity {
        n_seq_max: u32,
        output_capacity: u32,
    },
    #[error("the llama.cpp library ran out of memory")]
    FitLlamaCppOutOfMemory,
    #[error("{actual} memory margins were given, one per device needs exactly {expected}")]
    FitMarginsDeviceCountMismatch { expected: usize, actual: usize },
    #[error("not enough memory")]
    FitNotEnoughMemory,
    #[error("the model path cannot be passed to llama.cpp")]
    FitModelPathInvalid(#[source] PathCStringError),
    #[error("{message}")]
    FitReported { message: String },
    #[error(
        "{n_seq_max} sequences need an output each, but a batch holds at most {output_capacity} outputs"
    )]
    FitSequencesExceedOutputCapacity {
        n_seq_max: u32,
        output_capacity: u32,
    },
    #[error("Max devices exceeded. Max devices is {0}")]
    MaxDevicesExceeded(usize),
    #[error("No available slot in override vector")]
    NoAvailableOverrideSlot,
    #[error("no parameter combination fits available memory")]
    NoFittingMemoryLayout,
    #[error("the key needs {length_with_nul} bytes with its NUL, but llama.cpp holds {capacity}")]
    OverrideKeyTooLong {
        length_with_nul: usize,
        capacity: usize,
    },
    #[error("Override slot is not empty")]
    OverrideSlotNotEmpty,
    #[error("the string value of override {key:?} holds no NUL within the bytes llama.cpp reads")]
    OverrideTextNotNulTerminated { key: CString },
    #[error("attention type {0} is not a value llama.cpp defines")]
    UnknownAttentionType(llama_cpp_bindings_sys::bindings::llama_attention_type),
    #[error("parameter fitting returned an unknown status code: {code}")]
    UnknownFitStatus { code: i32 },
    #[error("unknown KV override tag: {0}")]
    UnknownKvOverrideTag(llama_cpp_bindings_sys::bindings::llama_model_kv_override_type),
    #[error("unknown llama lazy mode {value}")]
    UnknownLazyMode { value: i64 },
    #[error("unknown llama load mode {value}")]
    UnknownLoadMode { value: i32 },
    #[error("pooling type {0} is not a value llama.cpp defines")]
    UnknownPoolingType(llama_cpp_bindings_sys::bindings::llama_pooling_type),
    #[error("rope scaling type {0} is not a value llama.cpp defines")]
    UnknownRopeScalingType(llama_cpp_bindings_sys::bindings::llama_rope_scaling_type),
    #[error("unknown split mode value: {value}")]
    UnknownSplitMode {
        value: llama_cpp_bindings_sys::bindings::llama_split_mode,
    },
}

impl ExceptionMessageError for ParamsError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}
