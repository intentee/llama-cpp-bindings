use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("no parameter combination fits available memory")]
    NoFittingMemoryLayout,
    #[error("parameter fitting aborted")]
    Aborted,
    #[error("{actual} memory margins were given, one per device needs exactly {expected}")]
    MarginsDeviceCountMismatch { expected: usize, actual: usize },
    #[error("parameter fitting returned an unknown status code: {code}")]
    UnknownStatus { code: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
