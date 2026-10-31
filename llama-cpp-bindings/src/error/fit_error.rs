#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    #[error(transparent)]
    FfiStatus(#[from] llama_cpp_ffi_status::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] llama_cpp_ffi_status::FfiContractError),
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
