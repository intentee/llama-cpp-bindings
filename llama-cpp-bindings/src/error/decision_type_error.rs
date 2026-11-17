use llama_cpp_ffi_status::ffi_error::FfiError;

use crate::error::unknown_decision_type::UnknownDecisionType;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecisionTypeError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error(transparent)]
    UnknownDecisionType(#[from] UnknownDecisionType),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
