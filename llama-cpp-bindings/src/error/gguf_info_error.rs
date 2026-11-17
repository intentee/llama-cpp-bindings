use llama_cpp_ffi_status::cpp_string_error::CppStringError;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

use crate::error::path_c_string_error::PathCStringError;
use crate::error::unknown_decision_type::UnknownDecisionType;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum GgufInfoError {
    #[error(transparent)]
    CppString(#[from] CppStringError),
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error(transparent)]
    Path(#[from] PathCStringError),
    #[error(transparent)]
    UnknownDecisionType(#[from] UnknownDecisionType),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
