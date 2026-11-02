use llama_cpp_ffi_status::cpp_string_error::CppStringError;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

use crate::context::context_limits::ContextLimits;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum BatchTransferError {
    #[error(transparent)]
    CppString(#[from] CppStringError),
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error("the batch was built for context limits {batch:?}, but the context has {context:?}")]
    LimitsMismatch {
        batch: ContextLimits,
        context: ContextLimits,
    },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
