use llama_cpp_ffi_status::cpp_string_error::CppStringError;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MemorySeqPosMaxError {
    #[error(transparent)]
    CppString(#[from] CppStringError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error(transparent)]
    SequenceIdOutOfRange(#[from] SequenceIdOutOfRange),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
