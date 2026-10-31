use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum StateDataError {
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error(
        "the llama.cpp deserializer restored nothing from a {provided_bytes}-byte snapshot; \
         llama.cpp logs the cause and reports zero bytes rather than throwing"
    )]
    NothingRestored { provided_bytes: usize },
    #[error("{message}")]
    Reported { message: String },
}
