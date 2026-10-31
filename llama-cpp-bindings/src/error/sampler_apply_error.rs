use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SamplerApplyError {
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error("the sampler pointer was null when applying to the token data array")]
    NullSampler,
    #[error("the sampler excluded every candidate, leaving no token to select")]
    EveryCandidateExcluded,
    #[error("the sampler ran out of memory while applying to the token data array")]
    NotEnoughMemory,
    #[error("the llama.cpp sampler ran out of memory")]
    LlamaCppOutOfMemory,
    #[error(
        "the llama.cpp sampler threw a C++ exception while applying to the token data array: {message}"
    )]
    Reported { message: String },
}
