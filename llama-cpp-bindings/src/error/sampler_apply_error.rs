#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SamplerApplyError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("the sampler pointer was null when applying to the token data array")]
    NullSampler,
    #[error("the sampler ran out of memory while applying to the token data array")]
    NotEnoughMemory,
    #[error("the llama.cpp sampler ran out of memory")]
    LlamaCppOutOfMemory,
    #[error(
        "the llama.cpp sampler threw a C++ exception while applying to the token data array: {message}"
    )]
    Reported { message: String },
}
