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
    #[error("the vendored sampler ran out of memory")]
    VendoredOutOfMemory,
    #[error(
        "the vendored sampler threw a C++ exception while applying to the token data array: {message}"
    )]
    Reported { message: String },
}
