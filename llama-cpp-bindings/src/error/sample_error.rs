use crate::error::sampler_apply_error::SamplerApplyError;
use crate::error::token_to_string_error::TokenToStringError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SampleError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("applying the sampler to the token data array failed: {0}")]
    SamplerApply(#[from] SamplerApplyError),
    #[error("token detokenization failed during classification: {0}")]
    Detokenize(#[from] TokenToStringError),
    #[error("the grammar sampler callback failed during sampling: {message}")]
    GrammarCallbackFailed { message: String },
    #[error("{message}")]
    Reported { message: String },
}
