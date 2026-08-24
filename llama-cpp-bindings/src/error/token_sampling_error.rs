use crate::SamplingError;
use crate::error::sampler_apply_error::SamplerApplyError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenSamplingError {
    #[error("sampler initialization failed: {0}")]
    SamplerInitialization(#[from] SamplingError),
    #[error("No token was selected by the sampler")]
    NoTokenSelected,
    #[error("applying the sampler to the token data array failed: {0}")]
    SamplerApply(#[from] SamplerApplyError),
}
