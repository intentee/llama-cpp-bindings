use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

use crate::error::ingest_error::IngestError;
use crate::error::logits_error::LogitsError;
use crate::error::sampler_apply_error::SamplerApplyError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SampleError {
    #[error(transparent)]
    FfiStatus(#[from] FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] FfiContractError),
    #[error("the logits to sample from are unavailable: {0}")]
    Logits(#[from] LogitsError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("applying the sampler to the token data array failed: {0}")]
    SamplerApply(#[from] SamplerApplyError),
    #[error("classifying the sampled token failed: {0}")]
    Ingest(#[from] IngestError),
    #[error("the grammar sampler callback failed during sampling: {message}")]
    GrammarCallbackFailed { message: String },
    #[error("{message}")]
    Reported { message: String },
}
