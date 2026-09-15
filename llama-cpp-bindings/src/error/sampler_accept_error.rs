#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SamplerAcceptError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("grammar state corrupted during accept: {message}")]
    GrammarStateCorrupted { message: String },
    #[error("the grammar sampler callback failed during accept: {message}")]
    GrammarCallbackFailed { message: String },
}
