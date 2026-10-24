#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("a value does not fit into i32")]
    IntegerOverflow(#[source] std::num::TryFromIntError),
    #[error("the llama.cpp library ran out of memory initializing the {sampler} sampler")]
    LlamaCppOutOfMemory { sampler: &'static str },
    #[error("not enough memory to report why the sampler could not be initialized")]
    NotEnoughMemory,
    #[error("initializing the {sampler} sampler failed: {message}")]
    Reported {
        sampler: &'static str,
        message: String,
    },
    #[error("{sampler} sampler could not be initialized")]
    SamplerUnavailable { sampler: &'static str },
}
