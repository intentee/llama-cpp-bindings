use std::ffi::NulError;

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
    #[error("DRY sequence breaker {index} is empty")]
    DrySequenceBreakerEmpty { index: usize },
    #[error(
        "DRY sequence breaker {index} is {length} bytes long, more than the {max_length} bytes llama.cpp uses"
    )]
    DrySequenceBreakerTooLong {
        index: usize,
        length: usize,
        max_length: usize,
    },
    #[error("DRY sequence breaker {index} contains an interior NUL byte")]
    DrySequenceBreakerContainsNul {
        index: usize,
        #[source]
        source: NulError,
    },
    #[error(
        "{distinct} distinct DRY sequence breakers were given, more than the {max} llama.cpp uses"
    )]
    TooManyDistinctDrySequenceBreakers { distinct: usize, max: usize },
}
