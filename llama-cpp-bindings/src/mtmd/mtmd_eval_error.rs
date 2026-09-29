use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;
use crate::mtmd::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdEvalError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit { requested: i32, context_max: u32 },
    #[error("batch size {requested} must be positive")]
    NonPositiveBatchSize { requested: i32 },
    #[error(
        "a chunk decoded non-causally has {} tokens but a single decode fits {}",
        .0.chunk_tokens,
        .0.micro_batch_tokens,
    )]
    NonCausalChunkExceedsMicroBatch(NonCausalChunkMicroBatchMismatch),
    #[error("multimodal chunk eval failed with code: {code}")]
    EvalFailed { code: i32 },
    #[error("the chunk type could not be classified before evaluating it: {0}")]
    UnknownChunkType(#[from] MtmdInputChunkTypeError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
