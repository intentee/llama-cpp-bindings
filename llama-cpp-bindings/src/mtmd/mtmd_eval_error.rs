use crate::mtmd::image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;
use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdEvalError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit { requested: i32, context_max: u32 },
    #[error(
        "image chunk has {} tokens but n_batch is {}",
        .0.image_tokens,
        .0.n_batch,
    )]
    ImageChunkExceedsBatchSize(ImageChunkBatchSizeMismatch),
    #[error("multimodal chunk eval failed with code: {code}")]
    EvalFailed { code: i32 },
    #[error("the chunk type could not be classified before evaluating it: {0}")]
    UnknownChunkType(#[from] MtmdInputChunkTypeError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
