use crate::context::context_limits::ContextLimits;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum BatchTransferError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("the batch was built for context limits {batch:?}, but the context has {context:?}")]
    LimitsMismatch {
        batch: ContextLimits,
        context: ContextLimits,
    },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
