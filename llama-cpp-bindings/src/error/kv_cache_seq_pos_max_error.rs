#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvCacheSeqPosMaxError {
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error("sequence id {seq_id} is outside the context sequence range")]
    SequenceIdOutOfRange { seq_id: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
