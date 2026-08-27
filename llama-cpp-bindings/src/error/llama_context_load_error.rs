#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("context could not be constructed")]
    Unconstructible,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
