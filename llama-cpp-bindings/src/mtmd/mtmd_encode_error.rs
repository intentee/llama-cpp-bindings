#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdEncodeError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("multimodal chunk encoding failed with code: {code}")]
    EncodingFailed { code: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
