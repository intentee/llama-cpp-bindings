#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum StateDataError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error(
        "the vendored deserializer restored nothing from a {provided_bytes}-byte snapshot; \
         llama.cpp logs the cause and reports zero bytes rather than throwing"
    )]
    NothingRestored { provided_bytes: usize },
    #[error("{message}")]
    Reported { message: String },
}
