#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApplyChatTemplateError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("the model has no vocab")]
    NoVocab,
    #[error("the model's chat template rendered an empty prompt or could not be rendered")]
    TemplateApplicationFailed,
    #[error("not enough memory to render the chat template")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
