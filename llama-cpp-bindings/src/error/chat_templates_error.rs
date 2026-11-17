use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplatesError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("the model has no chat template")]
    ModelHasNoChatTemplate,
    #[error(
        "an override template must not be empty, because llama.cpp reads an empty override as no override"
    )]
    EmptyTemplateOverride,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
