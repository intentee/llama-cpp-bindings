use llama_cpp_backend::backend_error::BackendError;
use llama_cpp_gbnf::gbnf_error::GbnfError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_params::params_error::ParamsError;

use crate::batch_add_error::BatchAddError;
use crate::error::decode_error::DecodeError;
use crate::error::embeddings_error::EmbeddingsError;
use crate::error::encode_error::EncodeError;
use crate::error::llama_context_load_error::LlamaContextLoadError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaCppError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
    #[error(transparent)]
    EncodeError(#[from] EncodeError),
    #[error(transparent)]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
    #[error(transparent)]
    BatchAddError(#[from] BatchAddError),
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingsError),
    #[error(transparent)]
    GbnfError(#[from] GbnfError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Params(#[from] ParamsError),
}
