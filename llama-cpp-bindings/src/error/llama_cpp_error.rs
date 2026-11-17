use llama_cpp_backend::backend_error::BackendError;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_gbnf::gbnf_error::GbnfError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_params::params_error::ParamsError;
use llama_cpp_sampler::sampler_error::SamplerError;

use crate::error::ingest_error::IngestError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaCppError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error(transparent)]
    GbnfError(#[from] GbnfError),
    #[error("classifying the sampled token failed")]
    Ingest(#[from] IngestError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Params(#[from] ParamsError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
}
