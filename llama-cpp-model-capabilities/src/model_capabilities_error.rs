use std::num::TryFromIntError;

use llama_cpp_decision::decision_error::DecisionError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;
use llama_cpp_params::params_error::ParamsError;

#[derive(Debug, thiserror::Error)]
pub enum ModelCapabilitiesError {
    #[error("the chat templates' capabilities cannot be read")]
    ChatTemplateCapabilitiesUnreadable(#[source] ModelError),
    #[error("the model's training context size does not fit a context size")]
    ContextSizeUnrepresentable(#[source] TryFromIntError),
    #[error("the model declares a pooling type llama.cpp does not define")]
    DeclaredPoolingTypeUnknown(#[source] ParamsError),
    #[error("the model declares a decision type, but llama.cpp cannot serve its decisions")]
    DecisionModelRejected(#[source] DecisionError),
    #[error("pooling {pooling:?} was chosen for embeddings, but it produces no embedding vectors")]
    PoolingProducesNoEmbeddingVectors { pooling: LlamaPoolingType },
}
