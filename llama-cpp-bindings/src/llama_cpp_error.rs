use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_token_usage::token_usage_error::TokenUsageError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaCppError {
    #[error("chunk index {0} out of bounds during post-eval walk")]
    ChunkOutOfBounds(usize),
    #[error("the chat template's parser could not classify the answer")]
    Classify(#[from] ChatError),
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error("the generated token could not be detokenized")]
    Detokenize(#[from] ModelError),
    #[error(transparent)]
    MultimodalChunksEval(#[from] MtmdError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error(transparent)]
    TokenUsage(#[from] TokenUsageError),
}
