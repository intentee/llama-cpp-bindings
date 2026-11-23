use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_token_usage::token_usage_error::TokenUsageError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaCppError {
    #[error("the chat template's parser could not classify the answer")]
    Classify(#[source] ChatError),
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error("the generated token could not be detokenized")]
    Detokenize(#[source] ModelError),
    #[error(transparent)]
    EmptyAnswerReading(ChatError),
    #[error(transparent)]
    MultimodalChunksEval(MtmdError),
    #[error("classifying the sampled token failed")]
    SampleIngest(#[source] Box<Self>),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error(transparent)]
    TokenUsage(#[from] TokenUsageError),
}
