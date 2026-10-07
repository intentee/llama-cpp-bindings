#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    #[error("Logits were not enabled for the given token")]
    LogitsNotEnabled,
    #[error("Can't use sequence embeddings with a model supporting only LLAMA_POOLING_TYPE_NONE")]
    NonePoolType,
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbeddingDimension(#[source] std::num::TryFromIntError),
    #[error(
        "NextN embeddings need a context with LLAMA_POOLING_TYPE_NONE, but it pools with {pooling_type:?}"
    )]
    NextnEmbeddingsRequireNonePooling {
        pooling_type: crate::context::params::LlamaPoolingType,
    },
    #[error("NextN embeddings weren't enabled on this context")]
    NextnEmbeddingsNotEnabled,
    #[error(
        "No NextN embedding exists for token {token_index}; it was not marked as an output of the last decoded batch"
    )]
    NextnEmbeddingUnavailable { token_index: i32 },
}
