#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    #[error("token {token_index} names no output row of the last decode")]
    TokenNotAnOutput { token_index: i32 },
    #[error(
        "Sequence embeddings need a pooling context, but this context uses LLAMA_POOLING_TYPE_NONE"
    )]
    NonePoolType,
    #[error(
        "No sequence embedding exists for sequence {sequence_index}; it was not part of the last decoded batch"
    )]
    SequenceEmbeddingUnavailable { sequence_index: i32 },
    #[error(
        "Token embeddings need a context with LLAMA_POOLING_TYPE_NONE, but it pools with {pooling_type:?}"
    )]
    TokenEmbeddingsRequireNonePooling {
        pooling_type: crate::context::llama_pooling_type::LlamaPoolingType,
    },
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbeddingDimension(#[source] std::num::TryFromIntError),
}
