use crate::context::llama_pooling_type::LlamaPoolingType;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LogitsError {
    #[error("logits data pointer is null")]
    NullLogits,
    #[error(
        "the last decode pooled embeddings with {pooling_type:?} pooling, which produces no logits"
    )]
    PooledEmbeddingsProduceNoLogits { pooling_type: LlamaPoolingType },
    #[error("the model graph of the last decode produces no logits")]
    GraphProducesNoLogits,
    #[error("logit for token index {0} is not initialized")]
    TokenNotInitialized(i32),
    #[error("n_vocab does not fit into usize: {0}")]
    VocabSizeOverflow(#[source] std::num::TryFromIntError),
}
