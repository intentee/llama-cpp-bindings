#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    #[error("Logits were not enabled for the given token")]
    LogitsNotEnabled,
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
        pooling_type: crate::context::params::LlamaPoolingType,
    },
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
        "NextN embeddings are unavailable because this model's graph produces no NextN hidden state"
    )]
    NextnEmbeddingsUnsupported,
    #[error("llama.cpp failed to reserve the graph that checks for a NextN hidden state")]
    NextnProbeGraphReservationFailed,
    #[error("not enough memory to report why the NextN hidden state check failed")]
    NextnProbeNotEnoughMemory,
    #[error("llama.cpp ran out of memory while checking for a NextN hidden state")]
    NextnProbeLlamaCppOutOfMemory,
    #[error("llama.cpp failed while checking for a NextN hidden state: {message}")]
    NextnProbeReported { message: String },
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(
        "No NextN embedding exists for token {token_index}; it was not marked as an output of the last decoded batch"
    )]
    NextnEmbeddingUnavailable { token_index: i32 },
}
