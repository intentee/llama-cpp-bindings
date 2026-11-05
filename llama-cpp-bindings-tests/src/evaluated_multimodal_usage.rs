use llama_cpp_bindings_types::token_usage::TokenUsage;

use crate::chunk_token_breakdown::ChunkTokenBreakdown;

pub struct EvaluatedMultimodalUsage {
    pub usage: TokenUsage,
    pub chunk_breakdown: ChunkTokenBreakdown,
}
