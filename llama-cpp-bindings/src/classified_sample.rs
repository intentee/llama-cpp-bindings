use llama_cpp_model::llama_token::LlamaToken;

use crate::generation_progress::GenerationProgress;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClassifiedSample {
    pub token: LlamaToken,
    pub progress: GenerationProgress,
}
