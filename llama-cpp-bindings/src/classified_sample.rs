use crate::generation_progress::GenerationProgress;
use crate::token::LlamaToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClassifiedSample {
    pub token: LlamaToken,
    pub progress: GenerationProgress,
}
