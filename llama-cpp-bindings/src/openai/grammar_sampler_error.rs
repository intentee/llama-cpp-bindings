use crate::GrammarError;

/// Error type for grammar sampler construction.
#[derive(Debug, thiserror::Error)]
pub enum GrammarSamplerError {
    /// Lazy grammar mode is enabled but no triggers were provided.
    #[error("grammar_lazy enabled but no triggers provided")]
    MissingTriggers,
    /// A trigger word is not in the preserved tokens set.
    #[error("grammar trigger word should be a preserved token: {0}")]
    TriggerWordNotPreserved(String),
    /// Failed to tokenize a trigger or preserved token.
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
    /// Failed to initialize the grammar sampler.
    #[error("grammar sampler init failed: {0}")]
    GrammarInitFailed(#[from] GrammarError),
}
