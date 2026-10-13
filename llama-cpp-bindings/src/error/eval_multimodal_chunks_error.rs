use crate::mtmd::MtmdEvalError;

#[derive(Debug, thiserror::Error)]
pub enum EvalMultimodalChunksError {
    #[error("{0}")]
    EvalFailed(#[from] MtmdEvalError),
    #[error("chunk index {0} out of bounds during post-eval walk")]
    ChunkOutOfBounds(usize),
}
