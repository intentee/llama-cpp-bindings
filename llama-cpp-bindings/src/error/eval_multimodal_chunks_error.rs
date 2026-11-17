use llama_cpp_mtmd::mtmd_error::MtmdError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EvalMultimodalChunksError {
    #[error(transparent)]
    EvalFailed(#[from] MtmdError),
    #[error("chunk index {0} out of bounds during post-eval walk")]
    ChunkOutOfBounds(usize),
}
