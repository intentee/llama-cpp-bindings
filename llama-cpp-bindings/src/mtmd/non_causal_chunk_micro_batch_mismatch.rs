#[derive(Debug, PartialEq, Eq)]
pub struct NonCausalChunkMicroBatchMismatch {
    pub chunk_tokens: usize,
    pub micro_batch_tokens: u32,
}
