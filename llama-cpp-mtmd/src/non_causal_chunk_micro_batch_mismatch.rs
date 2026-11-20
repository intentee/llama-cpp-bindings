#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NonCausalChunkMicroBatchMismatch {
    pub chunk_tokens: usize,
    pub micro_batch_tokens: u32,
}
