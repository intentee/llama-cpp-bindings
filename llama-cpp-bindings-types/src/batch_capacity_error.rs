#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum BatchCapacityError {
    #[error("a batch must hold at least one token")]
    Zero,
    #[error("a batch of {tokens} tokens exceeds the {max_tokens} tokens llama.cpp can address")]
    TooLarge { tokens: u32, max_tokens: u32 },
}
