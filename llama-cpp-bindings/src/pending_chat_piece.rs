use std::ops::Range;

use llama_cpp_model::llama_token::LlamaToken;

#[derive(Debug)]
pub struct PendingChatPiece {
    pub token: LlamaToken,
    pub bytes: Range<usize>,
    pub piece: String,
}
