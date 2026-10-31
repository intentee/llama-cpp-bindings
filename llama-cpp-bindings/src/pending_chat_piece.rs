use std::ops::Range;

use crate::token::LlamaToken;

#[derive(Debug)]
pub struct PendingChatPiece {
    pub token: LlamaToken,
    pub bytes: Range<usize>,
    pub piece: String,
}
