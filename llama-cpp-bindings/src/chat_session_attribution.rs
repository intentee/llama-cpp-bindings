use std::collections::VecDeque;

use crate::chat_output::ChatOutput;
use crate::chat_session::ChatSession;
use crate::pending_chat_piece::PendingChatPiece;
use crate::token::LlamaToken;

pub struct ChatSessionAttribution<'session, 'model> {
    pub session: &'session ChatSession<'model>,
    pub preserved_tokens: Vec<LlamaToken>,
    pub output: ChatOutput,
    pub pending_pieces: VecDeque<PendingChatPiece>,
}
