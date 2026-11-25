use std::collections::VecDeque;

use llama_cpp_chat::chat_output::ChatOutput;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_model::llama_token::LlamaToken;

use crate::chat_output_reading::ChatOutputReading;
use crate::pending_chat_piece::PendingChatPiece;

pub struct ChatSessionAttribution {
    pub session: ChatSession,
    pub preserved_tokens: Vec<LlamaToken>,
    pub output: ChatOutput,
    pub pending_pieces: VecDeque<PendingChatPiece>,
    pub latest_reading: ChatOutputReading,
}
