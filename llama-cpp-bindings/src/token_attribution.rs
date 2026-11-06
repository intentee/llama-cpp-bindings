use crate::chat_session_attribution::ChatSessionAttribution;

pub enum TokenAttribution {
    Undeterminable,
    ChatSession(Box<ChatSessionAttribution>),
}
