use crate::chat_session_attribution::ChatSessionAttribution;

pub enum TokenAttribution<'session, 'model> {
    Undeterminable,
    ChatSession(ChatSessionAttribution<'session, 'model>),
}
