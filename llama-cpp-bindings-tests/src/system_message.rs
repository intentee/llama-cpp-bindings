use llama_cpp_bindings_types::chat_message::ChatMessage;
use llama_cpp_bindings_types::chat_message_content::ChatMessageContent;

#[must_use]
pub fn system_message(content: &str) -> ChatMessage {
    ChatMessage::Message {
        role: "system".to_owned(),
        content: ChatMessageContent::Text(content.to_owned()),
        reasoning_content: None,
    }
}
