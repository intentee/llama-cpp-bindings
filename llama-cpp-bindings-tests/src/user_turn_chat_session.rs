use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::error::chat_session_error::ChatSessionError;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_types::chat_tool::ChatTool;

use crate::user_message::user_message;

/// A chat session that answers one user turn, offering `tools`.
///
/// # Errors
///
/// Forwards chat session errors.
///
/// # Panics
///
/// Panics when the model has no chat templates.
pub fn user_turn_chat_session(
    model: &LlamaModel,
    content: &str,
    tools: &[ChatTool],
    enable_thinking: bool,
) -> Result<ChatSession, ChatSessionError> {
    ChatSession::new(
        model,
        model
            .chat_templates()
            .expect("the model must have chat templates"),
        &[user_message(content)],
        tools,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )
}
