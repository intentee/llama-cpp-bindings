use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_types::chat_tool::ChatTool;

use crate::user_message::user_message;

/// A chat session that answers one user turn, offering `tools`.
///
/// # Errors
///
/// Fails when the model has no chat templates or its template cannot serve the request.
pub fn user_turn_chat_session(
    model: &LlamaModel,
    content: &str,
    tools: &[ChatTool],
    enable_thinking: bool,
) -> Result<ChatSession> {
    ChatSession::new(
        model,
        model
            .chat_templates()
            .context("the model must have chat templates")?,
        &[user_message(content)],
        tools,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )
    .context("the model's template must serve the user turn")
}
