use anyhow::Result;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;

/// A chat session that answers one user turn, offering the tools of `tools_json`.
///
/// # Errors
///
/// Forwards message construction and chat session errors.
pub fn user_turn_chat_session<'model>(
    model: &'model LlamaModel,
    content: &str,
    tools_json: &str,
    enable_thinking: bool,
) -> Result<ChatSession<'model>> {
    Ok(ChatSession::new(
        model,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            content.to_owned(),
        )?],
        tools_json,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?)
}
