use anyhow::Result;
use llama_cpp_bindings::ChatAssistantTurn;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ChatSessionOptions;
use llama_cpp_bindings::ChatToolChoice;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::LlamaModel;

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
