use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_chat::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_chat::chat_session_options::ChatSessionOptions;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_model::llama_model::LlamaModel;

use crate::get_weather_tools::get_weather_tools;
use crate::user_message::user_message;

/// A chat session that asks for the weather in Paris, offering the weather tool with
/// `tool_choice`.
///
/// # Errors
///
/// Fails when the model's template cannot serve the request.
pub fn weather_chat_session(
    model: &LlamaModel,
    tool_choice: ChatToolChoice,
) -> Result<ChatSession> {
    ChatSession::new(
        model,
        model
            .chat_templates()
            .context("the model must have chat templates")?,
        &[user_message("What is the weather in Paris?")],
        &get_weather_tools(),
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: false,
            parallel_tool_calls: false,
            tool_choice,
        },
    )
    .context("the model's template must serve the weather request")
}
