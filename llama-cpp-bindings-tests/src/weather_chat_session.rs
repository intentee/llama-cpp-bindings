use llama_cpp_bindings::ChatAssistantTurn;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ChatSessionOptions;
use llama_cpp_bindings::ChatToolChoice;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::LlamaModel;

use crate::get_weather_tools_json::GET_WEATHER_TOOLS_JSON;

/// A chat session that asks for the weather in Paris, offering the weather tool with
/// `tool_choice`.
///
/// # Panics
///
/// Panics when the model's template cannot serve the request.
#[must_use]
pub fn weather_chat_session(model: &LlamaModel, tool_choice: ChatToolChoice) -> ChatSession<'_> {
    ChatSession::new(
        model,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            "What is the weather in Paris?".to_owned(),
        )
        .expect("the question holds no NUL byte")],
        GET_WEATHER_TOOLS_JSON,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: false,
            parallel_tool_calls: false,
            tool_choice,
        },
    )
    .expect("the model's template must serve the weather request")
}
