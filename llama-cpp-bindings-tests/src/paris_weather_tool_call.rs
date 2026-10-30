use llama_cpp_bindings::ParsedToolCall;
use llama_cpp_bindings::ParsedToolCallId;
use llama_cpp_bindings::ToolCallArguments;
use serde_json::json;

/// The `get_weather` call for Paris that the tool-call fixtures of every format encode.
#[must_use]
pub fn paris_weather_tool_call() -> ParsedToolCall {
    ParsedToolCall::new(
        ParsedToolCallId::NotProvidedByModel,
        "get_weather".to_owned(),
        ToolCallArguments::ValidJson(json!({"location": "Paris"})),
    )
}
