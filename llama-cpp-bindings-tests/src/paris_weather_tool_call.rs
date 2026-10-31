use serde_json::json;

use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;
use llama_cpp_bindings_types::parsed_tool_call_id::ParsedToolCallId;
use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;

/// The `get_weather` call for Paris that the tool-call fixtures of every format encode.
#[must_use]
pub fn paris_weather_tool_call() -> ParsedToolCall {
    ParsedToolCall::new(
        ParsedToolCallId::NotProvidedByModel,
        "get_weather".to_owned(),
        ToolCallArguments::ValidJson(json!({"location": "Paris"})),
    )
}
