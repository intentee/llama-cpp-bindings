use llama_cpp_bindings_types::chat_tool::ChatTool;
use serde_json::json;

#[must_use]
pub fn get_weather_tools() -> Vec<ChatTool> {
    vec![ChatTool {
        name: "get_weather".to_owned(),
        description: "Get the current weather for a location".to_owned(),
        parameters: Some(json!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"}
            },
            "required": ["location"]
        })),
    }]
}
