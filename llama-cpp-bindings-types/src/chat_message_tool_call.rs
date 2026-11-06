use serde_json::Value;
use serde_json::json;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessageToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl From<&ChatMessageToolCall> for Value {
    fn from(
        ChatMessageToolCall {
            id,
            name,
            arguments,
        }: &ChatMessageToolCall,
    ) -> Self {
        json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::ChatMessageToolCall;

    #[test]
    fn a_tool_call_is_an_openai_function_call() {
        assert_eq!(
            Value::from(&ChatMessageToolCall {
                id: "call_weather".to_owned(),
                name: "get_weather".to_owned(),
                arguments: r#"{"location":"Paris"}"#.to_owned(),
            }),
            json!({
                "id": "call_weather",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Paris\"}",
                },
            })
        );
    }
}
