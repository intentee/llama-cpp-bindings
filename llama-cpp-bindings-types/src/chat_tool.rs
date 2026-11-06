use serde_json::Map;
use serde_json::Value;
use serde_json::json;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
}

impl From<&ChatTool> for Value {
    fn from(
        ChatTool {
            name,
            description,
            parameters,
        }: &ChatTool,
    ) -> Self {
        let mut function = Map::from_iter([
            ("name".to_owned(), Self::String(name.clone())),
            ("description".to_owned(), Self::String(description.clone())),
        ]);

        if let Some(parameters) = parameters {
            function.insert("parameters".to_owned(), parameters.clone());
        }

        json!({"type": "function", "function": function})
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::ChatTool;

    #[test]
    fn a_tool_with_parameters_is_an_openai_function_with_its_schema() {
        let schema = json!({"type": "object", "properties": {"location": {"type": "string"}}});

        assert_eq!(
            Value::from(&ChatTool {
                name: "get_weather".to_owned(),
                description: "Reports the weather".to_owned(),
                parameters: Some(schema.clone()),
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Reports the weather",
                    "parameters": schema,
                },
            })
        );
    }

    #[test]
    fn a_tool_without_parameters_is_an_openai_function_without_them() {
        assert_eq!(
            Value::from(&ChatTool {
                name: "get_time".to_owned(),
                description: "Reports the time".to_owned(),
                parameters: None,
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Reports the time",
                },
            })
        );
    }
}
