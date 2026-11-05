use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::invalid_arguments_json::InvalidArgumentsJson;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolCallArguments {
    ValidJson(Value),
    InvalidJson {
        raw: String,
        error: InvalidArgumentsJson,
    },
}

impl ToolCallArguments {
    #[must_use]
    pub fn from_string(raw: String) -> Self {
        match serde_json::from_str::<Value>(&raw) {
            Ok(value) => Self::ValidJson(value),
            Err(error) => Self::InvalidJson {
                raw,
                error: error.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ToolCallArguments;
    use crate::invalid_arguments_json::InvalidArgumentsJson;

    #[test]
    fn from_string_object_returns_valid() {
        let result = ToolCallArguments::from_string(r#"{"location":"Paris"}"#.to_owned());

        assert_eq!(
            result,
            ToolCallArguments::ValidJson(json!({"location": "Paris"}))
        );
    }

    #[test]
    fn from_string_array_returns_valid() {
        let result = ToolCallArguments::from_string("[1,2,3]".to_owned());

        assert_eq!(result, ToolCallArguments::ValidJson(json!([1, 2, 3])));
    }

    #[test]
    fn from_string_scalar_returns_valid() {
        let result = ToolCallArguments::from_string("42".to_owned());

        assert_eq!(result, ToolCallArguments::ValidJson(json!(42)));
    }

    #[test]
    fn unparseable_arguments_keep_their_text_and_where_parsing_failed() {
        let raw = "{not really json".to_owned();

        assert_eq!(
            ToolCallArguments::from_string(raw.clone()),
            ToolCallArguments::InvalidJson {
                raw,
                error: InvalidArgumentsJson {
                    message: "key must be a string at line 1 column 2".to_owned(),
                    line: 1,
                    column: 2,
                },
            }
        );
    }

    #[test]
    fn empty_arguments_are_invalid_at_their_end() {
        assert_eq!(
            ToolCallArguments::from_string(String::new()),
            ToolCallArguments::InvalidJson {
                raw: String::new(),
                error: InvalidArgumentsJson {
                    message: "EOF while parsing a value at line 1 column 0".to_owned(),
                    line: 1,
                    column: 0,
                },
            }
        );
    }
}
