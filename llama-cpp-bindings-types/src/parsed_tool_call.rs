use serde::Deserialize;
use serde::Serialize;

use crate::parsed_tool_call_id::ParsedToolCallId;
use crate::tool_call_arguments::ToolCallArguments;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParsedToolCall {
    pub id: ParsedToolCallId,
    pub name: String,
    pub arguments: ToolCallArguments,
}

impl ParsedToolCall {
    #[must_use]
    pub const fn new(id: ParsedToolCallId, name: String, arguments: ToolCallArguments) -> Self {
        Self {
            id,
            name,
            arguments,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ParsedToolCall;
    use crate::parsed_tool_call_id::ParsedToolCallId;
    use crate::tool_call_arguments::ToolCallArguments;

    #[test]
    fn new_assigns_fields_in_order() {
        let parsed = ParsedToolCall::new(
            ParsedToolCallId::ModelProvided("id-1".to_owned()),
            "tool".to_owned(),
            ToolCallArguments::ValidJson(json!({})),
        );

        assert_eq!(
            parsed.id,
            ParsedToolCallId::ModelProvided("id-1".to_owned())
        );
        assert_eq!(parsed.name, "tool");
        assert_eq!(parsed.arguments, ToolCallArguments::ValidJson(json!({})));
    }
}
