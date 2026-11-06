use serde_json::Map;
use serde_json::Value;

use crate::chat_message_content::ChatMessageContent;
use crate::chat_message_tool_call::ChatMessageToolCall;

const ASSISTANT_ROLE: &str = "assistant";
const TOOL_ROLE: &str = "tool";

fn message_of(role: &str, content: &ChatMessageContent) -> Map<String, Value> {
    Map::from_iter([
        ("role".to_owned(), Value::String(role.to_owned())),
        ("content".to_owned(), Value::from(content)),
    ])
}

fn with_reasoning(
    mut message: Map<String, Value>,
    reasoning_content: Option<&String>,
) -> Map<String, Value> {
    if let Some(reasoning_content) = reasoning_content {
        message.insert(
            "reasoning_content".to_owned(),
            Value::String(reasoning_content.clone()),
        );
    }

    message
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatMessage {
    Message {
        role: String,
        content: ChatMessageContent,
        reasoning_content: Option<String>,
    },
    AssistantToolCalls {
        content: ChatMessageContent,
        reasoning_content: Option<String>,
        tool_calls: Vec<ChatMessageToolCall>,
    },
    ToolResult {
        content: ChatMessageContent,
        tool_call_id: String,
    },
}

impl From<&ChatMessage> for Value {
    fn from(message: &ChatMessage) -> Self {
        Self::Object(match message {
            ChatMessage::Message {
                role,
                content,
                reasoning_content,
            } => with_reasoning(message_of(role, content), reasoning_content.as_ref()),
            ChatMessage::AssistantToolCalls {
                content,
                reasoning_content,
                tool_calls,
            } => {
                let mut assistant_turn = with_reasoning(
                    message_of(ASSISTANT_ROLE, content),
                    reasoning_content.as_ref(),
                );

                assistant_turn.insert(
                    "tool_calls".to_owned(),
                    tool_calls.iter().map(Self::from).collect(),
                );

                assistant_turn
            }
            ChatMessage::ToolResult {
                content,
                tool_call_id,
            } => {
                let mut tool_result = message_of(TOOL_ROLE, content);

                tool_result.insert(
                    "tool_call_id".to_owned(),
                    Self::String(tool_call_id.clone()),
                );

                tool_result
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::ChatMessage;
    use crate::chat_message_content::ChatMessageContent;
    use crate::chat_message_tool_call::ChatMessageToolCall;

    #[test]
    fn a_message_carries_its_own_role() {
        assert_eq!(
            Value::from(&ChatMessage::Message {
                role: "user".to_owned(),
                content: ChatMessageContent::Text("Hello".to_owned()),
                reasoning_content: None,
            }),
            json!({"role": "user", "content": "Hello"})
        );
    }

    #[test]
    fn a_reasoned_message_carries_its_reasoning() {
        assert_eq!(
            Value::from(&ChatMessage::Message {
                role: "assistant".to_owned(),
                content: ChatMessageContent::Text("Paris".to_owned()),
                reasoning_content: Some("The capital of France".to_owned()),
            }),
            json!({
                "role": "assistant",
                "content": "Paris",
                "reasoning_content": "The capital of France",
            })
        );
    }

    #[test]
    fn an_assistant_tool_call_turn_is_an_assistant_message_with_tool_calls() {
        assert_eq!(
            Value::from(&ChatMessage::AssistantToolCalls {
                content: ChatMessageContent::Text(String::new()),
                reasoning_content: Some("I need the weather".to_owned()),
                tool_calls: vec![ChatMessageToolCall {
                    id: "call_weather".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: "{}".to_owned(),
                }],
            }),
            json!({
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need the weather",
                "tool_calls": [{
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            })
        );
    }

    #[test]
    fn a_tool_result_is_a_tool_message_answering_its_call() {
        assert_eq!(
            Value::from(&ChatMessage::ToolResult {
                content: ChatMessageContent::Text("sunny".to_owned()),
                tool_call_id: "call_weather".to_owned(),
            }),
            json!({"role": "tool", "content": "sunny", "tool_call_id": "call_weather"})
        );
    }
}
