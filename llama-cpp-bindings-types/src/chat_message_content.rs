use serde_json::Value;

use crate::chat_message_content_part::ChatMessageContentPart;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatMessageContentPart>),
}

impl From<&ChatMessageContent> for Value {
    fn from(content: &ChatMessageContent) -> Self {
        match content {
            ChatMessageContent::Text(text) => Self::String(text.clone()),
            ChatMessageContent::Parts(parts) => parts.iter().map(Self::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::ChatMessageContent;
    use crate::chat_message_content_part::ChatMessageContentPart;

    #[test]
    fn text_content_is_a_plain_string() {
        assert_eq!(
            Value::from(&ChatMessageContent::Text("Hello".to_owned())),
            json!("Hello")
        );
    }

    #[test]
    fn content_parts_are_an_array_of_typed_parts() {
        assert_eq!(
            Value::from(&ChatMessageContent::Parts(vec![
                ChatMessageContentPart::MediaMarker("<__media__>".to_owned()),
                ChatMessageContentPart::Text("Describe".to_owned()),
            ])),
            json!([
                {"type": "media_marker", "text": "<__media__>"},
                {"type": "text", "text": "Describe"},
            ])
        );
    }
}
