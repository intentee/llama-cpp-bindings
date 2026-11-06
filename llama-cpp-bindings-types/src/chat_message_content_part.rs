use serde_json::Value;
use serde_json::json;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatMessageContentPart {
    Text(String),
    MediaMarker(String),
}

impl From<&ChatMessageContentPart> for Value {
    fn from(part: &ChatMessageContentPart) -> Self {
        match part {
            ChatMessageContentPart::Text(text) => json!({"type": "text", "text": text}),
            ChatMessageContentPart::MediaMarker(marker) => {
                json!({"type": "media_marker", "text": marker})
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::ChatMessageContentPart;

    #[test]
    fn a_text_part_is_typed_as_text() {
        assert_eq!(
            Value::from(&ChatMessageContentPart::Text("Describe".to_owned())),
            json!({"type": "text", "text": "Describe"})
        );
    }

    #[test]
    fn a_media_marker_part_is_typed_as_a_media_marker() {
        assert_eq!(
            Value::from(&ChatMessageContentPart::MediaMarker(
                "<__media__>".to_owned()
            )),
            json!({"type": "media_marker", "text": "<__media__>"})
        );
    }
}
