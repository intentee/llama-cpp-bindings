use serde::Deserialize;
use serde::Serialize;

use crate::model_provided_tool_call_id::ModelProvidedToolCallId;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ParsedToolCallId {
    ModelProvided(ModelProvidedToolCallId),
    NotProvidedByModel,
}

#[cfg(test)]
mod tests {
    use serde_json::error::Category;

    use super::ParsedToolCallId;

    #[test]
    fn an_empty_model_provided_id_is_rejected_when_deserialized() {
        assert_eq!(
            serde_json::from_str::<ParsedToolCallId>(r#"{"ModelProvided":""}"#)
                .map_err(|deserialize_error| deserialize_error.classify()),
            Err(Category::Data)
        );
    }
}
