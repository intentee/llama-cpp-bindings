use serde::Deserialize;
use serde::Serialize;

use crate::model_provided_tool_call_id_error::ModelProvidedToolCallIdError;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(try_from = "String")]
pub struct ModelProvidedToolCallId(String);

impl ModelProvidedToolCallId {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ModelProvidedToolCallId {
    type Error = ModelProvidedToolCallIdError;

    fn try_from(tool_call_id: String) -> Result<Self, Self::Error> {
        if tool_call_id.is_empty() {
            return Err(ModelProvidedToolCallIdError::Empty);
        }

        Ok(Self(tool_call_id))
    }
}

#[cfg(test)]
mod tests {
    use super::ModelProvidedToolCallId;
    use crate::model_provided_tool_call_id_error::ModelProvidedToolCallIdError;

    #[test]
    fn rejects_an_empty_tool_call_id() {
        assert_eq!(
            ModelProvidedToolCallId::try_from(String::new()),
            Err(ModelProvidedToolCallIdError::Empty)
        );
    }

    #[test]
    fn keeps_the_tool_call_id_the_model_gave() {
        assert_eq!(
            ModelProvidedToolCallId::try_from("call_7".to_owned())
                .as_ref()
                .map(ModelProvidedToolCallId::as_str),
            Ok("call_7")
        );
    }
}
