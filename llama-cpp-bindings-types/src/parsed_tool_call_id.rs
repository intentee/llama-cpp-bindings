use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub enum ParsedToolCallId {
    ModelProvided(String),
    NotProvidedByModel,
}
