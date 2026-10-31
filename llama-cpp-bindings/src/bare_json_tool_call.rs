use serde::Deserialize;
use serde::de::IgnoredAny;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BareJsonToolCall {
    pub name: String,
    #[serde(rename = "arguments")]
    _arguments: Option<IgnoredAny>,
}
