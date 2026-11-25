use serde::Deserialize;
use serde::Serialize;

use crate::tool_calling_capability::ToolCallingCapability;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatTemplateCapabilities {
    pub system_role: bool,
    pub thinking: bool,
    pub tool_calling: ToolCallingCapability,
}
