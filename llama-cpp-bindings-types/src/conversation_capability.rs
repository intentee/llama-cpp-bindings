use serde::Deserialize;
use serde::Serialize;

use crate::chat_template_capabilities::ChatTemplateCapabilities;
use crate::conversation_unsupported_reason::ConversationUnsupportedReason;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ConversationCapability {
    Supported(ChatTemplateCapabilities),
    Unsupported(ConversationUnsupportedReason),
}
