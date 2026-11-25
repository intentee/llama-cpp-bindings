use serde::Deserialize;
use serde::Serialize;

use crate::text_generation_unsupported_reason::TextGenerationUnsupportedReason;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TextGenerationCapability {
    Supported,
    Unsupported(TextGenerationUnsupportedReason),
}
