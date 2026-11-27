use serde::Deserialize;
use serde::Serialize;

use crate::embeddings_unsupported_reason::EmbeddingsUnsupportedReason;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EmbeddingsCapability {
    Supported,
    Unsupported(EmbeddingsUnsupportedReason),
}
