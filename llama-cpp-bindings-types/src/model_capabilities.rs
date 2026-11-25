use serde::Deserialize;
use serde::Serialize;

use crate::conversation_capability::ConversationCapability;
use crate::decisions_capability::DecisionsCapability;
use crate::embeddings_capability::EmbeddingsCapability;
use crate::input_media_capabilities::InputMediaCapabilities;
use crate::model_defaults::ModelDefaults;
use crate::text_generation_capability::TextGenerationCapability;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilities {
    pub conversation: ConversationCapability,
    pub decisions: DecisionsCapability,
    pub embeddings: EmbeddingsCapability,
    pub input_media: InputMediaCapabilities,
    pub model_defaults: ModelDefaults,
    pub text_generation: TextGenerationCapability,
}
