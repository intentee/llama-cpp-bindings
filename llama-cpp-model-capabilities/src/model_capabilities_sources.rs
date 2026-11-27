use std::sync::Arc;

use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_context::MtmdContext;

use crate::conversation_templates::ConversationTemplates;
use crate::embedding_pooling_choice::EmbeddingPoolingChoice;

pub struct ModelCapabilitiesSources<'sources> {
    pub conversation_templates: ConversationTemplates<'sources>,
    pub embedding_pooling: EmbeddingPoolingChoice,
    pub model: &'sources Arc<LlamaModel>,
    pub projector: Option<&'sources MtmdContext>,
}
