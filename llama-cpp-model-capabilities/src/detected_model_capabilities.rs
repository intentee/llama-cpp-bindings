use std::sync::Arc;

use enumflags2::BitFlags;
use llama_cpp_bindings_types::conversation_capability::ConversationCapability;
use llama_cpp_bindings_types::conversation_unsupported_reason::ConversationUnsupportedReason;
use llama_cpp_bindings_types::decision_capabilities::DecisionCapabilities;
use llama_cpp_bindings_types::decisions_capability::DecisionsCapability;
use llama_cpp_bindings_types::decisions_unsupported_reason::DecisionsUnsupportedReason;
use llama_cpp_bindings_types::embeddings_capability::EmbeddingsCapability;
use llama_cpp_bindings_types::embeddings_unsupported_reason::EmbeddingsUnsupportedReason;
use llama_cpp_bindings_types::input_media_capabilities::InputMediaCapabilities;
use llama_cpp_bindings_types::model_capabilities::ModelCapabilities;
use llama_cpp_bindings_types::model_defaults::ModelDefaults;
use llama_cpp_bindings_types::text_generation_capability::TextGenerationCapability;
use llama_cpp_bindings_types::text_generation_unsupported_reason::TextGenerationUnsupportedReason;
use llama_cpp_decision::decision_error::DecisionError;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_capability::MtmdCapability;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_params::llama_context_params::LlamaContextParams;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;

use crate::conversation_templates::ConversationTemplates;
use crate::detected_decisions::DetectedDecisions;
use crate::embedding_pooling_choice::EmbeddingPoolingChoice;
use crate::model_capabilities_error::ModelCapabilitiesError;
use crate::model_capabilities_sources::ModelCapabilitiesSources;

fn projector_capabilities(projector: Option<&MtmdContext>) -> BitFlags<MtmdCapability> {
    projector.map_or_else(
        || MtmdCapability::Chat | MtmdCapability::TextGeneration,
        MtmdContext::capabilities,
    )
}

fn text_generation_of(
    model: &LlamaModel,
    projector: BitFlags<MtmdCapability>,
) -> TextGenerationCapability {
    if !unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_model_outputs_text(
            model.as_ptr(),
            projector.contains(MtmdCapability::Chat),
            projector.contains(MtmdCapability::AudioGeneration),
        )
    } {
        return TextGenerationCapability::Unsupported(
            TextGenerationUnsupportedReason::NoTextOutput,
        );
    }

    if model.has_encoder() {
        return TextGenerationCapability::Unsupported(
            TextGenerationUnsupportedReason::EncoderDecoderArchitecture,
        );
    }

    if !unsafe { llama_cpp_bindings_sys::bindings::llama_rs_model_causal_attn(model.as_ptr()) } {
        return TextGenerationCapability::Unsupported(
            TextGenerationUnsupportedReason::AttentionIsNotCausal,
        );
    }

    TextGenerationCapability::Supported
}

fn conversation_of(
    text_generation: TextGenerationCapability,
    conversation_templates: ConversationTemplates<'_>,
) -> Result<ConversationCapability, ModelCapabilitiesError> {
    match (text_generation, conversation_templates) {
        (TextGenerationCapability::Unsupported(_), _) => Ok(ConversationCapability::Unsupported(
            ConversationUnsupportedReason::ModelGeneratesNoText,
        )),
        (TextGenerationCapability::Supported, ConversationTemplates::Absent) => {
            Ok(ConversationCapability::Unsupported(
                ConversationUnsupportedReason::ModelHasNoChatTemplate,
            ))
        }
        (TextGenerationCapability::Supported, ConversationTemplates::Present(chat_templates)) => {
            chat_templates
                .capabilities()
                .map(ConversationCapability::Supported)
                .map_err(ModelCapabilitiesError::ChatTemplateCapabilitiesUnreadable)
        }
    }
}

const fn produces_embedding_vectors(pooling: LlamaPoolingType) -> bool {
    matches!(
        pooling,
        LlamaPoolingType::Mean | LlamaPoolingType::Cls | LlamaPoolingType::Last
    )
}

fn embeddings_of(
    model: &LlamaModel,
    embedding_pooling: EmbeddingPoolingChoice,
) -> Result<EmbeddingsCapability, ModelCapabilitiesError> {
    match embedding_pooling {
        EmbeddingPoolingChoice::Explicit(pooling) if produces_embedding_vectors(pooling) => {
            Ok(EmbeddingsCapability::Supported)
        }
        EmbeddingPoolingChoice::Explicit(pooling) => {
            Err(ModelCapabilitiesError::PoolingProducesNoEmbeddingVectors { pooling })
        }
        EmbeddingPoolingChoice::ModelDefault => {
            let declared = LlamaPoolingType::try_from(unsafe {
                llama_cpp_bindings_sys::bindings::llama_rs_model_declared_pooling_type(
                    model.as_ptr(),
                )
            })
            .map_err(ModelCapabilitiesError::DeclaredPoolingTypeUnknown)?;

            Ok(if produces_embedding_vectors(declared) {
                EmbeddingsCapability::Supported
            } else {
                EmbeddingsCapability::Unsupported(
                    EmbeddingsUnsupportedReason::ModelDeclaresNoVectorPooling,
                )
            })
        }
    }
}

fn decisions_of(model: &Arc<LlamaModel>) -> Result<DetectedDecisions, ModelCapabilitiesError> {
    match DecisionModel::new(Arc::clone(model)) {
        Ok(decision_model) => Ok(DetectedDecisions::Supported(decision_model)),
        Err(DecisionError::ModelIsNotADecisionModel) => Ok(DetectedDecisions::Unsupported(
            DecisionsUnsupportedReason::ModelIsNotADecisionModel,
        )),
        Err(DecisionError::DecisionTypeUnsupportedByLlamaCpp) => {
            Ok(DetectedDecisions::Unsupported(
                DecisionsUnsupportedReason::DecisionTypeUnsupportedByLlamaCpp,
            ))
        }
        Err(rejection) => Err(ModelCapabilitiesError::DecisionModelRejected(rejection)),
    }
}

/// What a loaded model can serve, determined once from llama.cpp's own answers, and how every
/// context of the model must be configured for it.
#[derive(Clone)]
pub struct DetectedModelCapabilities {
    pub conversation: ConversationCapability,
    pub decisions: DetectedDecisions,
    pub embeddings: EmbeddingsCapability,
    pub input_media: InputMediaCapabilities,
    pub model_defaults: ModelDefaults,
    pub needs_classifier_free_guidance: bool,
    pub text_generation: TextGenerationCapability,
}

impl DetectedModelCapabilities {
    /// # Errors
    ///
    /// Returns [`ModelCapabilitiesError::DecisionModelRejected`] when the model declares a
    /// decision type llama.cpp knows but cannot serve,
    /// [`ModelCapabilitiesError::PoolingProducesNoEmbeddingVectors`] when the explicit embedding
    /// pooling yields no vectors, and
    /// [`ModelCapabilitiesError::ChatTemplateCapabilitiesUnreadable`] when llama.cpp cannot read
    /// the chat templates.
    pub fn detect(
        ModelCapabilitiesSources {
            conversation_templates,
            embedding_pooling,
            model,
            projector,
        }: ModelCapabilitiesSources<'_>,
    ) -> Result<Self, ModelCapabilitiesError> {
        let projector_capabilities = projector_capabilities(projector);
        let text_generation = text_generation_of(model, projector_capabilities);

        Ok(Self {
            conversation: conversation_of(text_generation, conversation_templates)?,
            decisions: decisions_of(model)?,
            embeddings: embeddings_of(model, embedding_pooling)?,
            input_media: InputMediaCapabilities {
                audio: projector.is_some()
                    && projector_capabilities.contains(MtmdCapability::Audio),
                image: projector.is_some()
                    && projector_capabilities.contains(MtmdCapability::Vision),
            },
            model_defaults: ModelDefaults {
                context_size: model
                    .n_ctx_train()
                    .map_err(ModelCapabilitiesError::ContextSizeUnrepresentable)?,
                sampling: model.sampling_defaults(),
            },
            needs_classifier_free_guidance: model.classifier_free_guidance(),
            text_generation,
        })
    }

    /// `context_params` adjusted the way llama.cpp configures every context of this model.
    #[must_use]
    pub const fn context_params(&self, context_params: LlamaContextParams) -> LlamaContextParams {
        if !self.needs_classifier_free_guidance {
            return context_params;
        }

        let n_seq_max = context_params.n_seq_max();

        context_params
            .with_kv_unified(true)
            .with_n_seq_max(n_seq_max.saturating_mul(2))
    }

    #[must_use]
    pub fn report(&self) -> ModelCapabilities {
        ModelCapabilities {
            conversation: self.conversation,
            decisions: match &self.decisions {
                DetectedDecisions::Supported(decision_model) => {
                    DecisionsCapability::Supported(DecisionCapabilities {
                        reads_images: decision_model.can_use_images() && self.input_media.image,
                        shares_prompt_prefix: decision_model.shares_prompt_prefix(),
                    })
                }
                DetectedDecisions::Unsupported(reason) => DecisionsCapability::Unsupported(*reason),
            },
            embeddings: self.embeddings,
            input_media: self.input_media,
            model_defaults: self.model_defaults,
            text_generation: self.text_generation,
        }
    }
}
