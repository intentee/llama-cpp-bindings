use std::ffi::CString;
use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::model_declaring_decision_type::model_declaring_decision_type;
use llama_cpp_bindings_tests::model_with_patched_metadata::model_with_patched_metadata;
use llama_cpp_bindings_types::conversation_capability::ConversationCapability;
use llama_cpp_bindings_types::conversation_unsupported_reason::ConversationUnsupportedReason;
use llama_cpp_bindings_types::decision_capabilities::DecisionCapabilities;
use llama_cpp_bindings_types::decisions_capability::DecisionsCapability;
use llama_cpp_bindings_types::decisions_unsupported_reason::DecisionsUnsupportedReason;
use llama_cpp_bindings_types::embeddings_capability::EmbeddingsCapability;
use llama_cpp_bindings_types::embeddings_unsupported_reason::EmbeddingsUnsupportedReason;
use llama_cpp_bindings_types::input_media_capabilities::InputMediaCapabilities;
use llama_cpp_bindings_types::model_capabilities::ModelCapabilities;
use llama_cpp_bindings_types::text_generation_capability::TextGenerationCapability;
use llama_cpp_bindings_types::text_generation_unsupported_reason::TextGenerationUnsupportedReason;
use llama_cpp_bindings_types::tool_calling_capability::ToolCallingCapability;
use llama_cpp_model::llama_chat_template::LlamaChatTemplate;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model_capabilities::conversation_templates::ConversationTemplates;
use llama_cpp_model_capabilities::detected_model_capabilities::DetectedModelCapabilities;
use llama_cpp_model_capabilities::embedding_pooling_choice::EmbeddingPoolingChoice;
use llama_cpp_model_capabilities::model_capabilities_error::ModelCapabilitiesError;
use llama_cpp_model_capabilities::model_capabilities_sources::ModelCapabilitiesSources;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_params::llama_context_params::LlamaContextParams;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn detected(
    model: &Arc<LlamaModel>,
    projector: Option<&MtmdContext>,
    embedding_pooling: EmbeddingPoolingChoice,
) -> Result<DetectedModelCapabilities, ModelCapabilitiesError> {
    let chat_templates = model.chat_templates();

    DetectedModelCapabilities::detect(ModelCapabilitiesSources {
        conversation_templates: chat_templates
            .as_ref()
            .map_or(ConversationTemplates::Absent, |templates| {
                ConversationTemplates::Present(templates)
            }),
        embedding_pooling,
        model,
        projector,
    })
}

fn detected_with_template(
    model: &Arc<LlamaModel>,
    template: &str,
) -> Result<Result<DetectedModelCapabilities, ModelCapabilitiesError>> {
    let chat_templates = model.chat_templates_with_override(&LlamaChatTemplate::new(template)?)?;

    Ok(DetectedModelCapabilities::detect(
        ModelCapabilitiesSources {
            conversation_templates: ConversationTemplates::Present(&chat_templates),
            embedding_pooling: EmbeddingPoolingChoice::ModelDefault,
            model,
            projector: None,
        },
    ))
}

fn reported(model: &Arc<LlamaModel>, projector: Option<&MtmdContext>) -> Result<ModelCapabilities> {
    Ok(detected(model, projector, EmbeddingPoolingChoice::ModelDefault)?.report())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_chat_model_converses_with_what_its_template_renders(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = reported(model, None)?;

    ensure!(
        matches!(
            capabilities.conversation,
            ConversationCapability::Supported(template)
                if template.thinking && template.tool_calling != ToolCallingCapability::Unsupported
        ),
        "capabilities were {capabilities:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_chat_model_declaring_no_pooling_serves_no_embeddings_by_default(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        reported(model, None)?.embeddings
            == EmbeddingsCapability::Unsupported(
                EmbeddingsUnsupportedReason::ModelDeclaresNoVectorPooling
            )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_explicit_vector_pooling_serves_embeddings_of_any_model(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        detected(
            model,
            None,
            EmbeddingPoolingChoice::Explicit(LlamaPoolingType::Mean)
        )?
        .report()
        .embeddings
            == EmbeddingsCapability::Supported
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_explicit_pooling_producing_no_vectors_is_a_configuration_error(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(matches!(
        detected(
            model,
            None,
            EmbeddingPoolingChoice::Explicit(LlamaPoolingType::Rank)
        ),
        Err(ModelCapabilitiesError::PoolingProducesNoEmbeddingVectors {
            pooling: LlamaPoolingType::Rank
        })
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_without_a_chat_template_does_not_converse(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = DetectedModelCapabilities::detect(ModelCapabilitiesSources {
        conversation_templates: ConversationTemplates::Absent,
        embedding_pooling: EmbeddingPoolingChoice::ModelDefault,
        model,
        projector: None,
    })?;

    ensure!(
        capabilities.conversation
            == ConversationCapability::Unsupported(
                ConversationUnsupportedReason::ModelHasNoChatTemplate
            )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_declaring_its_sampling_defaults_is_served_with_them(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_with_patched_metadata(
        fixture,
        "llama_test_model_declaring_sampling_defaults.gguf",
        |gguf| {
            gguf.set_val_str(c"general.sampling.temp", c"0.25");

            Ok(())
        },
    )?;

    ensure!(
        reported(&model, None)?
            .model_defaults
            .sampling
            .temperature
            .to_bits()
            == 0.25_f32.to_bits()
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_needing_classifier_free_guidance_gets_a_unified_cache_with_paired_sequences(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let guidance_key = CString::new(format!(
        "{}.classifier_free_guidance",
        fixture.model.meta_val_str("general.architecture")?
    ))?;
    let model = model_with_patched_metadata(
        fixture,
        "llama_test_model_needing_classifier_free_guidance.gguf",
        |gguf| {
            gguf.set_val_bool(&guidance_key, true);

            Ok(())
        },
    )?;
    let context_params = detected(&model, None, EmbeddingPoolingChoice::ModelDefault)?
        .context_params(LlamaContextParams::default().with_n_seq_max(3));

    ensure!(context_params.kv_unified() && context_params.n_seq_max() == 6);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_decision_model_llama_cpp_cannot_serve_fails_detection(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_declaring_decision_type(
        fixture,
        c"kev",
        "llama_test_capabilities_of_a_decision_model_without_template.gguf",
    )?;

    ensure!(matches!(
        detected(&model, None, EmbeddingPoolingChoice::ModelDefault),
        Err(ModelCapabilitiesError::DecisionModelRejected(_))
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn a_projector_that_sees_images_takes_image_input(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        reported(model, mtmd_context)?.input_media
            == InputMediaCapabilities {
                audio: false,
                image: true,
            }
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
fn a_projector_that_hears_audio_takes_audio_input(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        reported(model, mtmd_context)?.input_media
            == InputMediaCapabilities {
                audio: true,
                image: false,
            }
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn an_embedding_model_serves_embeddings_with_its_declared_pooling(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(reported(model, None)?.embeddings == EmbeddingsCapability::Supported);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q2_K.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_non_causal_model_generates_no_text(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = reported(model, None)?;

    ensure!(
        capabilities.text_generation
            == TextGenerationCapability::Unsupported(
                TextGenerationUnsupportedReason::AttentionIsNotCausal
            )
            && capabilities.conversation
                == ConversationCapability::Unsupported(
                    ConversationUnsupportedReason::ModelGeneratesNoText
                ),
        "capabilities were {capabilities:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_encoder_decoder_model_generates_no_text(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        reported(model, None)?.text_generation
            == TextGenerationCapability::Unsupported(
                TextGenerationUnsupportedReason::EncoderDecoderArchitecture
            )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Kev-0.8B-GGUF", "Kev-0.8B-BF16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_decision_model_serves_decisions_and_generates_no_text(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = reported(model, None)?;

    ensure!(
        matches!(capabilities.decisions, DecisionsCapability::Supported(_))
            && capabilities.text_generation
                == TextGenerationCapability::Unsupported(
                    TextGenerationUnsupportedReason::NoTextOutput
                ),
        "capabilities were {capabilities:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_declaring_a_decision_type_llama_cpp_does_not_know_serves_no_decisions_but_its_text(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_declaring_decision_type(
        fixture,
        c"not-a-decision-type",
        "llama_test_capabilities_of_an_unknown_decision_type.gguf",
    )?;
    let capabilities = reported(&model, None)?;

    ensure!(
        capabilities.decisions
            == DecisionsCapability::Unsupported(
                DecisionsUnsupportedReason::DecisionTypeUnsupportedByLlamaCpp
            )
            && capabilities.text_generation == TextGenerationCapability::Supported,
        "capabilities were {capabilities:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_template_rendering_one_tool_call_per_turn_calls_tools_one_at_a_time(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = detected_with_template(
        model,
        "{%- if tools %}{{ tools[0].function.name }}\n{% endif -%}\
         {%- for message in messages -%}\
         {{ message.role }}: {{ message.content }}\
         {%- if message.tool_calls %} {{ message.tool_calls[0].function.name }}{% endif %}\n\
         {% endfor -%}\
         {%- if add_generation_prompt %}assistant: {% endif -%}",
    )??
    .report();

    ensure!(
        matches!(
            capabilities.conversation,
            ConversationCapability::Supported(template)
                if template.tool_calling == ToolCallingCapability::OneCallPerTurn
        ),
        "capabilities were {capabilities:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_template_that_never_renders_tools_calls_no_tools(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = detected_with_template(
        model,
        "{%- for message in messages -%}{{ message.role }}: {{ message.content }}\n{% endfor -%}\
         {%- if add_generation_prompt %}assistant: {% endif -%}",
    )??
    .report();

    ensure!(
        matches!(
            capabilities.conversation,
            ConversationCapability::Supported(template)
                if template.tool_calling == ToolCallingCapability::Unsupported
        ),
        "capabilities were {capabilities:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_template_llama_cpp_cannot_probe_fails_detection(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(matches!(
        detected_with_template(
            model,
            "{% if messages[0]['role'] != 'system' %}\
             {{ raise_exception('the first message must be the system message') }}\
             {% endif %}{{ messages[0]['content'] }}",
        )?,
        Err(ModelCapabilitiesError::ChatTemplateCapabilitiesUnreadable(
            _
        ))
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_declaring_a_pooling_type_llama_cpp_does_not_know_fails_detection(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let pooling_key = CString::new(format!(
        "{}.pooling_type",
        fixture.model.meta_val_str("general.architecture")?
    ))?;
    let model = model_with_patched_metadata(
        fixture,
        "llama_test_model_declaring_an_unknown_pooling_type.gguf",
        |gguf| {
            gguf.set_val_u32(&pooling_key, 99);

            Ok(())
        },
    )?;

    ensure!(matches!(
        detected(&model, None, EmbeddingPoolingChoice::ModelDefault),
        Err(ModelCapabilitiesError::DeclaredPoolingTypeUnknown(_))
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_whose_training_context_llama_cpp_cannot_report_fails_detection(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context_length_key = CString::new(format!(
        "{}.context_length",
        fixture.model.meta_val_str("general.architecture")?
    ))?;
    let model = model_with_patched_metadata(
        fixture,
        "llama_test_model_declaring_an_unreportable_context_length.gguf",
        |gguf| {
            gguf.set_val_u32(&context_length_key, u32::MAX);

            Ok(())
        },
    )?;

    ensure!(matches!(
        detected(&model, None, EmbeddingPoolingChoice::ModelDefault),
        Err(ModelCapabilitiesError::ContextSizeUnrepresentable(_))
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_needing_no_guidance_keeps_its_context_parameters(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let context_params = detected(model, None, EmbeddingPoolingChoice::ModelDefault)?
        .context_params(LlamaContextParams::default().with_n_seq_max(3));

    ensure!(!context_params.kv_unified() && context_params.n_seq_max() == 3);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("openbmb/MiniCPM-OCR-Jev-gguf", "MiniCPM-OCR-Jev-IQ3_XS.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 4096,
    n_ubatch = 4096,
    mmproj_source = HuggingFace("openbmb/MiniCPM-OCR-Jev-gguf", "mmproj-model-f16.gguf"),
)]
fn a_decision_model_with_a_projector_that_sees_reads_images(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let capabilities = reported(model, mtmd_context)?;

    ensure!(
        capabilities.decisions
            == DecisionsCapability::Supported(DecisionCapabilities { reads_images: true }),
        "capabilities were {capabilities:?}"
    );

    Ok(())
}
