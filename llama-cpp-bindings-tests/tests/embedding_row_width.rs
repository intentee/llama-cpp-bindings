use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::params::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings_types::BatchCapacity;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const LFM2_COLBERT_OUTPUT_PROJECTION_WIDTH: usize = 128;
const QWEN3_RERANKER_CLASSIFIER_LABELS: usize = 2;
const TEXT: &str = "The parcel arrived on the last working day of the week.";

fn context_after_decoding_text<'fixture>(
    fixture: &'fixture LlamaFixture<'_>,
    pooling_type: LlamaPoolingType,
) -> Result<LlamaContext<'fixture>> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(pooling_type),
    )?;
    let tokens = fixture
        .model
        .str_to_token(TEXT, AddBos::Always, ParseSpecialTokens::Never)?;
    let mut batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    Ok(context)
}

#[llama_test(
    model_source = HuggingFace("LiquidAI/LFM2-ColBERT-350M-GGUF", "LFM2-ColBERT-350M-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn token_embeddings_are_as_wide_as_the_output_projection(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::None)?;

    assert_eq!(
        context.embeddings_ith(0)?.len(),
        LFM2_COLBERT_OUTPUT_PROJECTION_WIDTH
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("LiquidAI/LFM2-ColBERT-350M-GGUF", "LFM2-ColBERT-350M-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn pooled_sequence_embeddings_are_as_wide_as_the_output_projection(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean)?;

    assert_eq!(
        context.embeddings_seq_ith(0)?.len(),
        LFM2_COLBERT_OUTPUT_PROJECTION_WIDTH
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", "qwen3-reranker-0.6b-q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn rank_pooled_sequence_embeddings_hold_one_score_per_classifier_label(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Rank)?;

    assert_eq!(
        context.embeddings_seq_ith(0)?.len(),
        QWEN3_RERANKER_CLASSIFIER_LABELS
    );

    Ok(())
}
