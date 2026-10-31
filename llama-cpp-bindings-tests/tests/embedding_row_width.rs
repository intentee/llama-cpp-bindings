use anyhow::Result;

use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings_tests::context_after_decoding_text::context_after_decoding_text;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const LFM2_COLBERT_OUTPUT_PROJECTION_WIDTH: usize = 128;
const QWEN3_RERANKER_CLASSIFIER_LABELS: usize = 2;

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
    let context = context_after_decoding_text(fixture, LlamaPoolingType::None);

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
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean);

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
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Rank);

    assert_eq!(
        context.embeddings_seq_ith(0)?.len(),
        QWEN3_RERANKER_CLASSIFIER_LABELS
    );

    Ok(())
}
