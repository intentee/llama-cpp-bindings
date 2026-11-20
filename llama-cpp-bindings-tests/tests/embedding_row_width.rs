use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::context_after_decoding_text::context_after_decoding_text;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const LAYA_OUTPUT_WIDTH: usize = 3;

const QWEN3_RERANKER_CLASSIFIER_LABELS: usize = 2;

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn token_embeddings_are_as_wide_as_the_model_output_rather_than_its_hidden_state(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::None)?;
    let row_width = context.embeddings_ith(0)?.len();

    ensure!(row_width == LAYA_OUTPUT_WIDTH);
    ensure!(row_width == usize::try_from(fixture.model.n_embd_out())?);
    ensure!(row_width != usize::try_from(fixture.model.n_embd())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn pooled_sequence_embeddings_are_as_wide_as_the_model_output_rather_than_its_hidden_state(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean)?;
    let row_width = context.embeddings_seq_ith(0)?.len();

    ensure!(row_width == LAYA_OUTPUT_WIDTH);
    ensure!(row_width == usize::try_from(fixture.model.n_embd_out())?);
    ensure!(row_width != usize::try_from(fixture.model.n_embd())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", "qwen3-reranker-0.6b-q8_0.gguf"),
    n_gpu_layers = -1,
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

    ensure!(context.embeddings_seq_ith(0)?.len() == QWEN3_RERANKER_CLASSIFIER_LABELS);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn a_sequence_embedding_is_as_wide_as_the_model_embedding(
    fixture @ &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let embeddings = context.embeddings_seq_ith(0)?;

    ensure!(embeddings.len() == usize::try_from(model.n_embd())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn a_token_embedding_is_as_wide_as_the_model_embedding(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::None),
    )?;
    let tokens = model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let last_index = i32::try_from(
        tokens
            .len()
            .checked_sub(1)
            .context("`tokens.len() - 1` must not overflow")?,
    )?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let embeddings = context.embeddings_ith(last_index)?;

    ensure!(embeddings.len() == usize::try_from(model.n_embd())?);

    Ok(())
}
