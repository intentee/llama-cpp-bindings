use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings::error::embeddings_error::EmbeddingsError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn token_embedding_reports_embeddings_not_enabled_on_a_context_without_embeddings(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    assert_eq!(context.embeddings_ith(0), Err(EmbeddingsError::NotEnabled));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn sequence_embedding_reports_embeddings_not_enabled_on_a_context_without_embeddings(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    assert_eq!(
        context.embeddings_seq_ith(0),
        Err(EmbeddingsError::NotEnabled)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn sequence_embedding_of_a_sequence_missing_from_the_decoded_batch_is_unavailable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::Last),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    assert_eq!(
        context.embeddings_seq_ith(999),
        Err(EmbeddingsError::SequenceEmbeddingUnavailable {
            sequence_index: 999
        })
    );

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
fn token_embedding_of_an_index_without_a_decoded_output_names_no_output_row(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::None),
    )?;

    assert_eq!(
        context.embeddings_ith(999),
        Err(EmbeddingsError::TokenNotAnOutput { token_index: 999 })
    );

    Ok(())
}
