use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn encode_on_non_encoder_model_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&batch);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn encode_succeeds_with_encoder_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&batch);

    assert!(result.is_ok());

    Ok(())
}
