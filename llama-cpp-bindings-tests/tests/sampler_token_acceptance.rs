use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
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
fn accept_many_iterates_over_borrowed_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::chain([LlamaSampler::greedy()?])?;
    let tokens = vec![
        fixture
            .model
            .token_bos()
            .context("model declares a BOS token")?,
        fixture
            .model
            .token_eos()
            .context("model declares an EOS token")?,
    ];

    sampler.accept_many(&tokens)?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn with_tokens_returns_self_after_accepting_each_token(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = LlamaSampler::chain([LlamaSampler::greedy()?])?;
    let tokens = [
        fixture
            .model
            .token_bos()
            .context("model declares a BOS token")?,
        fixture
            .model
            .token_eos()
            .context("model declares an EOS token")?,
    ];

    let _consumed = sampler.with_tokens(tokens.iter().copied())?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn accept_consumes_a_single_token(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::chain([LlamaSampler::greedy()?])?;

    sampler.accept(
        fixture
            .model
            .token_bos()
            .context("model declares a BOS token")?,
    )?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn accept_succeeds_for_a_valid_token(_fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::chain([LlamaSampler::greedy()?])?;

    sampler.accept(LlamaToken::new(0))?;

    Ok(())
}
