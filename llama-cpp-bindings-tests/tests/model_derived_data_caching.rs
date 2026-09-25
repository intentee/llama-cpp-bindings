use std::ptr;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn approximate_tok_env_is_cached_across_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let first = fixture.model.approximate_tok_env()?;
    let second = fixture.model.approximate_tok_env()?;

    assert!(Arc::ptr_eq(&first, &second));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn streaming_markers_are_cached_across_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let first = fixture.model.streaming_markers()?;
    let second = fixture.model.streaming_markers()?;

    assert!(Arc::ptr_eq(&first, &second));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn reasoning_markers_are_cached_across_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let first = fixture
        .model
        .reasoning_markers()?
        .context("Qwen3.5 must expose reasoning markers")?;
    let second = fixture
        .model
        .reasoning_markers()?
        .context("Qwen3.5 must expose reasoning markers")?;

    assert!(ptr::eq(first, second));

    Ok(())
}
