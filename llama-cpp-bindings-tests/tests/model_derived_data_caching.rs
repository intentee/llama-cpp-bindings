use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;

use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn approximate_tok_env_is_cached_across_calls(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let first = model.approximate_tok_env()?;
    let second = model.approximate_tok_env()?;

    ensure!(Arc::ptr_eq(&first, &second));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn chat_templates_are_prepared_once_per_model(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let first = model.chat_templates()?;
    let second = model.chat_templates()?;

    ensure!(Arc::ptr_eq(&first, &second));

    Ok(())
}
