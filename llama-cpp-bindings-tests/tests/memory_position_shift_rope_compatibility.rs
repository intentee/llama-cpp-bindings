use std::num::NonZeroU8;

use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::error::memory_seq_add_error::MemorySeqAddError;
use llama_cpp_bindings::error::memory_seq_div_error::MemorySeqDivError;
use llama_cpp_bindings_tests::prime_memory::prime_memory;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_add_returns_error_for_mrope_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let result = context.memory_seq_add(0, Some(0), None, 1);

    assert_eq!(result.unwrap_err(), MemorySeqAddError::IncompatibleRopeType);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_div_returns_error_for_mrope_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;
    let result = context.memory_seq_div(0, Some(0), None, divisor);

    assert_eq!(result.unwrap_err(), MemorySeqDivError::IncompatibleRopeType);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_add_succeeds_on_embedding_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let result = context.memory_seq_add(0, Some(0), None, 1);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_div_succeeds_on_embedding_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;
    let result = context.memory_seq_div(0, Some(0), None, divisor);

    assert!(result.is_ok());

    Ok(())
}
