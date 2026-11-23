use std::num::NonZeroU8;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::two_sequence_context::two_sequence_context;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_context_memory::context_memory_error::ContextMemoryError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn an_encoder_context_keeps_no_memory(fixture: &LlamaFixture<'_>) -> Result<()> {
    ensure!(!fixture.build_context()?.has_memory());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn a_causal_embedding_context_keeps_memory(fixture: &LlamaFixture<'_>) -> Result<()> {
    ensure!(fixture.build_context()?.has_memory());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_threads_batch = 8,
    embeddings = true,
)]
fn memory_edits_without_a_memory_module_trivially_succeed(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = fixture.build_context()?;

    context.memory_clear();

    ensure!(context.memory_seq_keep(0) == Ok(()));
    ensure!(context.memory_seq_rm(Some(0), None, None) == Ok(()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_threads_batch = 8,
    embeddings = true,
)]
fn memory_operations_needing_a_memory_module_report_its_absence(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = two_sequence_context(fixture)?;
    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;

    ensure!(
        context.memory_seq_cp(0, 1, None, None)
            == Err(ContextMemoryError::SeqCpMemoryHandleUnavailable)
    );
    ensure!(
        context.memory_seq_add(0, None, None, 1)
            == Err(ContextMemoryError::SeqAddMemoryHandleUnavailable)
    );
    ensure!(
        context.memory_seq_div(0, None, None, divisor)
            == Err(ContextMemoryError::SeqDivMemoryHandleUnavailable)
    );
    ensure!(
        context.memory_seq_pos_max(0) == Err(ContextMemoryError::SeqPosMaxMemoryHandleUnavailable)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_threads_batch = 8,
    embeddings = true,
)]
fn removing_a_sequence_without_a_memory_module_still_validates_its_id(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = fixture.build_context()?;

    ensure!(
        context.memory_seq_rm(Some(u32::MAX), None, None).err()
            == i32::try_from(u32::MAX)
                .err()
                .map(ContextMemoryError::SeqRmSequenceIdTooLarge)
    );

    Ok(())
}
