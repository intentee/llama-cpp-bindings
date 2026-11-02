use std::num::NonZeroU8;

use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::error::memory_seq_add_error::MemorySeqAddError;
use llama_cpp_bindings::error::memory_seq_cp_error::MemorySeqCpError;
use llama_cpp_bindings::error::memory_seq_div_error::MemorySeqDivError;
use llama_cpp_bindings::error::memory_seq_pos_max_error::MemorySeqPosMaxError;
use llama_cpp_bindings::error::memory_seq_rm_error::MemorySeqRmError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn an_encoder_context_keeps_no_memory(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(!fixture.build_context()?.has_memory());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn a_causal_embedding_context_keeps_memory(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(fixture.build_context()?.has_memory());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
    let mut context = fixture.build_context()?;

    context.memory_clear();

    assert_eq!(context.memory_seq_keep(0), Ok(()));
    assert_eq!(context.memory_seq_rm(Some(0), None, None), Ok(()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
    let mut context = fixture.build_context()?;
    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;

    assert_eq!(
        context.memory_seq_cp(0, 1, None, None),
        Err(MemorySeqCpError::MemoryHandleUnavailable)
    );
    assert_eq!(
        context.memory_seq_add(0, None, None, 1),
        Err(MemorySeqAddError::MemoryHandleUnavailable)
    );
    assert_eq!(
        context.memory_seq_div(0, None, None, divisor),
        Err(MemorySeqDivError::MemoryHandleUnavailable)
    );
    assert_eq!(
        context.memory_seq_pos_max(0),
        Err(MemorySeqPosMaxError::MemoryHandleUnavailable)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
    let mut context = fixture.build_context()?;

    assert_eq!(
        context.memory_seq_rm(Some(u32::MAX), None, None).err(),
        i32::try_from(u32::MAX)
            .err()
            .map(MemorySeqRmError::SeqIdTooLarge)
    );

    Ok(())
}
