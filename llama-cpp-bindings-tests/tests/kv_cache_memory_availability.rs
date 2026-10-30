use std::num::NonZeroU8;

use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::ClearKvCacheSeqError;
use llama_cpp_bindings::CopyKvCacheSeqError;
use llama_cpp_bindings::KvCacheSeqAddError;
use llama_cpp_bindings::KvCacheSeqDivError;
use llama_cpp_bindings::KvCacheSeqPosMaxError;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

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
fn kv_cache_operations_respect_an_embedding_context_without_memory(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    context.clear_kv_cache();
    assert_eq!(context.clear_kv_cache_seq(Some(0), None, None), Ok(()));
    context.kv_cache_seq_keep(0);

    assert_eq!(
        context.copy_kv_cache_seq(0, 1, None, None),
        Err(CopyKvCacheSeqError::MemoryHandleUnavailable)
    );
    assert_eq!(
        context.kv_cache_seq_add(0, None, None, 1),
        Err(KvCacheSeqAddError::MemoryHandleUnavailable)
    );
    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;
    assert_eq!(
        context.kv_cache_seq_div(0, None, None, divisor),
        Err(KvCacheSeqDivError::MemoryHandleUnavailable)
    );
    assert_eq!(
        context.kv_cache_seq_pos_max(0),
        Err(KvCacheSeqPosMaxError::MemoryHandleUnavailable)
    );
    assert!(matches!(
        context.clear_kv_cache_seq(Some(u32::MAX), None, None),
        Err(ClearKvCacheSeqError::SeqIdTooLarge(_))
    ));

    Ok(())
}
