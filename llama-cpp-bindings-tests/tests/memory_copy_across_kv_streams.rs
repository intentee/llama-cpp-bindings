use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::two_stream_context::two_stream_context;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_context_memory::context_memory_error::ContextMemoryError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_partial_copy_between_sequences_with_separate_kv_streams_is_refused(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = two_stream_context(fixture)?;

    ensure!(
        context.memory_seq_cp(0, 1, Some(0), Some(1))
            == Err(ContextMemoryError::SeqCpPartialCopyAcrossKvStreams {
                src: 0,
                dest: 1,
                p0: 0,
                p1: 1,
                cache_size: context.n_ctx_seq(),
            })
    );

    Ok(())
}
