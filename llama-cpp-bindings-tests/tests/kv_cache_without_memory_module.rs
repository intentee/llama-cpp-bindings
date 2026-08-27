use anyhow::Context;
use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

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
fn kv_cache_mutations_succeed_when_the_context_has_no_memory_module(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .context("unable to create context")?;

    ctx.clear_kv_cache()
        .context("clearing an absent KV cache must succeed")?;
    ctx.clear_kv_cache_seq(Some(0), None, None)
        .context("removing a sequence from an absent KV cache must succeed")?;
    ctx.kv_cache_seq_keep(0)
        .context("keeping a sequence in an absent KV cache must succeed")?;

    Ok(())
}
