use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
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
fn synchronize_completes_without_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.synchronize();

    Ok(())
}
