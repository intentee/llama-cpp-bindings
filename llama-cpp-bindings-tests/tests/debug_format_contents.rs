use anyhow::Result;
use anyhow::ensure;

use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn context_debug_format_contains_struct_name(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?;
    let debug_output = format!("{context:?}");

    ensure!(debug_output.contains("LlamaContext"));

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
fn model_debug_format_includes_struct_name_and_model_field(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let formatted = format!("{model:?}");

    assert!(formatted.contains("LlamaModel"));
    assert!(formatted.contains("model"));
}
