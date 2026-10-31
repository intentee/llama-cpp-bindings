use anyhow::Result;

use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn embedding_model_chat_template_renders_tool_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(fixture.model.chat_templates()?.supports_tool_calls());

    Ok(())
}
