use llama_cpp_bindings::sampling::LlamaSampler;
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
fn logit_bias_with_empty_biases_succeeds(_fixture: &LlamaFixture<'_>) {
    let result = LlamaSampler::logit_bias(0, &[]);

    assert!(result.is_ok());
}
