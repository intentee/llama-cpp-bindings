use anyhow::Result;

use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_tests::logits_left_by::logits_left_by;
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
fn a_logit_bias_sampler_with_an_empty_bias_list_leaves_candidates_unchanged(
    _fixture: &LlamaFixture<'_>,
) -> Result<()> {
    assert_eq!(
        logits_left_by(&LlamaSampler::logit_bias(0, &[])?, &[LlamaToken::new(0)]),
        [0.0]
    );

    Ok(())
}
