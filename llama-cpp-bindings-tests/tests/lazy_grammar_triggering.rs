use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings_tests::logits_left_by::logits_left_by;
use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
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
fn a_lazy_grammar_awaiting_its_trigger_pattern_leaves_candidates_unconstrained(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let patterns = vec!["\\{.*".to_owned()];
    let sampler = LlamaSampler::grammar_lazy(model, "root ::= \"hello\"", "root", &patterns, &[])?;

    ensure!(
        logits_left_by(
            &sampler,
            &[single_token(model, "hello")?, single_token(model, "yes")?]
        )? == [0.0, 0.0]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn a_lazy_grammar_awaiting_its_trigger_token_leaves_candidates_unconstrained(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let sampler = LlamaSampler::grammar_lazy(
        model,
        "root ::= \"hello\"",
        "root",
        &[],
        &[single_token(model, "{")?],
    )?;

    ensure!(
        logits_left_by(
            &sampler,
            &[single_token(model, "hello")?, single_token(model, "yes")?]
        )? == [0.0, 0.0]
    );

    Ok(())
}
