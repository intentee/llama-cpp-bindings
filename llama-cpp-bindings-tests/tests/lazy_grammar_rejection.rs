use std::ffi::CString;

use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_sampler::sampler_error::SamplerError;
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
fn grammar_lazy_without_its_root_reports_the_missing_root(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let patterns = vec!["function".to_owned()];
    let result = LlamaSampler::grammar_lazy(model, "expr ::= \"hello\"", "root", &patterns, &[]);

    assert_eq!(result.err(), Some(SamplerError::RootNotFound));
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_refuses_a_trigger_pattern_with_a_nul_byte(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let patterns = vec!["hel\0lo".to_owned()];
    let result = LlamaSampler::grammar_lazy(model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert_eq!(
        result.err(),
        CString::new("hel\0lo")
            .err()
            .map(SamplerError::TriggerPatternContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_refuses_a_grammar_with_a_nul_byte(&LlamaFixture { model, .. }: &LlamaFixture<'_>) {
    let result = LlamaSampler::grammar_lazy(model, "root ::= \"hel\0lo\"", "root", &[], &[]);

    assert_eq!(
        result.err(),
        CString::new("root ::= \"hel\0lo\"")
            .err()
            .map(SamplerError::GrammarContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_with_malformed_regex_returns_invalid_trigger_pattern(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let patterns = vec!["[".to_owned()];
    let result = LlamaSampler::grammar_lazy(model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert!(matches!(
        result,
        Err(SamplerError::InvalidTriggerPattern { .. }),
    ));
}
