use std::ffi::CString;

use anyhow::Result;

use llama_cpp_bindings::error::grammar_error::GrammarError;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::logits_left_by::logits_left_by;
use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn a_lazy_grammar_awaiting_its_trigger_pattern_leaves_candidates_unconstrained(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let patterns = vec!["\\{.*".to_owned()];
    let sampler = LlamaSampler::grammar_lazy(model, "root ::= \"hello\"", "root", &patterns, &[])?;

    assert_eq!(
        logits_left_by(
            &sampler,
            &[single_token(model, "hello"), single_token(model, "yes")]
        ),
        [0.0, 0.0]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_without_its_root_reports_the_missing_root(fixture: &LlamaFixture<'_>) {
    let patterns = vec!["function".to_owned()];
    let result =
        LlamaSampler::grammar_lazy(fixture.model, "expr ::= \"hello\"", "root", &patterns, &[]);

    assert_eq!(result.err(), Some(GrammarError::RootNotFound));
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_refuses_a_trigger_pattern_with_a_nul_byte(fixture: &LlamaFixture<'_>) {
    let patterns = vec!["hel\0lo".to_owned()];
    let result =
        LlamaSampler::grammar_lazy(fixture.model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert_eq!(
        result.err(),
        CString::new("hel\0lo")
            .err()
            .map(GrammarError::TriggerPatternContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn a_lazy_grammar_awaiting_its_trigger_token_leaves_candidates_unconstrained(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let sampler = LlamaSampler::grammar_lazy(
        model,
        "root ::= \"hello\"",
        "root",
        &[],
        &[single_token(model, "{")],
    )?;

    assert_eq!(
        logits_left_by(
            &sampler,
            &[single_token(model, "hello"), single_token(model, "yes")]
        ),
        [0.0, 0.0]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_refuses_a_grammar_with_a_nul_byte(fixture: &LlamaFixture<'_>) {
    let result =
        LlamaSampler::grammar_lazy(fixture.model, "root ::= \"hel\0lo\"", "root", &[], &[]);

    assert_eq!(
        result.err(),
        CString::new("root ::= \"hel\0lo\"")
            .err()
            .map(GrammarError::GrammarContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_with_malformed_regex_returns_invalid_trigger_pattern(fixture: &LlamaFixture<'_>) {
    let patterns = vec!["[".to_owned()];
    let result =
        LlamaSampler::grammar_lazy(fixture.model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert!(matches!(
        result,
        Err(GrammarError::InvalidTriggerPattern { .. }),
    ));
}
