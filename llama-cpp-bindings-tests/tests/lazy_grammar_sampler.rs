use std::ffi::CString;

use anyhow::Result;

use llama_cpp_bindings::error::grammar_error::GrammarError;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampling::LlamaSampler;
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
fn grammar_lazy_returns_sampler_for_valid_grammar_with_patterns(fixture: &LlamaFixture<'_>) {
    let patterns = vec!["\\{.*".to_owned()];
    let sampler =
        LlamaSampler::grammar_lazy(fixture.model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert!(sampler.is_ok());
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_with_root_not_found_returns_error(fixture: &LlamaFixture<'_>) {
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
fn grammar_lazy_with_null_byte_in_pattern_returns_error(fixture: &LlamaFixture<'_>) {
    let patterns = vec!["hel\0lo".to_owned()];
    let result =
        LlamaSampler::grammar_lazy(fixture.model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert_eq!(
        result.err(),
        Some(GrammarError::TriggerPatternContainsNul(
            CString::new("hel\0lo").expect_err("the pattern carries a nul byte")
        ))
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
fn grammar_lazy_returns_sampler_for_valid_grammar_with_trigger_tokens(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let trigger_tokens =
        fixture
            .model
            .str_to_token("{", AddBos::Never, ParseSpecialTokens::Always)?;

    assert!(
        !trigger_tokens.is_empty(),
        "the tokenizer must produce at least one trigger token"
    );

    let sampler = LlamaSampler::grammar_lazy(
        fixture.model,
        "root ::= \"hello\"",
        "root",
        &[],
        &trigger_tokens,
    );

    assert!(
        sampler.is_ok(),
        "trigger tokens alone must be enough to build a lazy grammar sampler"
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
fn grammar_lazy_with_null_byte_in_grammar_returns_error(fixture: &LlamaFixture<'_>) {
    let result =
        LlamaSampler::grammar_lazy(fixture.model, "root ::= \"hel\0lo\"", "root", &[], &[]);

    assert_eq!(
        result.err(),
        Some(GrammarError::GrammarContainsNul(
            CString::new("root ::= \"hel\0lo\"").expect_err("the grammar carries a nul byte")
        ))
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
