use std::ffi::CString;

use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_gbnf::gbnf_error::GbnfError;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_grammar_without_its_root_reports_the_missing_root(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        LlamaSampler::grammar(model, r#"answer ::= <think> "x""#, "root").err(),
        Some(SamplerError::RootNotFound)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_grammar_with_a_syntax_error_reports_it(&LlamaFixture { model, .. }: &LlamaFixture<'_>) {
    assert_eq!(
        LlamaSampler::grammar(model, "root ::= (<think>", "root").err(),
        Some(SamplerError::GrammarRejected(GbnfError::SyntaxError))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_grammar_with_a_nul_byte_is_refused(&LlamaFixture { model, .. }: &LlamaFixture<'_>) {
    assert_eq!(
        LlamaSampler::grammar(model, "root ::= \"a\0\"", "root").err(),
        CString::new("root ::= \"a\0\"")
            .err()
            .map(SamplerError::GrammarContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_valid_grammar_llama_cpp_refuses_to_load_is_reported_malformed(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_SAMPLER_INIT_GRAMMAR_RETURNS_NULL,
    );

    assert_eq!(
        LlamaSampler::grammar(model, "root ::= \"a\"", "root").err(),
        Some(SamplerError::GrammarMalformed)
    );
}
