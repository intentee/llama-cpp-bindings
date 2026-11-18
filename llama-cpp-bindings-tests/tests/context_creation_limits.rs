use std::num::NonZeroU32;

use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_context_size_overflowing_llama_cpp_padding_is_unconstructible(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        fixture.build_context().err(),
        Some(ContextError::ContextUnconstructible)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 1,
    n_ubatch = 1,
)]
fn new_context_rejects_more_sequences_than_one_batch_can_output(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) {
    let context_load = LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_n_seq_max(2),
    );

    assert_eq!(
        context_load.err(),
        Some(ContextError::SequencesExceedOutputCapacity {
            n_seq_max: 2,
            output_capacity: 1,
        })
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 2,
    n_ubatch = 2,
)]
fn new_context_rejects_more_sequences_than_a_causal_context_can_output(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) {
    let context_load = LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_n_ctx(NonZeroU32::new(1))
            .with_n_seq_max(2),
    );

    assert_eq!(
        context_load.err(),
        Some(ContextError::SequencesExceedOutputCapacity {
            n_seq_max: 2,
            output_capacity: 1,
        })
    );
}
