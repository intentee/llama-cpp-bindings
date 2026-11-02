use std::num::NonZeroU32;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::error::llama_context_load_error::LlamaContextLoadError;
use llama_cpp_bindings::error::unknown_attention_type::UnknownAttentionType;
use llama_cpp_bindings::error::unknown_pooling_type::UnknownPoolingType;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4294967295,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_context_size_overflowing_llama_cpp_padding_is_unconstructible(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        fixture.build_context().err(),
        Some(LlamaContextLoadError::Unconstructible)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 1,
    n_ubatch = 1,
)]
fn new_context_rejects_more_sequences_than_one_batch_can_output(fixture: &LlamaFixture<'_>) {
    let context_load = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_n_seq_max(2),
    );

    assert_eq!(
        context_load.err(),
        Some(LlamaContextLoadError::SequencesExceedOutputCapacity {
            n_seq_max: 2,
            output_capacity: 1,
        })
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 2,
    n_ubatch = 2,
)]
fn new_context_rejects_more_sequences_than_a_causal_context_can_output(fixture: &LlamaFixture<'_>) {
    let context_load = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_n_ctx(NonZeroU32::new(1))
            .with_n_seq_max(2),
    );

    assert_eq!(
        context_load.err(),
        Some(LlamaContextLoadError::SequencesExceedOutputCapacity {
            n_seq_max: 2,
            output_capacity: 1,
        })
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
fn creating_a_context_reports_a_batch_llama_cpp_cannot_allocate(fixture: &LlamaFixture<'_>) {
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_BATCH_EXT_INIT_THROWS);

    assert_eq!(
        LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )
        .err(),
        Some(LlamaContextLoadError::Reported {
            message: injected_fault_message(),
        })
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_context_whose_pooling_type_llama_cpp_does_not_define_is_refused(fixture: &LlamaFixture<'_>) {
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_POOLING_TYPE_UNKNOWN);

    assert_eq!(
        fixture.build_context().err(),
        Some(LlamaContextLoadError::UnknownPoolingType(
            UnknownPoolingType(llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_RANK + 1)
        ))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn context_params_with_an_attention_type_llama_cpp_does_not_define_are_refused(
    fixture: &LlamaFixture<'_>,
) {
    let mut params = (*fixture.context_params).into_llama_context_params();
    params.context_params.attention_type = 99;

    assert_eq!(
        LlamaContext::from_model(fixture.model, fixture.backend, params).err(),
        Some(LlamaContextLoadError::UnknownAttentionType(
            UnknownAttentionType(99)
        ))
    );
}
