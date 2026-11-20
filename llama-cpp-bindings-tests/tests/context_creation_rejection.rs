use anyhow::Result;
use anyhow::ensure;

use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_params::params_error::ParamsError;
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
fn creating_a_context_reports_a_batch_llama_cpp_cannot_allocate(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_BATCH_EXT_INIT_THROWS,
    );

    ensure!(
        fixture.build_context().err()
            == Some(ContextError::CreateReported {
                message: injected_fault_message()?,
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_context_whose_pooling_type_llama_cpp_does_not_define_is_refused(fixture: &LlamaFixture<'_>) {
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_POOLING_TYPE_UNKNOWN);

    assert_eq!(
        fixture.build_context().err(),
        Some(ContextError::Params(ParamsError::UnknownPoolingType(
            llama_cpp_bindings_sys::bindings::LLAMA_POOLING_TYPE_RANK + 1
        )))
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
fn context_params_with_an_attention_type_llama_cpp_does_not_define_are_refused(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) {
    let mut params = (*context_params).into_llama_context_params();
    unsafe { (*params.as_mut_ffi()).attention_type = 99 };

    assert_eq!(
        LlamaContext::from_model(model, backend, params).err(),
        Some(ContextError::Params(ParamsError::UnknownAttentionType(99)))
    );
}
