use llama_cpp_backend::backend_error::BackendError;
use llama_cpp_backend::llama_backend_device::LlamaBackendDevice;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn listing_with_missing_text_after(passing_hits: usize) -> Option<BackendError> {
    let _fault = ArmedFault::after_passing(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_BACKEND_DEVICE_TEXT_NULL,
        passing_hits,
    );

    LlamaBackendDevice::list_all().err()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_device_text_ggml_does_not_report_refuses_the_listing(_fixture: &LlamaFixture<'_>) {
    assert_eq!(
        [0, 1, 2].map(listing_with_missing_text_after),
        [
            Some(BackendError::DeviceNameMissing),
            Some(BackendError::DeviceDescriptionMissing),
            Some(BackendError::DeviceBackendNameMissing),
        ]
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
fn a_device_type_ggml_does_not_define_refuses_the_listing(_fixture: &LlamaFixture<'_>) {
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_BACKEND_DEVICE_TYPE_UNKNOWN,
    );

    assert_eq!(
        LlamaBackendDevice::list_all().err(),
        Some(BackendError::UnknownDeviceType(
            llama_cpp_bindings_sys::bindings::GGML_BACKEND_DEVICE_TYPE_META + 1
        ))
    );
}
