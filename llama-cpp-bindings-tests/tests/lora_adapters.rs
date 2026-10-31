use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::error::llama_lora_adapters_error::LlamaLoraAdaptersError;
use llama_cpp_bindings::model::lora_adapter_scale::LoraAdapterScale;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::download_model::download_model;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q2_k.gguf"
    ),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn set_lora_adapters_applies_a_real_adapter(fixture: &LlamaFixture<'_>) -> Result<()> {
    let adapter_path = download_model(
        "ggml-org/LoRA-Deepthink-Reasoning-Qwen2.5-7B-Instruct-Q8_0-GGUF",
        "Deepthink-Reasoning-Adapter-q8_0.gguf",
    )?;
    let adapter = fixture.model.lora_adapter_init(&adapter_path)?;
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    assert_eq!(
        context.set_lora_adapters(&[LoraAdapterScale {
            adapter: &adapter,
            scale: 1.0,
        }]),
        Ok(())
    );

    assert_eq!(context.set_lora_adapters(&[]), Ok(()));

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
fn set_lora_adapters_accepts_an_empty_adapter_set(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let result = context.set_lora_adapters(&[]);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q2_k.gguf"
    ),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn setting_lora_adapters_reports_a_status_llama_cpp_returns(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let adapter_path = download_model(
        "ggml-org/LoRA-Deepthink-Reasoning-Qwen2.5-7B-Instruct-Q8_0-GGUF",
        "Deepthink-Reasoning-Adapter-q8_0.gguf",
    )?;
    let adapter = fixture.model.lora_adapter_init(&adapter_path)?;
    let context = fixture.build_context()?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SET_ADAPTERS_LORA_FAILS);

    assert_eq!(
        context.set_lora_adapters(&[LoraAdapterScale {
            adapter: &adapter,
            scale: 1.0,
        }]),
        Err(LlamaLoraAdaptersError::ErrorResult(-1))
    );

    Ok(())
}
