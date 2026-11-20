use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::freshly_loaded_model::freshly_loaded_model;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_model::llama_lora_adapter::LlamaLoraAdapter;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::lora_adapter_scale::LoraAdapterScale;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;
use llama_cpp_test_model_provisioning::download_model::download_model;

fn model_with_the_deepthink_reasoning_adapter(fixture: &LlamaFixture<'_>) -> Result<LlamaModel> {
    let adapter_path = download_model(
        "ggml-org/LoRA-Deepthink-Reasoning-Qwen2.5-7B-Instruct-Q8_0-GGUF",
        "Deepthink-Reasoning-Adapter-q8_0.gguf",
    )?;
    let mut model = freshly_loaded_model(fixture)?;

    model.load_lora_adapter(&adapter_path)?;

    Ok(model)
}

fn first_lora_adapter(model: &LlamaModel) -> Result<LlamaLoraAdapter<'_>> {
    model
        .lora_adapters()
        .first()
        .copied()
        .context("the loaded adapter must be listed")
}

#[llama_test(
    model_source = HuggingFace(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q2_k.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn set_lora_adapters_accepts_a_downloaded_adapter_at_full_scale(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_with_the_deepthink_reasoning_adapter(fixture)?;
    let adapter = first_lora_adapter(&model)?;
    let context = LlamaContext::from_model(
        &model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    ensure!(
        context.set_lora_adapters(&[LoraAdapterScale {
            adapter,
            scale: 1.0,
        }]) == Ok(())
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
fn set_lora_adapters_accepts_an_empty_adapter_set(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?;
    ensure!(context.set_lora_adapters(&[]) == Ok(()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q2_k.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn setting_lora_adapters_reports_a_status_llama_cpp_returns(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_with_the_deepthink_reasoning_adapter(fixture)?;
    let adapter = first_lora_adapter(&model)?;
    let context = LlamaContext::from_model(
        &model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_SET_ADAPTERS_LORA_FAILS,
    );

    ensure!(
        context.set_lora_adapters(&[LoraAdapterScale {
            adapter,
            scale: 1.0,
        }]) == Err(ContextError::SetLoraAdaptersFailed(-1))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q2_k.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn set_lora_adapters_refuses_an_adapter_loaded_into_another_model(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let another_model = model_with_the_deepthink_reasoning_adapter(fixture)?;
    let adapter = first_lora_adapter(&another_model)?;
    let context = fixture.build_context()?;

    ensure!(
        context.set_lora_adapters(&[LoraAdapterScale {
            adapter,
            scale: 1.0,
        }]) == Err(ContextError::LoraAdapterBelongsToAnotherModel { position: 0 })
    );

    Ok(())
}
