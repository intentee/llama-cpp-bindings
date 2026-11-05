use std::env;
use std::ffi::CString;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::error::llama_lora_adapter_init_error::LlamaLoraAdapterInitError;
use llama_cpp_bindings::error::llama_lora_adapters_error::LlamaLoraAdaptersError;
use llama_cpp_bindings::model::lora_adapter_scale::LoraAdapterScale;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;
use llama_cpp_test_model_provisioning::download_model::download_model;

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
fn set_lora_adapters_accepts_an_empty_adapter_set(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    assert_eq!(context.set_lora_adapters(&[]), Ok(()));

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_with_a_missing_file_reports_it_not_found(fixture: &LlamaFixture<'_>) {
    let result = fixture
        .model
        .lora_adapter_init("/nonexistent/path/lora.gguf");
    assert_eq!(
        result.unwrap_err(),
        LlamaLoraAdapterInitError::FileNotFound(PathBuf::from("/nonexistent/path/lora.gguf"))
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
fn lora_adapter_init_with_an_invalid_gguf_reports_it_unloadable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let dummy_path = env::temp_dir().join("llama_test_dummy_lora.gguf");
    fs::write(&dummy_path, b"not a valid gguf")?;

    let result = fixture.model.lora_adapter_init(&dummy_path);

    assert_eq!(result.unwrap_err(), LlamaLoraAdapterInitError::Unloadable);
    fs::remove_file(&dummy_path)?;
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_refuses_a_path_that_is_not_utf8(fixture: &LlamaFixture<'_>) {
    let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
    let result = fixture.model.lora_adapter_init(non_utf8_path);

    assert_eq!(
        result.unwrap_err(),
        LlamaLoraAdapterInitError::PathToStrError(non_utf8_path.to_path_buf())
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
fn lora_adapter_init_refuses_a_path_with_a_nul_byte(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        fixture
            .model
            .lora_adapter_init("/adapters/adapter\0.gguf")
            .err(),
        CString::new("/adapters/adapter\0.gguf")
            .err()
            .map(LlamaLoraAdapterInitError::NullError)
    );
}
