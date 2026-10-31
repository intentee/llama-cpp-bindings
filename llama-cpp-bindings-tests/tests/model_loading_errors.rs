use std::env;
use std::ffi::CString;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;

use llama_cpp_bindings::error::llama_lora_adapter_init_error::LlamaLoraAdapterInitError;
use llama_cpp_bindings::error::llama_model_load_error::LlamaModelLoadError;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn load_model_with_invalid_path_returns_error(fixture: &LlamaFixture<'_>) {
    let model_params = LlamaModelParams::default();
    let result =
        LlamaModel::load_from_file(fixture.backend, "/nonexistent/model.gguf", &model_params);

    assert!(matches!(
        result.unwrap_err(),
        LlamaModelLoadError::FileNotFound(path) if path == Path::new("/nonexistent/model.gguf"),
    ));
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn load_model_with_invalid_file_content_returns_unloadable_or_reported(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model_params = LlamaModelParams::default();
    let dummy_path = env::temp_dir().join("llama_test_invalid_model.gguf");
    fs::write(&dummy_path, b"not a valid gguf model file")?;

    let result = LlamaModel::load_from_file(fixture.backend, &dummy_path, &model_params);

    assert!(matches!(
        result.unwrap_err(),
        LlamaModelLoadError::Unloadable | LlamaModelLoadError::Reported { .. },
    ));
    let _ = fs::remove_file(&dummy_path);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn load_model_with_non_utf8_path_returns_path_to_str_error(fixture: &LlamaFixture<'_>) {
    let model_params = LlamaModelParams::default();
    let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));

    let result = LlamaModel::load_from_file(fixture.backend, non_utf8_path, &model_params);

    assert!(matches!(
        result.unwrap_err(),
        LlamaModelLoadError::PathToStrError(path) if path == non_utf8_path.to_path_buf()
    ));
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_with_invalid_path_returns_error(fixture: &LlamaFixture<'_>) {
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
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_with_invalid_gguf_returns_unloadable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let dummy_path = env::temp_dir().join("llama_test_dummy_lora.gguf");
    fs::write(&dummy_path, b"not a valid gguf")?;

    let result = fixture.model.lora_adapter_init(&dummy_path);

    assert_eq!(result.unwrap_err(), LlamaLoraAdapterInitError::Unloadable);
    let _ = fs::remove_file(&dummy_path);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_with_non_utf8_path_returns_error(fixture: &LlamaFixture<'_>) {
    let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
    let result = fixture.model.lora_adapter_init(non_utf8_path);

    assert_eq!(
        result.unwrap_err(),
        LlamaLoraAdapterInitError::PathToStrError(non_utf8_path.to_path_buf())
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn load_model_with_a_nul_byte_in_the_path_returns_error(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        LlamaModel::load_from_file(
            fixture.backend,
            "/models/model\0.gguf",
            &LlamaModelParams::default()
        )
        .err(),
        CString::new("/models/model\0.gguf")
            .err()
            .map(LlamaModelLoadError::NullError)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_with_a_nul_byte_in_the_path_returns_error(fixture: &LlamaFixture<'_>) {
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
