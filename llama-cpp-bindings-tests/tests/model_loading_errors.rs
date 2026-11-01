use std::env;
use std::ffi::CString;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;

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
fn load_model_with_a_missing_file_reports_it_not_found(fixture: &LlamaFixture<'_>) {
    let model_params = LlamaModelParams::default();
    let result =
        LlamaModel::load_from_file(fixture.backend, "/nonexistent/model.gguf", &model_params);

    assert_eq!(
        result.err(),
        Some(LlamaModelLoadError::FileNotFound(PathBuf::from(
            "/nonexistent/model.gguf"
        )))
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
fn load_model_with_invalid_file_content_reports_it_unloadable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model_params = LlamaModelParams::default();
    let dummy_path = env::temp_dir().join("llama_test_invalid_model.gguf");
    fs::write(&dummy_path, b"not a valid gguf model file")?;

    let result = LlamaModel::load_from_file(fixture.backend, &dummy_path, &model_params);

    assert_eq!(result.err(), Some(LlamaModelLoadError::Unloadable));
    fs::remove_file(&dummy_path)?;
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
fn load_model_refuses_a_path_that_is_not_utf8(fixture: &LlamaFixture<'_>) {
    let model_params = LlamaModelParams::default();
    let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));

    let result = LlamaModel::load_from_file(fixture.backend, non_utf8_path, &model_params);

    assert_eq!(
        result.err(),
        Some(LlamaModelLoadError::PathToStrError(
            non_utf8_path.to_path_buf()
        ))
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
fn load_model_refuses_a_path_with_a_nul_byte(fixture: &LlamaFixture<'_>) {
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
