use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use std::env;
use std::ffi::CString;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::path::PathBuf;

use llama_cpp_model::model_error::ModelError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn lora_adapter_init_with_a_missing_file_reports_it_not_found(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let result = model.lora_adapter_init("/nonexistent/path/lora.gguf");
    ensure!(
        result.err().context("the operation must fail")?
            == ModelError::LoraAdapterFileNotFound(PathBuf::from("/nonexistent/path/lora.gguf"))
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
fn lora_adapter_init_with_an_invalid_gguf_reports_it_unloadable(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let dummy_path = env::temp_dir().join("llama_test_dummy_lora.gguf");
    fs::write(&dummy_path, b"not a valid gguf")?;

    let result = model.lora_adapter_init(&dummy_path);

    ensure!(result.err().context("the operation must fail")? == ModelError::LoraAdapterUnloadable);
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
fn lora_adapter_init_refuses_a_path_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
    let result = model.lora_adapter_init(non_utf8_path);

    ensure!(
        result.err().context("the operation must fail")?
            == ModelError::LoraAdapterPathNotUtf8(non_utf8_path.to_path_buf())
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
fn lora_adapter_init_refuses_a_path_with_a_nul_byte(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        model.lora_adapter_init("/adapters/adapter\0.gguf").err(),
        CString::new("/adapters/adapter\0.gguf")
            .err()
            .map(ModelError::LoraAdapterPathContainsNulByte)
    );
}
