use std::env;
use std::ffi::CString;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::freshly_loaded_model::freshly_loaded_model;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_path_c_string::path_c_string_error::PathCStringError;
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
fn load_lora_adapter_with_a_missing_file_reports_it_not_found(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut model = freshly_loaded_model(fixture)?;
    let result = model.load_lora_adapter("/nonexistent/path/lora.gguf");
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
fn load_lora_adapter_with_an_invalid_gguf_reports_it_unloadable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut model = freshly_loaded_model(fixture)?;
    let dummy_path = env::temp_dir().join("llama_test_dummy_lora.gguf");
    fs::write(&dummy_path, b"not a valid gguf")?;

    let result = model.load_lora_adapter(&dummy_path);

    ensure!(
        result.err().context("the operation must fail")?
            == ModelError::LoraAdapterUnloadable {
                path: dummy_path.clone()
            }
    );
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
fn load_lora_adapter_refuses_a_path_that_is_not_utf8(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut model = freshly_loaded_model(fixture)?;
    let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
    let result = model.load_lora_adapter(non_utf8_path);

    ensure!(
        result.err().context("the operation must fail")?
            == ModelError::LoraAdapterPathInvalid(PathCStringError::NotUtf8(
                non_utf8_path.to_path_buf()
            ))
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
fn load_lora_adapter_refuses_a_path_with_a_nul_byte(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut model = freshly_loaded_model(fixture)?;

    ensure!(
        model.load_lora_adapter("/adapters/adapter\0.gguf").err()
            == CString::new("/adapters/adapter\0.gguf")
                .err()
                .map(PathCStringError::ContainsNulByte)
                .map(ModelError::LoraAdapterPathInvalid)
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
fn load_lora_adapter_reports_an_exception_llama_cpp_throws_with_the_adapter_path(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut model = freshly_loaded_model(fixture)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_ADAPTER_LORA_INIT_THROWS,
    );

    ensure!(
        model.load_lora_adapter(fixture.model_path)
            == Err(ModelError::LoraAdapterReported {
                path: fixture.model_path.to_path_buf(),
                message: injected_fault_message()?,
            })
    );

    Ok(())
}
