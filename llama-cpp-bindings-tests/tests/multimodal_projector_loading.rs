use std::ffi::CString;
use std::num::NonZeroU32;

use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;
use llama_cpp_bindings::mtmd::MtmdContextParamsError;
use llama_cpp_bindings::mtmd::MtmdImageTokenLimit;
use llama_cpp_bindings::mtmd::MtmdInitError;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn init_from_file_with_null_byte_in_path_returns_error(fixture: &LlamaFixture<'_>) {
    let mtmd_params = MtmdContextParams::default();

    assert_eq!(
        MtmdContext::init_from_file("path\0null", fixture.model, &mtmd_params).err(),
        CString::new("path\0null")
            .err()
            .map(MtmdInitError::CStringError)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn init_from_file_refuses_an_image_token_limit_a_c_int_cannot_hold(fixture: &LlamaFixture<'_>) {
    let limit = MtmdImageTokenLimit::Tokens(NonZeroU32::MAX);
    let mtmd_params = MtmdContextParams {
        image_max_tokens: limit,
        ..MtmdContextParams::default()
    };

    assert_eq!(
        MtmdContext::init_from_file("mmproj.gguf", fixture.model, &mtmd_params).err(),
        i32::try_from(u32::MAX).err().map(|source| {
            MtmdInitError::ContextParams(MtmdContextParamsError::ImageMaxTokensExceedCInt {
                limit,
                source,
            })
        })
    );
}
