use std::ffi::CString;
use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_context_params::MtmdContextParams;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_mtmd::mtmd_image_token_limit::MtmdImageTokenLimit;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn init_from_file_refuses_a_path_with_a_nul_byte(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let mtmd_params = MtmdContextParams::default();

    ensure!(
        MtmdContext::init_from_file("path\0null", Arc::clone(mtmd_ctx.model()), &mtmd_params).err()
            == CString::new("path\0null")
                .err()
                .map(MtmdError::MmprojPathContainsNul)
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
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn init_from_file_refuses_an_image_token_limit_a_c_int_cannot_hold(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let limit = MtmdImageTokenLimit::Tokens(NonZeroU32::MAX);
    let mtmd_params = MtmdContextParams {
        image_max_tokens: limit,
        ..MtmdContextParams::default()
    };

    ensure!(
        MtmdContext::init_from_file("mmproj.gguf", Arc::clone(mtmd_ctx.model()), &mtmd_params)
            .err()
            == i32::try_from(u32::MAX).err().map(|source| {
                MtmdError::InitContextParams(Box::new(MtmdError::ImageMaxTokensExceedCInt {
                    limit,
                    source,
                }))
            })
    );

    Ok(())
}
